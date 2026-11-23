import argparse
import hashlib
import json
import subprocess
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import support


ROOT = Path(__file__).resolve().parent
TRAIN_FILES = (
    ROOT / "trainrawyuv/BSD500train.yuv",
    ROOT / "traincompyuv/BSD500trainc.yuv",
)
TEST_FILES = (
    ROOT / "testrawyuv/BSD500val.yuv",
    ROOT / "testcompyuv/BSD500valc.yuv",
)
BASELINE_COMMIT = "af78856948403c3cce7705eb2d7f52f6f44a6824"
BASELINE_SHA256 = "0d62d7bf3ffc3ecc42420b61caadb9f32520b1925a8662ab5901c8e8eaa67d6f"
RUNS = ROOT / "runs"

VARIANTS = {"control": (1, 1), "wide": (2, 4)}
# Радиус наибольшего поля зрения. Считаем MSE внутри фрагмента,
# чтобы модель не училась предсказывать по искусственным границам вырезки.
LOSS_BORDER = 16


class EnhancerModel(nn.Module):
    def __init__(self, variant="baseline"):
        super().__init__()

        # Имена и размеры слоев базовой QECNN сохранены для готовых весов.
        self.conv1 = nn.Conv2d(3, 128, 9, padding=4)
        self.prelu1 = nn.PReLU(128)
        self.conv11 = nn.Conv2d(3, 128, 9, padding=4)
        self.prelu11 = nn.PReLU(128)

        self.conv2 = nn.Conv2d(128, 64, 7, padding=3)
        self.prelu2 = nn.PReLU(64)
        self.conv22 = nn.Conv2d(256, 64, 7, padding=3)
        self.prelu22 = nn.PReLU(64)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.prelu3 = nn.PReLU(64)
        self.conv33 = nn.Conv2d(128, 64, 3, padding=1)
        self.prelu33 = nn.PReLU(64)

        self.conv4 = nn.Conv2d(64, 32, 1)
        self.prelu4 = nn.PReLU(32)
        self.conv44 = nn.Conv2d(128, 32, 1)
        self.prelu44 = nn.PReLU(32)
        self.conv_out = nn.Conv2d(64, 3, 5, padding=2)

        self.context = None
        if variant != "baseline":
            first, second = VARIANTS[variant]
            self.context = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=first, dilation=first),
                nn.PReLU(64),
                nn.Conv2d(64, 64, 3, padding=second, dilation=second),
            )

    def forward(self, image):
        c1 = self.prelu1(self.conv1(image))
        c11 = self.prelu11(self.conv11(image))
        feat1 = torch.cat([c1, c11], dim=1)

        c2 = self.prelu2(self.conv2(c1))
        c22 = self.prelu22(self.conv22(feat1))
        feat2 = torch.cat([c2, c22], dim=1)

        c3 = self.prelu3(self.conv3(c2))
        c33 = self.prelu33(self.conv33(feat2))
        feat3 = torch.cat([c3, c33], dim=1)

        c4 = self.prelu4(self.conv4(c3))
        c44 = self.prelu44(self.conv44(feat3))
        feat4 = torch.cat([c4, c44], dim=1)

        if self.context is not None:
            feat4 = feat4 + self.context(feat4)
        # Сеть предсказывает поправку к уже декодированному RGB.
        return image + self.conv_out(feat4)


class TrainingPatches(Dataset):
    def __init__(self, original, decoded, indices, size, patches_per_image, seed, balance_images):
        self.size = size
        self.patches_per_image = patches_per_image
        self.rng = np.random.default_rng(seed)

        # Переводим обучающие кадры из YUV один раз и держим в памяти как float32.
        shape = (len(indices), 3, support.HEIGHT, support.WIDTH)
        self.original = np.empty(shape, dtype=np.float32)
        self.decoded = np.empty(shape, dtype=np.float32)
        self.weights = np.ones(len(indices), dtype=np.float32)
        for position, index in enumerate(indices):
            self.original[position] = original[index].transpose(2, 0, 1)
            self.decoded[position] = decoded[index].transpose(2, 0, 1)
            if balance_images:
                # Примерно одинаковый вес относительной ошибки каждого кадра.
                mse = np.mean((self.original[position] - self.decoded[position]) ** 2)
                self.weights[position] = 1 / max(mse, 1e-3)
        self.weights /= self.weights.mean()

    def __len__(self):
        return len(self.original) * self.patches_per_image

    def __getitem__(self, index):
        image = index // self.patches_per_image
        top = self.rng.integers(support.HEIGHT - self.size + 1)
        left = self.rng.integers(support.WIDTH - self.size + 1)
        rows = slice(top, top + self.size)
        columns = slice(left, left + self.size)

        # Одинаковые координаты в сжатом кадре и в оригинале.
        return (
            self.decoded[image, :, rows, columns],
            self.original[image, :, rows, columns],
            self.weights[image],
        )


@torch.inference_mode()
def predict(model, rgb, device, reflected=False):
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    batch = torch.from_numpy(chw).unsqueeze(0).to(device)
    restored = restore_reflected(model, batch) if reflected else model(batch)
    return restored[0].permute(1, 2, 0).cpu().numpy()


def restore_reflected(model, image):
    border = LOSS_BORDER
    padded = nn.functional.pad(image, (border, border, border, border), mode="reflect")
    return model(padded)[..., border:-border, border:-border]


class SymmetryEnsemble(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        total = torch.zeros_like(image)
        for turns in range(4):
            rotated = torch.rot90(image, turns, dims=(2, 3))
            for mirror in (False, True):
                view = rotated.flip(dims=(3,)) if mirror else rotated
                restored = restore_reflected(self.model, view)
                if mirror:
                    restored = restored.flip(dims=(3,))
                total += torch.rot90(restored, -turns, dims=(2, 3))
        return (total / 8).clamp(0, 1)


def validation_psnr(model, original, decoded, indices, device, reflected=False):
    model.eval()
    scores = []
    for index in indices:
        restored = predict(model, decoded[index], device, reflected=reflected)
        if reflected:
            restored = np.clip(restored, 0, 1)
        scores.append(support.cal_psnr(original[index], restored))
    return float(np.mean(scores))


def checkpoint_path(variant):
    return RUNS / variant / "best.pth"


def save_checkpoint(path, state_dict, variant, epoch, val_psnr, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary_path = Path(file.name)
    try:
        torch.save(
            {
                "variant": variant,
                "state_dict": state_dict,
                "epoch": epoch,
                "validation_psnr_db": val_psnr,
                "seed": seed,
            },
            temporary_path,
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def train(args, device):
    torch.manual_seed(args.seed)
    refining = args.initialize is not None
    original, decoded = support.open_pair(*TRAIN_FILES)
    if len(original) < 2 and not args.all_images:
        raise ValueError("Training requires at least one training and one validation image")
    indices = np.random.default_rng(args.seed).permutation(len(original))
    # При подборе настроек откладываем целые кадры до нарезки фрагментов.
    # Финальное обучение использует все исходные кадры и фиксированное число эпох.
    val_count = 0 if args.all_images else max(1, len(indices) // 5)
    train_indices, val_indices = indices[val_count:], indices[:val_count]
    print(f"Training images: {len(train_indices)} | Validation images: {len(val_indices)}")

    patches = TrainingPatches(
        original, decoded, train_indices, args.patch_size, args.patches_per_image, args.seed,
        balance_images=refining,
    )
    loader = DataLoader(
        patches,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == "cuda",
    )
    if refining:
        model = load_model(args.variant, args.initialize, device, expected_seed=args.seed)
    else:
        model = EnhancerModel(args.variant).to(device)

    use_amp = refining and device.type == "cuda"
    if use_amp:
        model = model.to(memory_format=torch.channels_last)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 if refining else 2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=2e-5
    )
    center = slice(LOSS_BORDER, -LOSS_BORDER)
    run_name = f"{args.variant}_refined" if refining else args.variant
    if args.all_images:
        run_name = f"{args.variant}_all"
    output = args.output or checkpoint_path(run_name)
    best_psnr = float("-inf")
    best_epoch = 0
    best_weights = None
    if refining and val_count:
        best_psnr = validation_psnr(model, original, decoded, val_indices, device, reflected=True)
        print(f"Before fine-tuning | validation PSNR {best_psnr:.3f} dB", flush=True)
        best_weights = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    augmentation = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for compressed, target, weights in loader:
            compressed = compressed.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            if refining:
                turns = int(augmentation.integers(4))
                compressed = torch.rot90(compressed, turns, dims=(2, 3))
                target = torch.rot90(target, turns, dims=(2, 3))
                if augmentation.integers(2):
                    compressed = compressed.flip(dims=(3,))
                    target = target.flip(dims=(3,))
            if use_amp:
                compressed = compressed.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad()
            with torch.autocast(device.type, enabled=use_amp):
                restored = model(compressed)
            restored_center = restored[:, :, center, center]
            target_center = target[:, :, center, center]
            if refining:
                mse = (restored_center - target_center).square().mean(dim=(1, 2, 3))
                loss = (mse * weights).mean()
            else:
                loss = nn.functional.mse_loss(restored_center, target_center)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(compressed)

        epoch_log = f"Epoch {epoch:2}/{args.epochs} | train loss {train_loss / len(patches):.6f}"
        if val_count:
            score = validation_psnr(model, original, decoded, val_indices, device, reflected=refining)
            print(f"{epoch_log} | validation PSNR {score:.3f} dB", flush=True)
            if score > best_psnr:
                best_psnr = score
                best_epoch = epoch
                best_weights = {
                    name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
                }
        else:
            print(epoch_log, flush=True)
        scheduler.step()
    if args.all_images:
        save_checkpoint(output, model.state_dict(), args.variant, args.epochs, None, args.seed)
        print(f"Saved final weights after {args.epochs} fixed epochs to {output}", flush=True)
    else:
        if best_weights is None:
            raise ValueError("Training finished without a valid validation score")
        # Батник считает этап завершённым только после записи этого файла.
        save_checkpoint(output, best_weights, args.variant, best_epoch, best_psnr, args.seed)
        print(f"Saved best weights from epoch {best_epoch} to {output}", flush=True)


def read_baseline_weights():
    # Эталон читается из истории Git только для итогового сравнения.
    reference = f"{BASELINE_COMMIT}:best_model2026.pth"
    try:
        weights = subprocess.check_output(["git", "show", reference], cwd=ROOT, stderr=subprocess.PIPE)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "The original QECNN benchmark is unavailable in local Git history. "
            "See the reference setup in README.md."
        ) from error
    if hashlib.sha256(weights).hexdigest() != BASELINE_SHA256:
        raise ValueError("The original QECNN benchmark failed its SHA-256 check")
    return weights


def load_model(variant, weights, device, expected_seed=None):
    if variant == "baseline":
        weights = BytesIO(read_baseline_weights())
    elif not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    model = EnhancerModel(variant).to(device)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if variant != "baseline":
        if state.get("variant") != variant:
            raise ValueError(f"{weights} does not contain weights for the {variant} model")
        if expected_seed is not None and state.get("seed", 42) != expected_seed:
            raise ValueError("Fine-tuning must use the same split seed as the initial checkpoint")
        state = state["state_dict"]
    model.load_state_dict(state)
    return model.eval()


def evaluate(args, device):
    models = {"QECNN": load_model("baseline", None, device)}
    proposed = None
    if args.command == "evaluate":
        if args.control is not None:
            models["QECNN control"] = load_model("control", args.control, device)
        if args.improved:
            weights = args.wide or checkpoint_path("wide_larger_patches")
            model = load_model("wide", weights, device)
            proposed = "QECNN improved"
            models[proposed] = SymmetryEnsemble(model).eval()
        else:
            weights = args.wide or checkpoint_path("wide")
            proposed = "QECNN wide"
            models[proposed] = load_model("wide", weights, device)

    original, decoded = support.open_pair(*TEST_FILES, expected_frames=100)
    scores = {"x265": []}
    scores.update({name: [] for name in models})
    for index in range(len(original)):
        target = original[index]
        compressed = decoded[index]
        scores["x265"].append(support.cal_psnr(target, compressed))
        for name, model in models.items():
            restored = predict(model, compressed, device)
            scores[name].append(support.cal_psnr(target, restored))
        if (index + 1) % 25 == 0:
            print(f"Evaluated {index + 1}/{len(original)} images", flush=True)

    means = {name: float(np.mean(values)) for name, values in scores.items()}
    for name, value in means.items():
        print(f"{name}: {value:.3f} dB")
    metrics = {"mean_psnr_db": means, "per_image_psnr_db": scores}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    differences = None
    if proposed is not None:
        gain = means[proposed] - means["QECNN"]
        metrics["proposed_model"] = proposed
        metrics["gain_over_baseline_db"] = gain
        metrics["gain_over_x265_db"] = means[proposed] - means["x265"]
        metrics["meets_target"] = gain >= 0.10
        print(f"Gain over baseline: {gain:+.3f} dB | target: +0.100 dB")
        if "QECNN control" in models:
            control_gain = means[proposed] - means["QECNN control"]
            metrics["gain_over_control_db"] = control_gain
            print(f"Gain over control: {control_gain:+.3f} dB")

        differences = np.array(scores[proposed]) - np.array(scores["QECNN"])
        order = np.argsort(differences)
        labels = ("Lowest gain", "Median gain", "Highest gain")
        indices = (order[0], order[len(order) // 2], order[-1])
        examples = []
        for label, index in zip(labels, indices):
            compressed = decoded[index]
            images = [original[index], compressed]
            images.append(predict(models["QECNN"], compressed, device))
            images.append(predict(models[proposed], compressed, device))
            examples.append((f"{label} | image {index + 1} ({differences[index]:+.2f} dB)", images))
        support.save_examples(args.output_dir / "examples.png", examples, proposed)

    support.save_psnr_plot(args.output_dir / "psnr.png", scores, means, differences)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Results saved in {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate QECNN on the included BSD500 data.")
    commands = parser.add_subparsers(dest="command")
    baseline = commands.add_parser("baseline", help="evaluate the supplied QECNN weights")
    baseline.add_argument("--output-dir", type=Path, default=RUNS / "baseline")

    training = commands.add_parser(
        "train",
        help="train a new QECNN on the training pair",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    training.add_argument("--variant", choices=VARIANTS, default="wide", help="model variant")
    training.add_argument("--epochs", type=int, default=25, help="number of epochs")
    training.add_argument("--batch-size", type=int, default=16, help="patches per batch")
    training.add_argument("--patch-size", type=int, default=64, help="patch side in pixels")
    training.add_argument("--patches-per-image", type=int, default=32, help="crops per image per epoch")
    training.add_argument("--seed", type=int, default=42, help="seed for split, crops and initialization")
    training.add_argument("--initialize", type=Path, help="fine-tune an existing local checkpoint")
    training.add_argument("--all-images", action="store_true", help="use all training images and save the last epoch")
    training.add_argument("--output", type=Path, help="override the checkpoint path")

    testing = commands.add_parser("evaluate", help="compare x265, the baseline and trained models")
    testing.add_argument("--wide", type=Path, help="override the wide model checkpoint")
    testing.add_argument("--control", type=Path, help="optionally include the trained control model")
    testing.add_argument("--improved", action="store_true", help="use fine-tuned weights and average eight views")
    testing.add_argument("--output-dir", type=Path)
    parser.set_defaults(command="baseline")
    args = parser.parse_args()
    if args.command == "baseline" and not hasattr(args, "output_dir"):
        args.output_dir = RUNS / "baseline"
    if args.command == "evaluate" and args.output_dir is None:
        args.output_dir = RUNS / ("final" if args.improved else "comparison")
    if args.command == "train":
        if min(args.epochs, args.batch_size, args.patches_per_image) < 1:
            training.error("epochs, batch size and patches per image must be positive")
        if not 2 * LOSS_BORDER < args.patch_size <= support.HEIGHT:
            training.error(f"patch size must be between {2 * LOSS_BORDER + 1} and {support.HEIGHT}")
        if args.seed < 0:
            training.error("seed must be non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    try:
        if args.command == "train":
            train(args, device)
        else:
            evaluate(args, device)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
