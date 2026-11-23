import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


WIDTH = 480
HEIGHT = 320
LUMA_BYTES = WIDTH * HEIGHT
CHROMA_BYTES = (WIDTH // 2) * (HEIGHT // 2)
FRAME_BYTES = LUMA_BYTES + 2 * CHROMA_BYTES


def cal_psnr(img_orig, img_out):
    """Покадровый RGB PSNR для изображений в шкале [0, 1]."""
    mse = np.mean(np.square(img_orig - img_out))
    return float(10 * np.log10(1.0 / (mse + 1e-10)))


def yuv2rgb(y, u, v, width, height):
    # Преобразование исходного проекта для сопоставимости с базовым PSNR.
    u = cv2.resize(u, (width, height), interpolation=cv2.INTER_CUBIC)
    v = cv2.resize(v, (width, height), interpolation=cv2.INTER_CUBIC)
    red = y + 1.4075 * (v - 128.0)
    green = y - 0.3455 * (u - 128.0) - 0.7169 * (v - 128.0)
    blue = y + 1.7790 * (u - 128.0)
    return tuple(np.clip(channel, 0, 255) for channel in (red, green, blue))


class YuvFrames:
    """Чтение отдельных кадров YUV420 без выгрузки промежуточных PNG."""

    def __init__(self, path):
        count, remainder = divmod(path.stat().st_size, FRAME_BYTES)
        if count == 0 or remainder:
            raise ValueError(f"{path}: expected complete {WIDTH}x{HEIGHT} YUV420 frames")
        self.frames = np.memmap(path, dtype=np.uint8, mode="r", shape=(count, FRAME_BYTES))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        y = frame[:LUMA_BYTES].reshape(HEIGHT, WIDTH)
        u = frame[LUMA_BYTES:LUMA_BYTES + CHROMA_BYTES].reshape(HEIGHT // 2, WIDTH // 2)
        v = frame[-CHROMA_BYTES:].reshape(HEIGHT // 2, WIDTH // 2)
        return np.stack(yuv2rgb(y, u, v, WIDTH, HEIGHT), axis=-1) / 255.0


def open_pair(original_path, decoded_path, expected_frames=None):
    original = YuvFrames(original_path)
    decoded = YuvFrames(decoded_path)
    if len(original) != len(decoded):
        raise ValueError(f"Frame counts differ: {original_path} and {decoded_path}")
    if expected_frames is not None and len(original) != expected_frames:
        raise ValueError(f"{original_path}: expected {expected_frames} frames, found {len(original)}")
    return original, decoded


def save_psnr_plot(path, scores, means, differences=None):
    if differences is None:
        figure, axis = plt.subplots(figsize=(12, 4), layout="constrained")
        axis.set_xlabel("Image number")
    else:
        figure, (axis, gains) = plt.subplots(
            2, 1, sharex=True, figsize=(12, 6), height_ratios=(2, 1), layout="constrained"
        )
        gains.plot(range(1, len(differences) + 1), differences, color="tab:purple", linewidth=1)
        gains.axhline(0, color="0.5", linewidth=1)
        gains.set(xlabel="Image number", ylabel="Gain over QECNN (dB)")
        gains.grid(alpha=0.25)

    for name, values in scores.items():
        axis.plot(range(1, len(values) + 1), values, label=f"{name} ({means[name]:.3f} dB)")
    axis.set_ylabel("RGB PSNR (dB)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def detail_crop(images):
    # Выбираем квадрат, где выходы базовой и новой моделей отличаются сильнее всего.
    # На всех четырёх изображениях показываем один и тот же квадрат.
    size = 80
    difference = np.abs(images[2] - images[3]).mean(axis=2)
    tiles = difference.reshape(HEIGHT // size, size, WIDTH // size, size)
    tile_scores = tiles.mean(axis=(1, 3))
    row, col = np.unravel_index(tile_scores.argmax(), tile_scores.shape)
    rows = slice(row * size, (row + 1) * size)
    cols = slice(col * size, (col + 1) * size)
    return [image[rows, cols] for image in images]


def save_examples(path, examples, proposed):
    figure, axes = plt.subplots(
        len(examples), 4, figsize=(10, 2.8 * len(examples)), squeeze=False, layout="constrained"
    )
    for axis, title in zip(axes[0], ("Original", "x265", "QECNN", proposed)):
        axis.set_title(title)

    for row, (label, images) in zip(axes, examples):
        for axis, image in zip(row, detail_crop(images)):
            axis.imshow(np.clip(image, 0, 1), interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        row[0].set_ylabel(label)
    figure.savefig(path, dpi=160)
    plt.close(figure)
