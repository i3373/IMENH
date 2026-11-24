# IMENH

Восстановление изображений BSD500 после сжатия x265. На 100 тестовых кадрах получили 24,840 дБ. Это на 0,103 дБ выше базовой QECNN. [Метрики](results.json) · [Презентация](output/pdf/qecnn-report.pdf)

## Установка на Windows

Подойдёт Python 3.10, 3.11 или 3.12 для Windows x64. Команды выполняются в PowerShell из папки с `qecnn.py`. Список установленных версий показывает `py -0p`.

```powershell
py -3.10 -m venv .venv
```

Для Python 3.11 или 3.12 в этой команде нужно указать `-3.11` или `-3.12`. Если `py` недоступен, но `python --version` показывает подходящую версию, можно выполнить `python -m venv .venv`.

Если PowerShell не находит установленный Python, можно указать путь к `python.exe`, например `& "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe" -m venv .venv`. После установки Python нужно открыть новое окно PowerShell. [Установка Python в Windows](https://docs.python.org/3/using/windows.html)

Достаточно установить одну сборку PyTorch. Для NVIDIA используется сборка с CUDA 11.8.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

Для запуска на процессоре используется сборка PyTorch для CPU.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

Остальные зависимости одинаковы для обеих сборок.

```powershell
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4 opencv-python==4.9.0.80 matplotlib==3.10.3
```

## Проверка запуска

В репозитории уже есть исходные YUV, декодированные x265 YUV и базовые веса `best_model2026.pth` из задания. Для проверки достаточно запустить сравнение на 100 тестовых изображениях.

```powershell
.\.venv\Scripts\python.exe qecnn.py baseline
```

Средний PSNR будет около 24,517 дБ для x265 и 24,738 дБ для базовой QECNN. Метрики и график сохраняются в `runs/baseline`.

## Обучение с нуля

Батник сам использует Python из `.venv`. Для полного обучения желательно использовать видеокарту NVIDIA.

```powershell
.\train_and_compare.bat --fresh
```

Флаг `--fresh` создаёт отдельную папку запуска. В `runs/repro-*/final` сохраняются `metrics.json`, `psnr.png` и `examples.png`. При запуске без флага батник использует уже пройденные этапы из `runs`, если они есть, и сохраняет итог в `runs/final`.

## Как собрать презентацию

```powershell
typst compile --root . output/pdf/qecnn-report.typ
```
