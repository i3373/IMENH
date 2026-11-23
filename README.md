# IMENH

Восстановление изображений BSD500 после сжатия x265. На 100 тестовых кадрах получили 24,840 дБ. Это на 0,103 дБ выше базовой QECNN. [Метрики](results.json) · [Презентация](output/pdf/qecnn-report.pdf)

## Как запустить

Для запуска нужны Git и Python 3.11. Команды установки для Windows с видеокартой NVIDIA выполняются в PowerShell из папки проекта.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4 opencv-python==4.9.0.80 matplotlib==3.10.3
```
К сожалению, датасеты оказались слишком большими для загрузки на github, поэтому запускать необходимо в папке с исходными датасетами.
Батник `train_and_compare.bat` можно запустить двойным щелчком или из PowerShell.

```powershell
.\train_and_compare.bat
```

При первом запуске батник обучит модели и сохранит метрики, график и примеры в `runs/final`. Если в истории Git нет эталона QECNN, нужно загрузить исходный коммит командой ниже и повторить запуск.

```powershell
git fetch https://ctlab.itmo.ru/gitlab/eabelyaev/cnnimagequalityenhancement.git af78856948403c3cce7705eb2d7f52f6f44a6824
```

Для обучения с нуля в отдельной папке есть флаг `--fresh`.

```powershell
.\train_and_compare.bat --fresh
```

Результат будет в отдельной папке `runs/repro-*/final`. Веса и результаты прогонов в `runs` исключены из Git.
Или можно удалить папку `runs` и запустить батник еще раз

## Как собрать презентацию

```powershell
typst compile --root . output/pdf/qecnn-report.typ
```
