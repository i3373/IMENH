@echo off
setlocal
cd /d "%~dp0" || goto :failed
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHON_MANAGER_AUTOMATIC_INSTALL=false"

call :find_python "%~dp0.venv\Scripts\python.exe" && goto :python_ready
call :find_python py.exe -3.11 && goto :python_ready
call :find_python python.exe && goto :python_ready
call :find_python "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" && goto :python_ready
call :find_python py.exe && goto :python_ready

echo Python was not found. Install Python 3.11 for Windows and follow README.md.
goto :failed

:python_ready
echo Python: "%qecnn_python%" %qecnn_python_args%
"%qecnn_python%" %qecnn_python_args% -c "import cv2, matplotlib, numpy, torch"
if errorlevel 1 (
    echo Python is available. Install the project dependencies using README.md.
    goto :failed
)

set "qecnn_runs=runs"
if "%~1"=="" goto :prepare
if /i "%~1"=="--fresh" goto :fresh
echo Usage: train_and_compare.bat [--fresh]
goto :failed

:fresh
set "qecnn_runs=runs\repro-%RANDOM%-%RANDOM%"
if exist "%qecnn_runs%" goto :fresh

:prepare
echo Run directory: %qecnn_runs%
if not exist "best_model2026.pth" (
    echo Place the supplied best_model2026.pth file next to qecnn.py for comparison.
    goto :failed
)

if exist "%qecnn_runs%\wide_larger_patches\best.pth" goto :control
if exist "%qecnn_runs%\wide_all\best.pth" goto :larger_patches
if exist "%qecnn_runs%\wide_refined\best.pth" goto :all_images

if not exist "%qecnn_runs%\wide\best.pth" (
    echo Training the wide model...
    "%qecnn_python%" %qecnn_python_args% qecnn.py train --variant wide --output "%qecnn_runs%\wide\best.pth" || goto :failed
)

echo Fine-tuning the wide model...
"%qecnn_python%" %qecnn_python_args% qecnn.py train --variant wide --initialize "%qecnn_runs%\wide\best.pth" --epochs 8 --patches-per-image 16 --batch-size 16 --output "%qecnn_runs%\wide_refined\best.pth" || goto :failed

:all_images
echo Fine-tuning on all 200 training images...
"%qecnn_python%" %qecnn_python_args% qecnn.py train --variant wide --initialize "%qecnn_runs%\wide_refined\best.pth" --all-images --epochs 8 --patches-per-image 16 --batch-size 16 --output "%qecnn_runs%\wide_all\best.pth" || goto :failed

:larger_patches
echo Fine-tuning on larger patches...
"%qecnn_python%" %qecnn_python_args% qecnn.py train --variant wide --initialize "%qecnn_runs%\wide_all\best.pth" --all-images --epochs 6 --patch-size 96 --patches-per-image 8 --batch-size 8 --output "%qecnn_runs%\wide_larger_patches\best.pth" || goto :failed

:control
if not exist "%qecnn_runs%\control\best.pth" (
    echo Training the control model...
    "%qecnn_python%" %qecnn_python_args% qecnn.py train --variant control --output "%qecnn_runs%\control\best.pth" || goto :failed
)

echo Comparing the improved model with the baseline...
"%qecnn_python%" %qecnn_python_args% qecnn.py evaluate --improved --wide "%qecnn_runs%\wide_larger_patches\best.pth" --control "%qecnn_runs%\control\best.pth" --output-dir "%qecnn_runs%\final" || goto :failed

echo.
echo Results saved in %qecnn_runs%\final.
pause
exit /b 0

:failed
echo.
echo The run stopped after an error. See the message above.
pause
exit /b 1

:find_python
"%~1" %~2 -c "import sys" >nul 2>&1 || exit /b 1
set "qecnn_python=%~1"
set "qecnn_python_args=%~2"
exit /b 0
