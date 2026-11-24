@echo off
setlocal
cd /d "%~dp0" || goto :failed
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHON_MANAGER_AUTOMATIC_INSTALL=false"

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

call :prepare_python || goto :setup_failed
echo Python: "%qecnn_python%"
"%qecnn_python%" --version
"%qecnn_python%" -m pip --version >nul 2>&1 || "%qecnn_python%" -m ensurepip || goto :setup_failed
echo Checking requirements.txt...
"%qecnn_python%" -m pip --disable-pip-version-check install --only-binary=:all: -r requirements.txt || goto :setup_failed
"%qecnn_python%" -c "import cv2, matplotlib, numpy, torch; torch.from_numpy(numpy.zeros(1)); print('PyTorch:', torch.__version__, '| NumPy:', numpy.__version__)" || goto :setup_failed

"%qecnn_python%" -c "import torch; torch.zeros(1, device='cuda').item() if torch.cuda.is_available() else None"
if errorlevel 1 (
    echo CUDA check failed. Using CPU for this run.
    set "CUDA_VISIBLE_DEVICES=-1"
)

if exist "%qecnn_runs%\wide_larger_patches\best.pth" goto :compare
if exist "%qecnn_runs%\wide_all\best.pth" goto :larger_patches
if exist "%qecnn_runs%\wide_refined\best.pth" goto :all_images

if not exist "%qecnn_runs%\wide\best.pth" (
    echo Training the wide model...
    "%qecnn_python%" qecnn.py train --variant wide --output "%qecnn_runs%\wide\best.pth" || goto :failed
)

echo Fine-tuning the wide model...
"%qecnn_python%" qecnn.py train --variant wide --initialize "%qecnn_runs%\wide\best.pth" --epochs 8 --patches-per-image 16 --batch-size 16 --output "%qecnn_runs%\wide_refined\best.pth" || goto :failed

:all_images
echo Fine-tuning on all 200 training images...
"%qecnn_python%" qecnn.py train --variant wide --initialize "%qecnn_runs%\wide_refined\best.pth" --all-images --epochs 8 --patches-per-image 16 --batch-size 16 --output "%qecnn_runs%\wide_all\best.pth" || goto :failed

:larger_patches
echo Fine-tuning on larger patches...
"%qecnn_python%" qecnn.py train --variant wide --initialize "%qecnn_runs%\wide_all\best.pth" --all-images --epochs 6 --patch-size 96 --patches-per-image 8 --batch-size 8 --output "%qecnn_runs%\wide_larger_patches\best.pth" || goto :failed

:compare
echo Comparing the improved model with the baseline...
"%qecnn_python%" qecnn.py evaluate --improved --wide "%qecnn_runs%\wide_larger_patches\best.pth" --output-dir "%qecnn_runs%\final" || goto :failed

echo.
echo Results saved in %qecnn_runs%\final.
pause
exit /b 0

:setup_failed
echo Setup failed. Check the message above, internet access and write permissions for the project folder.

:failed
echo.
echo The run stopped after an error. See the message above.
pause
exit /b 1

:prepare_python
set "qecnn_venv=%~dp0.venv310"
call :find_python "%qecnn_venv%\Scripts\python.exe" && exit /b 0

set "qecnn_runtime=%~dp0.python\3.10.0"
if exist "%qecnn_runtime%\ready" (
    call :find_python "%qecnn_runtime%\tools\python.exe" && goto :create_venv
)
call :find_python py.exe -3.10 && goto :create_venv
call :find_python python.exe && goto :create_venv
call :find_python "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" && goto :create_venv

call :download_python || exit /b 1
call :find_python "%qecnn_runtime%\tools\python.exe" || exit /b 1
type nul > "%qecnn_runtime%\ready"

:create_venv
echo Creating the virtual environment...
"%qecnn_python%" %qecnn_python_args% -m venv "%qecnn_venv%" || exit /b 1
set "qecnn_python=%qecnn_venv%\Scripts\python.exe"
exit /b 0

:find_python
"%~1" %~2 -c "import sys, sysconfig; sys.exit(not (sys.version_info[:3] == (3, 10, 0) and sysconfig.get_platform() == 'win-amd64'))" >nul 2>&1 || exit /b 1
set "qecnn_python=%~1"
set "qecnn_python_args=%~2"
exit /b 0

:download_python
echo Downloading Python 3.10.0 into the project folder...
if not exist "%qecnn_runtime%" (
    mkdir "%qecnn_runtime%" || exit /b 1
)
set "qecnn_python_zip=%qecnn_runtime%\python.zip"
set "qecnn_python_url=https://api.nuget.org/v3-flatcontainer/python/3.10.0/python.3.10.0.nupkg"
set "qecnn_python_sha=4474C83C25625D93E772E926F95F4CD398A0ABBB52793625FA30F39AF3D2CC00"
curl.exe --fail --location --silent --show-error --retry 2 --output "%qecnn_python_zip%" "%qecnn_python_url%" || exit /b 1
certutil.exe -hashfile "%qecnn_python_zip%" SHA256 | findstr.exe /i /x "%qecnn_python_sha%" >nul
if errorlevel 1 (
    echo Python download checksum mismatch.
    exit /b 1
)
tar.exe -xf "%qecnn_python_zip%" -C "%qecnn_runtime%" tools || exit /b 1
del /q "%qecnn_python_zip%"
exit /b 0
