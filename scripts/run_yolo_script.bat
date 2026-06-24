@echo off
chcp 65001 > nul
setlocal

rem ==========================================
rem YOLOv5 training script
rem ==========================================

rem 1. Paths
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
set "YOLO_DIR=%PROJECT_ROOT%\yolov5"
set "DATA_YAML=%PROJECT_ROOT%\Datasets\new_gap20_valtest_clean.yaml"
set "PROJECT_DIR=%PROJECT_ROOT%\runs"

rem 2. Model weights: yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt
set "MODEL_WEIGHTS=yolov5s.pt"

rem 3. Training hyperparameters
set "EPOCHS=20"
set "BATCH_SIZE=8"
set "IMG_SIZE=640"
set "WORKERS=0"

rem 4. Output run name
set "RUN_NAME=train_yolov5s_gap20_valtest_clean"

echo ========================================================
echo Config
echo ========================================================
echo Model weights : %MODEL_WEIGHTS%
echo Epochs        : %EPOCHS%
echo Batch size    : %BATCH_SIZE%
echo Image size    : %IMG_SIZE%
echo Train output  : %PROJECT_DIR%\%RUN_NAME%
echo Data yaml     : %DATA_YAML%
echo ========================================================
echo.

call conda activate YOLOv5s
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to activate conda environment YOLOv5s.
    pause
    exit /b 1
)

cd /d "%YOLO_DIR%"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to enter YOLOv5 directory: %YOLO_DIR%
    pause
    exit /b 1
)

python train.py ^
    --weights "%MODEL_WEIGHTS%" ^
    --data "%DATA_YAML%" ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --img %IMG_SIZE% ^
    --device 0 ^
    --workers %WORKERS% ^
    --project "%PROJECT_DIR%" ^
    --name "%RUN_NAME%"

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed. Please check whether train/val paths exist in %DATA_YAML%.
    pause
    exit /b 1
)

set "BEST_WEIGHTS=%PROJECT_DIR%\%RUN_NAME%\weights\best.pt"
if not exist "%BEST_WEIGHTS%" (
    echo.
    echo [ERROR] best.pt was not generated: %BEST_WEIGHTS%
    pause
    exit /b 1
)

echo.
echo ========================================================
echo Running test-set evaluation and saving metric plots...
echo Test output: %PROJECT_DIR%\%RUN_NAME%_test
echo ========================================================
echo.

python val.py ^
    --weights "%BEST_WEIGHTS%" ^
    --data "%DATA_YAML%" ^
    --img %IMG_SIZE% ^
    --batch-size %BATCH_SIZE% ^
    --device 0 ^
    --workers %WORKERS% ^
    --task test ^
    --project "%PROJECT_DIR%" ^
    --name "%RUN_NAME%_test" ^
    --exist-ok

if errorlevel 1 (
    echo.
    echo [ERROR] Test-set evaluation failed. Please check the test path in %DATA_YAML%.
    pause
    exit /b 1
)

echo.
echo [OK] Training and test-set evaluation finished.
cd /d "%PROJECT_ROOT%\scripts"
pause
