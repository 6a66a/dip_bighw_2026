@echo off
chcp 65001 > nul


REM 1. Project paths
for %%I in ("%~dp0..") do set "PROJECT_DIR=%%~fI"
set "YOLO_DIR=%PROJECT_DIR%\yolov5"

REM 2. Model weights
set "WEIGHTS=%PROJECT_DIR%\runs\train_yolov5s_gap20_valtest_clean\weights\best.pt"

REM 3. Input data to detect
set "SOURCE=%PROJECT_DIR%\Datasets\new_data_gap20_valtest_clean_hsi_i_darkest25\images\test"

REM 4. Output folders
set "RESULT_DIR=%PROJECT_DIR%\runs\detect\gamma__new_data_gap20_valtest_clean_darkest25"
set "OUTPUT=%RESULT_DIR%\boxed_images"

REM Optional validation/evaluation output
set RUN_EVAL=1
set "DATA_YAML=%PROJECT_DIR%\Datasets\new_gap20_valtest_clean_hsi_i_darkest25.yaml"
set EVAL_SPLIT=test
set "EVAL_OUTPUT=%RESULT_DIR%"

REM 5. Inference parameters
set IMG_SIZE=640
set CONF_THRES=0.25
set DEVICE=0
set EVAL_BATCH=16

REM Optional image preprocessing methods
REM PREPROCESS options:
REM   none               
REM   gamma              
REM   hsi_equalize       
REM   yuv_equalize       
REM   yuv_clahe          
REM   gamma_hsi_equalize 
set PREPROCESS=yuv_clahe
set PREPROCESS_GAMMA=0.8
set PREPROCESS_CLAHE_CLIP_LIMIT=2.0
set PREPROCESS_CLAHE_TILE_SIZE=8
set SAVE_TXT=1
set SAVE_CONF=1

echo ========================================================
echo YOLOv5 detection config
echo ========================================================
echo Weights : %WEIGHTS%
echo Source  : %SOURCE%
echo Result dir : %RESULT_DIR%
echo Boxed images : %OUTPUT%
echo Run eval : %RUN_EVAL%
echo Data yaml : %DATA_YAML%
echo Eval split : %EVAL_SPLIT%
echo Eval output : %EVAL_OUTPUT%
echo Image size : %IMG_SIZE%
echo Confidence : %CONF_THRES%
echo Device : %DEVICE%
echo Preprocess : %PREPROCESS%
echo Preprocess gamma : %PREPROCESS_GAMMA%
echo Preprocess CLAHE clip limit : %PREPROCESS_CLAHE_CLIP_LIMIT%
echo Preprocess CLAHE tile size : %PREPROCESS_CLAHE_TILE_SIZE%
echo Save txt : %SAVE_TXT%
echo Save confidence : %SAVE_CONF%
echo ========================================================
echo.

call conda activate YOLOv5s

cd /d "%PROJECT_DIR%"

set EXTRA_ARGS=
if "%SAVE_TXT%"=="1" set EXTRA_ARGS=%EXTRA_ARGS% --save-txt
if "%SAVE_CONF%"=="1" set EXTRA_ARGS=%EXTRA_ARGS% --save-conf
set EXTRA_ARGS=%EXTRA_ARGS% --preprocess %PREPROCESS% --preprocess-gamma %PREPROCESS_GAMMA% --preprocess-clahe-clip-limit %PREPROCESS_CLAHE_CLIP_LIMIT% --preprocess-clahe-tile-size %PREPROCESS_CLAHE_TILE_SIZE%
if "%RUN_EVAL%"=="1" set EXTRA_ARGS=%EXTRA_ARGS% --eval-data "%DATA_YAML%" --eval-split %EVAL_SPLIT% --eval-output "%EVAL_OUTPUT%" --eval-batch %EVAL_BATCH%

python detect_with_model.py ^
    --weights "%WEIGHTS%" ^
    --source "%SOURCE%" ^
    --output "%OUTPUT%" ^
    --yolo-dir "%YOLO_DIR%" ^
    --img %IMG_SIZE% ^
    --conf %CONF_THRES% ^
    --device %DEVICE% ^
    %EXTRA_ARGS%

echo.
echo Detection finished.
cd /d "%PROJECT_DIR%\scripts"
pause
