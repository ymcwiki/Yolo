@echo off
chcp 65001
echo ===================================
echo YOLOv8缺陷检测训练脚本
echo ===================================

REM 设置Python环境路径（如果需要）
REM set PATH=%PATH%;C:\Python39\

echo 正在安装依赖库...
pip install ultralytics -q

echo.
echo 请选择YOLOv8模型大小:
echo 1. YOLOv8n (最小, 最快)
echo 2. YOLOv8s (小型)
echo 3. YOLOv8m (中型)
echo 4. YOLOv8l (大型)
echo 5. YOLOv8x (最大, 最精确)
set /p model_choice="请选择(1-5): "

if "%model_choice%"=="1" set model_size=n
if "%model_choice%"=="2" set model_size=s
if "%model_choice%"=="3" set model_size=m
if "%model_choice%"=="4" set model_size=l
if "%model_choice%"=="5" set model_size=x

echo.
set /p epochs="请输入训练轮数 (默认: 100): "
if "%epochs%"=="" set epochs=100

echo.
set /p batch_size="请输入批量大小 (默认: 16): "
if "%batch_size%"=="" set batch_size=16

echo.
set /p workers="请输入工作进程数量 (默认: 2, 推荐: 1-4): "
if "%workers%"=="" set workers=2

echo.
set /p cache_choice="是否启用图像缓存以加速训练? (y/n, 默认: y): "
if "%cache_choice%"=="" set cache_choice=y

if /i "%cache_choice%"=="y" (
    set cache_flag=--cache
) else (
    set cache_flag=
)

echo.
echo 正在启动YOLOv8训练...
python train_yolo.py --model_size %model_size% --epochs %epochs% --batch_size %batch_size% --workers %workers% %cache_flag%

echo.
echo 训练完成! 按任意键退出...
pause > nul