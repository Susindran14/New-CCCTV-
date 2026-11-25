@echo off
REM CrimeShield AI Setup Script

echo.
echo ============================================
echo CrimeShield AI - Setup Script
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating project virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Error: Failed to create the virtual environment
        pause
        exit /b 1
    )
)

set "PYTHON=.venv\Scripts\python.exe"

echo [1/3] Python found. Installing dependencies...
"%PYTHON%" -m pip install -r requirements.txt

if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo [2/3] Checking for model files...
if not exist "best_tristream_model.pth" (
    echo Warning: best_tristream_model.pth not found
    echo Please ensure the model file is in the current directory
    echo.
)

if not exist "yolov8n-pose.pt" (
    echo Warning: yolov8n-pose.pt not found
    echo Downloading YOLOv8 pose model...
    "%PYTHON%" -c "from ultralytics import YOLO; model = YOLO('yolov8n-pose.pt')"
)

echo [3/3] Setup complete!
echo.
echo ============================================
echo To start the application:
echo   "%PYTHON%" app.py
echo.
echo Then open your browser to:
echo   http://localhost:5000
echo ============================================
echo.

pause
