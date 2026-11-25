@echo off
REM Quick start script for CrimeShield AI Web UI

echo.
echo ============================================
echo CrimeShield AI - Web UI Server
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please run setup.bat first
    pause
    exit /b 1
)

if not exist "best_tristream_model.pth" (
    echo Error: Model file 'best_tristream_model.pth' not found
    echo Please ensure the trained model is in the current directory
    pause
    exit /b 1
)

echo Starting Flask server...
echo.
echo Once started, open your browser to:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py
pause
