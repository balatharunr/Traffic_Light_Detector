@echo off
setlocal enabledelayedexpansion

REM Traffic Light Detection Application - Windows Batch Script
REM Equivalent to run_detector.sh

echo [INFO] Traffic Light Detection Application
echo [INFO] ===================================

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%venv"

echo [INFO] Script directory: %SCRIPT_DIR%

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "%SCRIPT_DIR%simple_ui.py" (
    echo [ERROR] simple_ui.py not found in %SCRIPT_DIR%
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%requirements.txt" (
    echo [ERROR] requirements.txt not found in %SCRIPT_DIR%
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo [INFO] Creating virtual environment at %VENV_DIR%
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        echo Please check your Python installation
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Show Python version
echo [INFO] Python: 
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Failed to get Python version
    pause
    exit /b 1
)

REM Install/Update dependencies
echo [INFO] Installing/Updating dependencies...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing anyway...
)

pip install -r "%SCRIPT_DIR%requirements.txt" --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo Please check the requirements.txt file
    pause
    exit /b 1
)

REM Test OpenCV installation
echo [INFO] Testing OpenCV installation...
python -c "import cv2; print('OpenCV version:', cv2.__version__)" 2>nul
if errorlevel 1 (
    echo [ERROR] OpenCV installation failed
    echo Please check the requirements.txt file
    pause
    exit /b 1
)

REM Launch the UI
echo [INFO] Launching UI (press q in window to quit, d to toggle debug masks)
echo [INFO] Starting Simple UI...

REM Set PYTHONPATH and run the simple UI
set "PYTHONPATH=%SCRIPT_DIR%"
python "%SCRIPT_DIR%simple_ui.py" %*

REM Check if the script ran successfully
if errorlevel 1 (
    echo [ERROR] Application encountered an error
    echo Error code: %errorlevel%
    pause
)

echo [INFO] Application finished
pause
