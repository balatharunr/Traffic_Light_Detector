@echo off
setlocal enabledelayedexpansion

REM Streamlit Traffic Light Detection Application - Windows Batch Script
REM ===================================================================

echo 🚦 Starting Traffic Light Detection Streamlit App...
echo ==================================================

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
if not exist "%SCRIPT_DIR%streamlit_app.py" (
    echo [ERROR] streamlit_app.py not found in %SCRIPT_DIR%
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%traffic_light_detector.py" (
    echo [ERROR] traffic_light_detector.py not found in %SCRIPT_DIR%
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
    echo 📦 Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        echo Please check your Python installation
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
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
echo 📥 Installing dependencies...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing anyway...
)

echo [INFO] Installing requirements from %SCRIPT_DIR%requirements.txt
pip install -r "%SCRIPT_DIR%requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo Please check the requirements.txt file
    pause
    exit /b 1
)

REM Test Streamlit installation
echo [INFO] Testing Streamlit installation...
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)" 2>nul
if errorlevel 1 (
    echo [ERROR] Streamlit installation failed
    echo Please check the requirements.txt file
    pause
    exit /b 1
)

REM Run Streamlit app
echo 🚀 Starting Streamlit app...
echo    Open your browser to: http://localhost:8501
echo    Press Ctrl+C to stop the app
echo.

REM Change to script directory and run Streamlit
cd /d "%SCRIPT_DIR%"
streamlit run streamlit_app.py

REM Check if the script ran successfully
if errorlevel 1 (
    echo [ERROR] Streamlit app encountered an error
    echo Error code: %errorlevel%
    pause
)

echo [INFO] Streamlit app finished
pause
