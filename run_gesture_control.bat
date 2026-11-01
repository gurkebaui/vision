@echo off
echo Vision Control v2 - Long-Range Gesture System
echo =================================================

set PYTHON_BIN=python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_BIN=.venv\Scripts\python.exe
)

%PYTHON_BIN% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.9+ is not installed or not in PATH.
    pause
    exit /b 1
)

if "%1"=="--studio" (
    %PYTHON_BIN% gesture_presentation_control.py %*
    exit /b %errorlevel%
)

%PYTHON_BIN% setup_and_run.py %*
pause
