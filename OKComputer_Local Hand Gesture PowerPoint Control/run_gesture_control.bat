@echo off
REM AI Gesture Presentation Control - Windows Runner
REM ================================================

echo 🤖 AI Gesture Presentation Control
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH!
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found!

REM Install requirements if needed
echo 📦 Checking and installing required packages...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Failed to install packages!
    echo Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

echo ✅ Packages installed successfully!

REM Provide instructions
echo.
echo 📖 Usage Instructions:
echo =======================
echo 1. Make sure PowerPoint is open with your presentation
echo 2. Start presentation mode (F5 or Shift+F5)
echo 3. Position your hand in front of the camera
echo 4. Lift a finger to activate the 3-second gesture window
echo 5. Perform a gesture within 3 seconds
echo 6. Watch as PowerPoint responds to your gestures!
echo.
echo 🎯 Gestures:
echo   ✋ Open Palm     - Play/Pause presentation
echo   ✊ Closed Fist   - Stop presentation
echo   ☝️ Point Up       - Next slide
echo   👇 Point Down     - Previous slide
echo   👍 Thumbs Up      - Zoom in
echo   ✌️ Peace Sign      - Toggle pointer mode
echo   👈 Swipe Left     - Previous slide
echo   👉 Swipe Right    - Next slide
echo.
echo ⚙️  Controls:
echo   'q' - Quit application
echo   'r' - Reset gesture detection
echo.
echo ⚠️  Make sure PowerPoint is in presentation mode!
echo.

REM Ask user if they want to continue
set /p start_app=Do you want to start the gesture control application? (y/n): 

if /i "%start_app%"=="y" (
    echo.
    echo 🎯 Starting AI Gesture Control...
    python gesture_presentation_control.py
) else if /i "%start_app%"=="yes" (
    echo.
    echo 🎯 Starting AI Gesture Control...
    python gesture_presentation_control.py
) else (
    echo.
    echo 👋 Setup complete. You can run the application later with:
    echo python gesture_presentation_control.py
)

pause