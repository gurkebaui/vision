#!/bin/bash

# AI Gesture Presentation Control - Linux/macOS Runner
# =====================================================

echo "🤖 AI Gesture Presentation Control"
echo "==============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH!"
    echo "Please install Python 3.7+ from https://python.org"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "✅ Python found!"

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📋 Python version: $python_version"

# Install requirements if needed
echo "📦 Checking and installing required packages..."
if python3 -m pip install -r requirements.txt; then
    echo "✅ Packages installed successfully!"
else
    echo "❌ Failed to install packages!"
    echo "Please run: pip3 install -r requirements.txt"
    read -p "Press Enter to exit..."
    exit 1
fi

# Provide instructions
echo ""
echo "📖 Usage Instructions:"
echo "======================="
echo "1. Make sure PowerPoint is open with your presentation"
echo "2. Start presentation mode (F5 or Shift+F5)"
echo "3. Position your hand in front of the camera"
echo "4. Lift a finger to activate the 3-second gesture window"
echo "5. Perform a gesture within 3 seconds"
echo "6. Watch as PowerPoint responds to your gestures!"
echo ""
echo "🎯 Gestures:"
echo "  ✋ Open Palm     - Play/Pause presentation"
echo "  ✊ Closed Fist   - Stop presentation"
echo "  ☝️ Point Up       - Next slide"
echo "  👇 Point Down     - Previous slide"
echo "  👍 Thumbs Up      - Zoom in"
echo "  ✌️ Peace Sign      - Toggle pointer mode"
echo "  👈 Swipe Left     - Previous slide"
echo "  👉 Swipe Right    - Next slide"
echo ""
echo "⚙️  Controls:"
echo "  'q' - Quit application"
echo "  'r' - Reset gesture detection"
echo ""
echo "⚠️  Make sure PowerPoint is in presentation mode!"
echo ""

# Ask user if they want to continue
read -p "Do you want to start the gesture control application? (y/n): " start_app

if [[ "$start_app" =~ ^[Yy]$ ]] || [[ "$start_app" == "" ]]; then
    echo ""
    echo "🎯 Starting AI Gesture Control..."
    python3 gesture_presentation_control.py
else
    echo ""
    echo "👋 Setup complete. You can run the application later with:"
    echo "python3 gesture_presentation_control.py"
fi

read -p "Press Enter to exit..."