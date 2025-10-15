#!/usr/bin/env python3
"""
Setup and Run Script for AI Gesture Presentation Control
Installs required packages and runs the gesture control application
"""

import subprocess
import sys
import os
import platform

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package):
    """Check if a package is installed"""
    try:
        __import__(package.replace('-', '_'))
        return True
    except ImportError:
        return False

def setup_environment():
    """Set up the Python environment with required packages"""
    print("🔧 Setting up AI Gesture Control environment...")
    print("=" * 50)
    
    required_packages = [
        "opencv-python",
        "mediapipe", 
        "pyautogui",
        "numpy",
        "Pillow"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        print(f"Checking {package}...", end=" ")
        if check_package(package):
            print("✅ Already installed")
        else:
            print("❌ Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        
        for package in missing_packages:
            print(f"Installing {package}...", end=" ")
            if install_package(package):
                print("✅ Success")
            else:
                print("❌ Failed")
                return False
    
    print("\n✅ Environment setup complete!")
    return True

def check_system_requirements():
    """Check system requirements"""
    print("\n🔍 Checking system requirements...")
    print("=" * 30)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("❌ Python 3.7 or higher is required")
        return False
    
    # Check operating system
    os_name = platform.system()
    print(f"Operating System: {os_name} {platform.release()}")
    
    if os_name not in ["Windows", "Darwin", "Linux"]:
        print("⚠️  Unsupported operating system")
        return False
    
    # Check for camera
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera detected")
            cap.release()
        else:
            print("⚠️  No camera detected - application may not work properly")
    except:
        print("⚠️  Cannot check camera - OpenCV not installed")
    
    return True

def provide_usage_instructions():
    """Provide usage instructions"""
    print("\n📖 Usage Instructions")
    print("=" * 30)
    print("1. Make sure PowerPoint is open with your presentation")
    print("2. Start the presentation mode (F5 or Shift+F5)")
    print("3. Run this application")
    print("4. Position your hand in front of the camera")
    print("5. Lift a finger to activate the 3-second gesture window")
    print("6. Perform a gesture within 3 seconds")
    print("7. Watch as PowerPoint responds to your gestures!")
    
    print("\n🎯 Gesture Commands:")
    print("  ✋ Open Palm     - Play/Pause presentation")
    print("  ✊ Closed Fist   - Stop presentation") 
    print("  ☝️ Point Up       - Next slide")
    print("  👇 Point Down     - Previous slide")
    print("  👍 Thumbs Up      - Zoom in")
    print("  ✌️ Peace Sign      - Toggle pointer mode")
    print("  👈 Swipe Left     - Previous slide")
    print("  👉 Swipe Right    - Next slide")
    
    print("\n⚙️  Controls:")
    print("  'q' - Quit application")
    print("  'r' - Reset gesture detection")

def main():
    """Main setup and run function"""
    print("🚀 AI Gesture Presentation Control Setup")
    print("=" * 50)
    print("This script will check and install required packages,")
    print("then run the gesture control application.")
    
    # Check system requirements
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please check the requirements above.")
        return
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed. Please check the error messages above.")
        return
    
    # Provide usage instructions
    provide_usage_instructions()
    
    # Ask user if they want to continue
    print("\n🚀 Ready to start the application!")
    response = input("Do you want to start the gesture control application now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '']:
        print("\n🎯 Starting AI Gesture Control...")
        try:
            # Import and run the main application
            from gesture_presentation_control import GesturePresentationController
            
            controller = GesturePresentationController()
            controller.run()
            
        except ImportError as e:
            print(f"❌ Error importing main module: {e}")
            print("\nPlease make sure all packages are installed correctly.")
            print("You can try installing them manually:")
            print("pip install -r requirements.txt")
        
        except Exception as e:
            print(f"❌ Error running application: {e}")
            print("\nPlease check the error message and try again.")
    
    else:
        print("\n👋 Setup complete. You can run the application later with:")
        print("python gesture_presentation_control.py")

if __name__ == "__main__":
    main()