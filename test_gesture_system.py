#!/usr/bin/env python3
"""
Test Script for AI Gesture Control System
Verifies that all components are working correctly
"""

import sys
import subprocess
import importlib
import platform
import os

def test_python_version():
    """Test Python version compatibility"""
    print("🔍 Testing Python version...", end=" ")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} (requires 3.7+)")
        return False

def test_package_imports():
    """Test if all required packages can be imported"""
    packages = [
        ('cv2', 'opencv-python'),
        ('mediapipe', 'mediapipe'),
        ('pyautogui', 'pyautogui'),
        ('numpy', 'numpy'),
        ('PIL', 'Pillow')
    ]
    
    print("\n📦 Testing package imports...")
    all_good = True
    
    for package_name, pip_name in packages:
        try:
            importlib.import_module(package_name)
            print(f"  ✅ {pip_name}")
        except ImportError as e:
            print(f"  ❌ {pip_name}: {e}")
            all_good = False
    
    return all_good

def test_camera_access():
    """Test camera accessibility"""
    print("\n📷 Testing camera access...", end=" ")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                print(f"✅ Camera detected ({frame.shape[1]}x{frame.shape[0]})")
                return True
            else:
                print("❌ Cannot read from camera")
                return False
        else:
            print("❌ Cannot open camera")
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_pyautogui():
    """Test PyAutoGUI functionality"""
    print("\n🎯 Testing PyAutoGUI...", end=" ")
    
    try:
        import pyautogui
        
        # Test basic functionality
        screen_width, screen_height = pyautogui.size()
        print(f"✅ Screen size: {screen_width}x{screen_height}")
        
        # Test if we can get mouse position (non-intrusive)
        mouse_x, mouse_y = pyautogui.position()
        print(f"  Mouse position: ({mouse_x}, {mouse_y})")
        
        return True
        
    except Exception as e:
        print(f"❌ PyAutoGUI test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe hands model"""
    print("\n🤖 Testing MediaPipe...", end=" ")
    
    try:
        import mediapipe as mp
        
        # Test hands initialization
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        print("✅ MediaPipe Hands initialized successfully")
        hands.close()
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'gesture_presentation_control.py',
        'requirements.txt',
        'README.md',
        'setup_and_run.py',
        'example_presentation_tips.txt'
    ]
    
    optional_files = [
        'run_gesture_control.bat',
        'run_gesture_control.sh'
    ]
    
    all_good = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (required)")
            all_good = False
    
    for file in optional_files:
        if os.path.exists(file):
            print(f"  ✅ {file} (optional)")
        else:
            print(f"  ⚠️  {file} (optional)")
    
    return all_good

def test_os_compatibility():
    """Test operating system compatibility"""
    print("\n💻 Testing OS compatibility...")
    
    os_name = platform.system()
    os_version = platform.release()
    
    print(f"  Operating System: {os_name} {os_version}")
    
    if os_name in ["Windows", "Darwin", "Linux"]:
        print(f"  ✅ {os_name} is supported")
        return True
    else:
        print(f"  ⚠️  {os_name} may not be fully supported")
        return False

def generate_test_report(results):
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("📊 AI GESTURE CONTROL SYSTEM - TEST REPORT")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n📈 Summary:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {total_tests - passed_tests}")
    print(f"  Success Rate: {success_rate:.1f}%")
    
    print(f"\n📋 Detailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\n🔍 Recommendations:")
    
    if success_rate == 100:
        print("  🎉 Perfect! Your system is ready to run the gesture control application.")
        print("  💡 You can now run: python gesture_presentation_control.py")
    elif success_rate >= 80:
        print("  ✅ Good! Your system should work with the gesture control application.")
        print("  ⚠️  Check the failed tests above and fix any issues before running.")
    elif success_rate >= 60:
        print("  ⚠️  Your system needs some fixes before it will work properly.")
        print("  🔧 Please address the failed tests above.")
    else:
        print("  ❌ Your system needs significant setup before it will work.")
        print("  📖 Please follow the setup instructions in README.md")
    
    print(f"\n🚀 Next Steps:")
    if results.get('Package Imports', False):
        print("  1. Run the setup script: python setup_and_run.py")
        print("  2. Or run directly: python gesture_presentation_control.py")
    else:
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Run this test again: python test_gesture_system.py")
        print("  3. Once all tests pass, run the application")

def main():
    """Main test function"""
    print("🧪 AI Gesture Control System - Test Suite")
    print("="*50)
    print("Testing your system for compatibility with the gesture control application...")
    
    # Run all tests
    test_results = {
        'Python Version': test_python_version(),
        'OS Compatibility': test_os_compatibility(),
        'File Structure': test_file_structure(),
        'Package Imports': test_package_imports(),
        'Camera Access': test_camera_access(),
        'MediaPipe': test_mediapipe(),
        'PyAutoGUI': test_pyautogui()
    }
    
    # Generate report
    generate_test_report(test_results)
    
    # Exit with appropriate code
    success_rate = sum(test_results.values()) / len(test_results) * 100
    
    if success_rate >= 80:
        print(f"\n🎉 System test completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️  System test completed with issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()