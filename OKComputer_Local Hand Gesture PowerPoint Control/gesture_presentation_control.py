#!/usr/bin/env python3
"""
AI Gesture-Controlled PowerPoint Presentation System
Uses OpenCV, MediaPipe, and PyAutoGUI for real hand gesture recognition
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import threading
from collections import deque
from datetime import datetime
import sys
import os

class GesturePresentationController:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture detection variables
        self.gesture_buffer = deque(maxlen=5)
        self.previous_landmarks = None
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0  # 1 second cooldown between gestures
        
        # Timing variables for your 3-second requirement
        self.finger_lift_time = 1
        self.finger_lift_detected = False
        self.gesture_window = 3.0  # 3 seconds to perform gesture after finger lift
        
        # Camera and display
        self.cap = None
        self.display_width = 1280
        self.display_height = 720

        # Upscaling options
        self.enable_ai_upscaling = False # Set to True to enable AI upscaling (if implemented)
        self.upscale_factor = 2 # Factor to upscale the ROI

        self.last_hand_bbox = None # Stores the bounding box of the last detected hand [x_min, y_min, x_max, y_max]
        self.frames_without_detection = 0 # Counter for consecutive frames without hand detection
        self.max_frames_to_search = 3 # Number of frames to search before zooming out (increased for smoother tracking)
        
        # Statistics
        self.detection_stats = {
            'total_frames': 0,
            'gestures_detected': 0,
            'start_time': time.time()
        }
        
        # Initialize PyAutoGUI
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
        print("🤖 AI Gesture Presentation Control initialized!")
        print("==============================================")
        print("Gestures:")
        print("  ✋ Open Palm     - Play/Pause presentation")
        print("  ✊ Closed Fist   - Stop presentation")
        print("  ☝️ Point Up       - Next slide")
        print("  👇 Point Down     - Previous slide")
        print("  👍 Thumbs Up      - Zoom in")
        print("  ✌️ Peace Sign      - Toggle pointer mode")
        print("  👈 Swipe Left     - Previous slide")
        print("  👉 Swipe Right    - Next slide")
        print("\n📸 Looking for camera...")
        
    def start_camera(self):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Error: Cannot access camera!")
            return False
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        print("✅ Camera initialized successfully!")
        return True
    
    def detect_fingers_extended(self, landmarks):
        """Count extended fingers using landmark positions"""
        extended_count = 0
        extended_fingers = []
        
        # Thumb (special case) not included for now
        """thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]

        if thumb_tip.x > thumb_ip.x and thumb_ip.x > thumb_mcp.x:
            extended_count += 1
            extended_fingers.append('thumb')"""
        
        # Other fingers
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        finger_pips = [7, 11, 15, 19]  # Finger joints
        finger_names = ['index', 'middle', 'ring', 'pinky']
        
        for i, (tip, pip, name) in enumerate(zip(finger_tips, finger_pips, finger_names)):
            if landmarks[tip].y < landmarks[pip].y - 0.02:  # Finger is extended
                extended_count += 1
                extended_fingers.append(name)
        
        return extended_count, extended_fingers
    
    def calculate_hand_center(self, landmarks):
        """Calculate center of hand from landmarks"""
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        return center_x, center_y
    
    def detect_gesture(self, landmarks):
        """Main gesture recognition logic"""
        current_time = time.time()
        
        # Count extended fingers
        extended_count, extended_fingers = self.detect_fingers_extended(landmarks)
        
        # Get hand center for movement detection
        center_x, center_y = self.calculate_hand_center(landmarks)
        
        gesture = None
        confidence = 0.0
        
        # Open Palm (all fingers extended)
        if extended_count >= 3:
            gesture = "open_palm"
            confidence = 0.1
        
        # Closed Fist (no fingers extended)
        elif extended_count == 0:
            gesture = "closed_fist"
            confidence = 0.85
        
        # Pointing Up (only index finger extended)
        elif extended_count == 1 and 'index' in extended_fingers:
            gesture = "pointing_up"
            confidence = 0.15
        
        # Thumbs Up (only thumb extended) not included for now
        elif extended_count == 1 and 'thumb' in extended_fingers:
            gesture = "thumbs_up"
            confidence = 0.8

        elif extended_count == 1 and 'pinky' in extended_fingers:
            gesture = "pinky_up"
            confidence = 0.45
        
        # Peace Sign (index and middle fingers extended)
        elif extended_count == 1 and  'middle' in extended_fingers:
            gesture = "peace_sign"
            confidence = 0.85
        
        # Swipe detection based on movement
        if self.previous_landmarks is not None:
            prev_center_x, prev_center_y = self.calculate_hand_center(self.previous_landmarks)
            
            delta_x = center_x - prev_center_x
            delta_y = center_y - prev_center_y
            
            # Detect horizontal swipes
            if abs(delta_x) > 0.05 and abs(delta_y) < 0.03:
                if delta_x > 0:
                    gesture = "swipe_right"
                    confidence = min(0.9, abs(delta_x) * 10)
                else:
                    gesture = "swipe_left"
                    confidence = min(0.9, abs(delta_x) * 10)
            
            # Detect vertical pointing
            elif abs(delta_y) > 0.05 and extended_count == 1:
                if delta_y > 0:
                    gesture = "pointing_down"
                    confidence = min(0.85, abs(delta_y) * 8)
        
        # Store current landmarks for next frame
        self.previous_landmarks = landmarks
        
        return gesture, confidence, extended_count
    
    def execute_presentation_command(self, gesture):
        """Execute PowerPoint control commands"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return False
        
        command_executed = False
        command_name = ""
        
        if gesture == "open_palm":
            pyautogui.press('space')
            command_name = "Play/Pause"
            command_executed = True
        
        elif gesture == "closed_fist":
            pyautogui.press('f')
            command_name = "Stop Presentation"
            command_executed = True
        
        elif gesture == "pointing_up" or gesture == "swipe_right":
            #pyautogui.press('right')
            command_name = "Next Slide"
            command_executed = True
        
        elif gesture == "pointing_down" or gesture == "swipe_left":
            pyautogui.press('left')
            command_name = "Previous Slide"
            command_executed = True
        
        elif gesture == "thumbs_up":
            #pyautogui.hotkey('ctrl', '+')
            command_name = "Zoom In"
            command_executed = True
        
        elif gesture == "peace_sign":
            pyautogui.hotkey('escape')
            command_name = "Toggle Pointer"
            command_executed = True
        
        if command_executed:
            self.last_gesture_time = current_time
            self.detection_stats['gestures_detected'] += 1
            print(f"🎯 {command_name} - {datetime.now().strftime('%H:%M:%S')}")
        
        return command_executed
    
    def check_finger_lift_trigger(self, extended_count, gesture ):
        """Check for finger lift to start 3-second gesture window"""
        current_time = time.time()

        
        # Detect finger lift (from closed fist to any gesture)
        if not self.finger_lift_detected and extended_count > 0 and gesture == "pointing_up" :
            # Check if we had a closed fist recently
            recent_gestures = [g for g in self.gesture_buffer if g['extended_count'] == 0]
            if recent_gestures:
                self.finger_lift_detected = True
                self.finger_lift_time = current_time
                print(f"👆 Finger lift detected! You have 3 seconds to perform a gesture...")
                return True
        
        # Reset if 3-second window expires
        if self.finger_lift_detected and (current_time - self.finger_lift_time) > self.gesture_window:
            self.finger_lift_detected = False
            print("⏰ Gesture window expired. Lift finger again to start new window.")
        
        return self.finger_lift_detected
    
    def draw_gesture_info(self, frame, gesture, confidence, extended_count):
        """Draw gesture information on frame"""
        h, w = frame.shape[:2]
        
        # Background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (30, 30, 30), 1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Gesture information
        y_offset = 35
        
        # Title
        cv2.putText(frame, "🤖 AI Gesture Control", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)
        y_offset += 25
        
        # Current gesture
        gesture_text = f"Gesture: {gesture or 'None'}"
        cv2.putText(frame, gesture_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        # Confidence
        confidence_text = f"Confidence: {confidence:.2f}"
        cv2.putText(frame, confidence_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        # Extended fingers
        fingers_text = f"Extended Fingers: {extended_count}"
        cv2.putText(frame, fingers_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        # 3-second window status
        window_text = "3s Window: " + ("ACTIVE" if self.finger_lift_detected else "WAITING")
        window_color = (0, 255, 0) if self.finger_lift_detected else (255, 255, 255)
        cv2.putText(frame, window_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, window_color, 1)
        
        # Instructions
        instructions = [
            "Lift a finger and perform gesture within 3 seconds",
            "✋ Open Palm: Play/Pause | ✊ Fist: Stop",
            "☝️ Point Up: Next | 👇 Point Down: Previous",
            "👍 Thumbs Up: Zoom | ✌️ Peace: Pointer",
            "Press 'q' to quit | 'r' to reset"
        ]
        
        y_start = h - 120
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, y_start + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def draw_statistics(self, frame):
        """Draw detection statistics"""
        h, w = frame.shape[:2]
        
        # Statistics overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 250, 10), (w - 10, 120), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Calculate statistics
        elapsed_time = time.time() - self.detection_stats['start_time']
        fps = self.detection_stats['total_frames'] / elapsed_time if elapsed_time > 0 else 0
        
        stats = [
            f"FPS: {fps:.1f}",
            f"Frames: {self.detection_stats['total_frames']}",
            f"Gestures: {self.detection_stats['gestures_detected']}",
            f"Runtime: {elapsed_time:.0f}s"
        ]
        
        y_offset = 35
        for stat in stats:
            cv2.putText(frame, stat, (w - 240, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
    
    def run(self):
        """Main application loop"""
        if not self.start_camera():
            return
        
        print("\n🚀 Starting gesture detection...")
        print("📊 Statistics will appear in the top-right corner")
        print("❓ Instructions are displayed on the left side")
        print("\n⚠️  Make sure PowerPoint is open and in presentation mode!")
        print("\n🔄 Looking for finger lift to trigger 3-second gesture window...")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Cannot read frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                processed_image = frame.copy()
                h, w, _ = processed_image.shape

                # Determine if we should use an ROI or the full frame
                use_roi = False
                if self.last_hand_bbox and self.frames_without_detection < self.max_frames_to_search:
                    use_roi = True
                    x_min, y_min, x_max, y_max = self.last_hand_bbox
                    # Add some padding to the bounding box
                    padding = 100 # Increased padding for better hand movement tolerance
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)

                    # Crop the image to the ROI
                    cropped_roi = frame[y_min:y_max, x_min:x_max]
                    
                    # Resize the ROI to a larger size (zoom effect)
                    if cropped_roi.shape[0] > 0 and cropped_roi.shape[1] > 0:
                        processed_image = cv2.resize(cropped_roi, (w, h), interpolation=cv2.INTER_LINEAR)
                    else:
                        processed_image = frame.copy() # Fallback to full frame if ROI is invalid
                        self.last_hand_bbox = None # Reset bbox if it became invalid
                        self.frames_without_detection = 0 # Reset counter as we are processing full frame
                        use_roi = False # No ROI used

                if not use_roi:
                    processed_image = frame.copy() # Process full frame

                processed_image.flags.writeable = False
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                # Apply contrast adjustment to enhance hand features
                processed_image = cv2.convertScaleAbs(processed_image, alpha=1.2, beta=0)
                # Apply sharpening filter to make hand features more distinct
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                processed_image = cv2.filter2D(processed_image, -1, kernel)

                # Placeholder for AI Upscaling (opt-in)
                if self.enable_ai_upscaling:
                    # Implement AI upscaling here if desired and feasible
                    # Example: processed_image = upscale_function(processed_image, self.upscale_factor)
                    pass # Currently no AI upscaling implemented
                
                # Process frame with MediaPipe
                results = self.hands.process(processed_image)
                
                gesture = None
                confidence = 0.0
                extended_count = 0
                
                if results.multi_hand_landmarks:
                    self.frames_without_detection = 0 # Reset counter on successful detection
                    # Use the first detected hand
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Calculate bounding box for the current hand
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    
                    # Convert normalized coordinates to pixel coordinates
                    x_min_px = int(min(x_coords) * w)
                    y_min_px = int(min(y_coords) * h)
                    x_max_px = int(max(x_coords) * w)
                    y_max_px = int(max(y_coords) * h)

                    # Store the current hand's bounding box
                    self.last_hand_bbox = [x_min_px, y_min_px, x_max_px, y_max_px]

                    # MediaPipe landmarks are normalized to the input image (processed_image).
                    # Since processed_image is resized to (w,h), these normalized landmarks
                    # can be directly used for drawing on the original 'frame'.

                    # Convert landmarks to proper format
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm)
                    
                    # Detect gesture
                    gesture, confidence, extended_count = self.detect_gesture(landmarks)
                    
                    # Add to gesture buffer
                    self.gesture_buffer.append({
                        'gesture': gesture,
                        'confidence': confidence,
                        'extended_count': extended_count,
                        'timestamp': time.time()
                    })
                    
                    # Check for finger lift trigger
                    finger_lift_active = self.check_finger_lift_trigger(extended_count, gesture)
                    
                    # Execute command if gesture detected and window is active
                    if gesture and finger_lift_active and confidence > 0.7: # Consider adjusting confidence threshold for ROI
                        self.execute_presentation_command(gesture)
                    
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                else:
                    self.frames_without_detection += 1 # Increment counter if no hand is detected
                    # self.last_hand_bbox is NOT reset here, it's maintained for a few frames

                # Update statistics
                self.detection_stats['total_frames'] += 1
                
                # Draw UI elements
                self.draw_gesture_info(frame, gesture, confidence, extended_count)
                self.draw_statistics(frame)
                
                # Draw ROI rectangle if an ROI was used
                if self.last_hand_bbox and self.frames_without_detection < self.max_frames_to_search:
                    x_min, y_min, x_max, y_max = self.last_hand_bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) # Green rectangle for ROI
                    cv2.putText(frame, "ROI Active", (x_min + 5, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Display frame
                cv2.imshow('AI Gesture Presentation Control', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Quitting application...")
                    break
                elif key == ord('r'):
                    print("\n🔄 Resetting gesture detection...")
                    self.finger_lift_detected = False
                    self.gesture_buffer.clear()
                    self.previous_landmarks = None
                    self.last_hand_bbox = None # Reset bbox on manual reset
                    self.frames_without_detection = 0 # Reset counter on manual reset

                # Update statistics
                self.detection_stats['total_frames'] += 1
                
                # Draw UI elements
                self.draw_gesture_info(frame, gesture, confidence, extended_count)
                self.draw_statistics(frame)
                
                # Display frame
                cv2.imshow('AI Gesture Presentation Control', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Quitting application...")
                    break
                elif key == ord('r'):
                    print("\n🔄 Resetting gesture detection...")
                    self.finger_lift_detected = False
                    self.gesture_buffer.clear()
                    self.previous_landmarks = None
        
        except KeyboardInterrupt:
            print("\n\n👋 Application interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print("\n📊 Final Statistics:")
        print(f"   Total Frames: {self.detection_stats['total_frames']}")
        print(f"   Gestures Detected: {self.detection_stats['gestures_detected']}")
        print(f"   Runtime: {time.time() - self.detection_stats['start_time']:.1f} seconds")
        
        if self.detection_stats['total_frames'] > 0:
            gesture_rate = (self.detection_stats['gestures_detected'] / self.detection_stats['total_frames']) * 100
            print(f"   Gesture Detection Rate: {gesture_rate:.2f}%")
        
        print("\n✅ Application closed successfully")

def main():
    """Main function"""
    print("🚀 AI Gesture-Controlled PowerPoint Presentation")
    print("===============================================")
    print("A Python application that uses hand gestures to control PowerPoint presentations.")
    print("\nRequirements:")
    print("  - Webcam")
    print("  - PowerPoint open and in presentation mode")
    print("  - Python packages: opencv-python, mediapipe, pyautogui, numpy")
    print("\nHow it works:")
    print("  1. Lift a finger to start 3-second gesture window")
    print("  2. Perform gesture within 3 seconds")
    print("  3. System automatically controls PowerPoint")
    print("\nPress Ctrl+C to quit")
    
    # Check if required packages are installed
    try:
        import cv2
        import mediapipe
        import pyautogui
        import numpy
        print("\n✅ All required packages are installed!")
    except ImportError as e:
        print(f"\n❌ Missing package: {e}")
        print("\nPlease install required packages:")
        print("pip install opencv-python mediapipe pyautogui numpy")
        return
    
    # Create and run the gesture controller
    controller = GesturePresentationController()
    controller.run()

if __name__ == "__main__":
    main()