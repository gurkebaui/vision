#!/usr/bin/env python3
"""
AI Gesture-Controlled PowerPoint Presentation System (Improved)
Uses OpenCV, MediaPipe, and PyAutoGUI with stabilized tracking and geometric gesture recognition.
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque
import math

# --- Configuration ---
SWIPE_THRESHOLD = 40        # Pixels required to register a swipe
GESTURE_HOLD_TIME = 0.3     # How long a gesture must be held to be valid (debouncing)

class HandSmoother:
    """Filters hand landmarks to reduce jitter using Exponential Moving Average."""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_landmarks = None

    def update(self, landmarks, image_shape):
        h, w = image_shape[:2]
        # Convert landmarks to numpy array for vector math
        current_landmarks = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
        
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return current_landmarks
            
        # EMA Filter: Smooth = Alpha * Current + (1 - Alpha) * Previous
        smoothed = self.alpha * current_landmarks + (1 - self.alpha) * self.prev_landmarks
        self.prev_landmarks = smoothed
        return smoothed

class GesturePresentationController:
    def __init__(self, camera_index=0, smoothing=0.6, mouse_speed=1.5, width=640, height=480):
        # Configuration
        self.camera_index = camera_index
        self.mouse_speed = mouse_speed
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Smoothing
        self.smoother = HandSmoother(alpha=smoothing)
        
        # Gesture State
        self.last_gesture = None
        self.gesture_start_time = 0
        self.last_action_time = 0
        
        # Movement tracking for Swipes
        self.center_history = deque(maxlen=10)
        
        # 3-Second Window Logic
        self.control_window_active = False
        self.control_window_start = 0
        self.control_window_duration = 2.0
        
        # Mouse Mode State
        self.mouse_mode = False
        self.mouse_clicked = False
        self.last_mouse_pos = None
        
        # Screen size for mouse mapping
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Camera Setup
        self.cap = None
        self.display_width = width
        self.display_height = height
        
        # PyAutoGUI Settings
        pyautogui.FAILSAFE = False # Disable failsafe for corner reaching
        pyautogui.PAUSE = 0.0 # No delay for smoother mouse

        print(f"🤖 AI Presentation Control Initialized (Cam: {camera_index}, Res: {width}x{height}, Smooth: {smoothing})")

    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("❌ Error: Cannot access camera!")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        return True

    def get_extended_fingers(self, landmarks_np):
        """
        Detects extended fingers using geometric distance from wrist.
        Robust against hand rotation.
        """
        extended = []
        WRIST = 0
        finger_tips = [8, 12, 16, 20]   # Index, Middle, Ring, Pinky
        finger_pips = [6, 10, 14, 18]   # Lower joints
        
        wrist_pos = landmarks_np[WRIST]
        
        # 1. Check Thumb
        thumb_tip = landmarks_np[4]
        thumb_ip = landmarks_np[3]
        index_mcp = landmarks_np[5]
        
        if np.linalg.norm(thumb_tip - index_mcp) > np.linalg.norm(thumb_ip - index_mcp) * 1.2:
             extended.append('thumb')

        # 2. Check Fingers
        names = ['index', 'middle', 'ring', 'pinky']
        for tip_idx, pip_idx, name in zip(finger_tips, finger_pips, names):
            tip_pos = landmarks_np[tip_idx]
            pip_pos = landmarks_np[pip_idx]
            
            dist_tip = np.linalg.norm(tip_pos - wrist_pos)
            dist_pip = np.linalg.norm(pip_pos - wrist_pos)
            
            if dist_tip > dist_pip * 1.1: 
                extended.append(name)
                
        return extended

    def detect_dynamic_gesture(self, current_center):
        """Detects swipes based on movement history."""
        self.center_history.append(current_center)
        if len(self.center_history) < 4:
            return None
            
        start_pos = self.center_history[0]
        end_pos = self.center_history[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        if abs(dx) > SWIPE_THRESHOLD and abs(dy) < SWIPE_THRESHOLD / 2:
            if dx > 0: return "swipe_right"
            else: return "swipe_left"
            
        return None

    def classify_pose(self, extended_fingers):
        """Maps extended fingers to a static pose name."""
        count = len(extended_fingers)
        
        if count == 5: return "open_palm"
        if count == 0: return "fist"
        if count == 1 and 'index' in extended_fingers: return "point_up"
        if count == 1 and 'thumb' in extended_fingers: return "thumbs_up"
        if count == 2 and 'index' in extended_fingers and 'middle' in extended_fingers: return "peace"
        if count == 2 and 'thumb' in extended_fingers and 'pinky' in extended_fingers: return "shaka"
        if count == 1 and 'middle' in extended_fingers: return "middle_finger"
        
        return "unknown"

    def handle_mouse_control(self, landmarks_np, frame_shape):
        """
        Controls the mouse cursor using the index finger tip.
        Performs click if thumb and index pinch.
        """
        h, w = frame_shape[:2]
        index_tip = landmarks_np[8]
        thumb_tip = landmarks_np[4]
        
        # Map Coordinates (Camera pixel -> Screen pixel)
        # We map a central "active area" of the camera to the full screen
        margin_x = w * 0.2
        margin_y = h * 0.2
        active_w = w - 2 * margin_x
        active_h = h - 2 * margin_y
        
        # Normalize relative to active area
        rel_x = (index_tip[0] - margin_x) / active_w
        rel_y = (index_tip[1] - margin_y) / active_h
        
        # Clamp to 0-1
        rel_x = max(0, min(1, rel_x))
        rel_y = max(0, min(1, rel_y))
        
        # Map to screen
        target_x = int(rel_x * self.screen_w)
        target_y = int(rel_y * self.screen_h)
        
        # Move Mouse
        pyautogui.moveTo(target_x, target_y)
        
        # Click Detection (Pinch)
        # Calculate distance between thumb and index tip in pixel space
        pinch_dist = np.linalg.norm(index_tip - thumb_tip)
        
        # Threshold depends on hand size (distance to wrist), but fixed pixel approx 30-40 works for 720p
        if pinch_dist < 40: 
            if not self.mouse_clicked:
                pyautogui.click()
                self.mouse_clicked = True
                return "click"
        else:
            self.mouse_clicked = False
            
        return "move"

    def process_logic(self, pose, dynamic_gesture, landmarks_np, frame_shape):
        current_time = time.time()
        
        # Toggle Mouse Mode (Hold Peace Sign for > 1s)
        if pose == "peace":
            if self.gesture_start_time == 0:
                self.gesture_start_time = current_time
            elif current_time - self.gesture_start_time > 1.0:
                # Toggle
                self.mouse_mode = not self.mouse_mode
                self.gesture_start_time = 0 # Reset
                print(f"🖱️ Mouse Mode: {'ON' if self.mouse_mode else 'OFF'}")
                return f"mouse_mode_{'on' if self.mouse_mode else 'off'}"
        else:
            self.gesture_start_time = 0

        # --- MOUSE MODE ---
        if self.mouse_mode:
            return self.handle_mouse_control(landmarks_np, frame_shape)

        # --- GESTURE MODE ---
        # Activation
        if pose == "point_up" and not self.control_window_active:
            self.control_window_active = True
            self.control_window_start = current_time
            return "activate"

        if self.control_window_active:
            # Expiry check
            if current_time - self.control_window_start > self.control_window_duration:
                self.control_window_active = False
                return "expired"
                
            # Cooldown
            if current_time - self.last_action_time < 0.8:
                return None

            command = None
            # Priority: Dynamic Swipes > Static Poses
            if dynamic_gesture == "swipe_right":
                command = "next_slide"
                pyautogui.press('right')
            elif dynamic_gesture == "swipe_left":
                command = "prev_slide"
                pyautogui.press('left')
            elif pose == "shaka": # 🤙 Alternative for Next Slide
                command = "next_slide"
                pyautogui.press('right')
            #elif pose == "open_palm":
                #command = "play_pause"
                #pyautogui.press('space')
            elif pose == "fist":
                command = "stop"
                pyautogui.press('escape')
            #elif pose == "thumbs_up":
                #command = "zoom_in"
                #pyautogui.hotkey('ctrl', '+') 
            elif dynamic_gesture == "middle_finger":
                command = "close"
                pyautogui.press('Q')

            if command:
                self.last_action_time = current_time
                
                # Special Case: Navigation commands force re-activation (Strict Mode)
                if command in ["next_slide", "prev_slide"]:
                    self.control_window_active = False
                else:
                    # Other commands (Zoom, Play, etc.) keep the window open
                    self.control_window_start = current_time 
                
                return command
                
        return None

    def draw_ui(self, frame, pose, feedback, extended_fingers):
        h, w = frame.shape[:2]
        
        # Top Status Bar
        cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
        
        # Mode Indicator
        if self.mouse_mode:
            cv2.putText(frame, "MOUSE MODE (Pinch to Click)", (30, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.circle(frame, (w-50, 30), 15, (0, 255, 255), -1)
        else:
            status = "ACTIVE" if self.control_window_active else "STANDBY"
            color = (0, 255, 0) if self.control_window_active else (100, 100, 100)
            cv2.putText(frame, f"MODE: GESTURE ({status})", (30, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Timer Bar
            if self.control_window_active:
                time_left = max(0, self.control_window_duration - (time.time() - self.control_window_start))
                bar_w = int((time_left / self.control_window_duration) * w)
                cv2.rectangle(frame, (0, 55), (bar_w, 60), (0, 255, 0), -1)

        # Feedback Text
        if feedback and not feedback.startswith("move"):
            cv2.putText(frame, str(feedback).upper(), (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Pose Debug
        cv2.putText(frame, f"Pose: {pose}", (w - 200, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Helper Text at Bottom
        help_text = "☝ Point: ACTIVATE | 🤙/Swipe: Next | ✋ Play | ✊ Stop | ✌️ Mouse"
        cv2.putText(frame, help_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def run(self):
        if not self.start_camera(): return
        print("\n🚀 Ready. Hold 'Peace' ✌️ to toggle Mouse Mode.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                pose = "none"
                dynamic_gesture = None
                feedback = None
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Smooth & Analyze
                    smoothed_np = self.smoother.update(hand_landmarks.landmark, frame.shape)
                    extended_fingers = self.get_extended_fingers(smoothed_np)
                    pose = self.classify_pose(extended_fingers)
                    
                    palm_center = smoothed_np[9] 
                    dynamic_gesture = self.detect_dynamic_gesture(palm_center)
                    
                    # Process
                    feedback = self.process_logic(pose, dynamic_gesture, smoothed_np, frame.shape)
                    
                    # Draw
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Draw Pointer in Mouse Mode
                    if self.mouse_mode:
                        idx_tip = smoothed_np[8].astype(int)
                        cv2.circle(frame, tuple(idx_tip), 10, (0, 255, 255), 2)

                self.draw_ui(frame, pose, feedback, [])
                cv2.imshow('AI Gesture Control Pro', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Default run
    GesturePresentationController().run()