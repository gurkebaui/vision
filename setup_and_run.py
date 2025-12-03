#!/usr/bin/env python3
"""
Setup and Launcher for AI Gesture Control
Provides a GUI to configure and start the application.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
import subprocess
import threading

# Try to import the main controller class
# We need to add the current directory to path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Gesture Control - Launcher")
        self.root.geometry("500x450")
        self.root.resizable(False, False)
        
        # Style
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TLabel", font=("Helvetica", 11))
        
        # Header
        header_frame = tk.Frame(root, bg="#2c3e50", height=80)
        header_frame.pack(fill="x")
        tk.Label(header_frame, text="AI Gesture Control", font=("Helvetica", 18, "bold"), 
                 bg="#2c3e50", fg="white").pack(pady=20)

        # Main Content
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # 1. Camera Selection
        ttk.Label(main_frame, text="Camera Index:").grid(row=0, column=0, sticky="w", pady=10)
        self.camera_var = tk.IntVar(value=0)
        camera_spin = ttk.Spinbox(main_frame, from_=0, to=10, textvariable=self.camera_var, width=5)
        camera_spin.grid(row=0, column=1, sticky="w", pady=10)
        
        # 2. Motion Smoothing
        ttk.Label(main_frame, text="Motion Smoothing:").grid(row=1, column=0, sticky="w", pady=10)
        self.smooth_var = tk.DoubleVar(value=0.6)
        smooth_scale = ttk.Scale(main_frame, from_=0.1, to=0.95, variable=self.smooth_var, orient="horizontal", length=200)
        smooth_scale.grid(row=1, column=1, sticky="w", pady=10)
        # Label to show value
        self.smooth_label = ttk.Label(main_frame, text="0.6")
        self.smooth_label.grid(row=1, column=2, padx=5)
        smooth_scale.configure(command=lambda v: self.smooth_label.config(text=f"{float(v):.2f}"))

        # 3. Mouse Sensitivity (Logic placeholder)
        ttk.Label(main_frame, text="Mouse Speed:").grid(row=2, column=0, sticky="w", pady=10)
        self.speed_var = tk.DoubleVar(value=1.5)
        speed_scale = ttk.Scale(main_frame, from_=0.5, to=5.0, variable=self.speed_var, orient="horizontal", length=200)
        speed_scale.grid(row=2, column=1, sticky="w", pady=10)
        self.speed_label = ttk.Label(main_frame, text="1.5")
        self.speed_label.grid(row=2, column=2, padx=5)
        speed_scale.configure(command=lambda v: self.speed_label.config(text=f"{float(v):.1f}"))

        # Instructions
        info_text = (
            "Instructions:\n"
            "• 'Peace' (✌️) > 1s: Toggle Mouse Mode\n"
            "• Point Up (☝️): Activate Gesture Mode (3s window)\n"
            "• Gestures: Palm=Play, Fist=Stop, Swipe=Slide"
        )
        info_label = tk.Label(main_frame, text=info_text, justify="left", bg="#f0f0f0", relief="sunken", padx=10, pady=10)
        info_label.grid(row=3, column=0, columnspan=3, sticky="we", pady=20)

        # Start Button
        self.start_btn = tk.Button(main_frame, text="🚀 START CONTROL", bg="#27ae60", fg="white", 
                                   font=("Helvetica", 12, "bold"), command=self.start_app)
        self.start_btn.grid(row=4, column=0, columnspan=3, sticky="we", pady=10)

        # Status
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="gray")
        self.status_label.grid(row=5, column=0, columnspan=3)

    def start_app(self):
        try:
            # Check imports
            import cv2
            import mediapipe
            import pyautogui
        except ImportError as e:
            messagebox.showerror("Missing Packages", f"Please install requirements first!\nError: {e}")
            return

        self.start_btn.config(state="disabled", text="Running...")
        self.status_label.config(text="Application is running. Press 'q' in camera window to stop.")
        
        # Run in separate thread to keep GUI responsive
        thread = threading.Thread(target=self.run_controller_thread)
        thread.daemon = True
        thread.start()

    def run_controller_thread(self):
        try:
            from gesture_presentation_control import GesturePresentationController
            
            cam_idx = self.camera_var.get()
            smooth = self.smooth_var.get()
            speed = self.speed_var.get()
            
            controller = GesturePresentationController(
                camera_index=cam_idx,
                smoothing=smooth,
                mouse_speed=speed
            )
            controller.run()
            
        except Exception as e:
            messagebox.showerror("Error", f"Application crashed:\n{e}")
        finally:
            # Reset UI when done
            self.root.after(0, lambda: self.start_btn.config(state="normal", text="🚀 START CONTROL"))
            self.root.after(0, lambda: self.status_label.config(text="Stopped."))

def main():
    # Check if packages are installed
    required = ["opencv-python", "mediapipe", "pyautogui", "numpy"]
    missing = []
    
    # Minimal check before GUI
    for pkg in required:
        try:
            __import__(pkg.split('-')[0].replace('opencv', 'cv2').replace('pyautogui', 'pyautogui'))
        except ImportError:
            missing.append(pkg)
            
    if missing:
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
