"""
Tkinter GUI for the Driver Drowsiness Detection system.

Wraps the existing detection pipeline (config.py, ear_utils.py, alarm.py)
in a class-based, non-blocking Tkinter interface. The OpenCV capture loop
runs via `window.after()` instead of a blocking `while True`, so the UI
stays responsive.

Run:
    python gui.py
"""

import time
import tkinter as tk
from tkinter import ttk

import cv2
import dlib
import imutils
from PIL import Image, ImageTk

from config import (
    SHAPE_PREDICTOR_PATH,
    EAR_THRESHOLD,
    EAR_CONSEC_FRAMES,
    CAMERA_INDEX,
    FRAME_WIDTH,
    LEFT_EYE_POINTS,
    RIGHT_EYE_POINTS,
)
from ear_utils import shape_to_np, average_ear
from alarm import Alarm


class DrowsinessDetectionApp:
    """Main application window."""

    def __init__(self, window: tk.Tk):
        self.window = window
        self.window.title("Driver Drowsiness Detection")
        self.window.geometry("880x600")
        self.window.minsize(760, 520)
        self.window.configure(bg="#1e1e2e")

        # --- Detection state ---
        self.cap = None
        self.detector = None
        self.predictor = None
        self.alarm = None
        self.running = False
        self.frame_counter = 0  # consecutive frames below EAR threshold
        self._after_id = None

        self._build_style()
        self._build_layout()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background="#1e1e2e")
        style.configure(
            "Control.TFrame", background="#27293d", relief="flat"
        )
        style.configure(
            "TButton",
            font=("Segoe UI", 10, "bold"),
            padding=10,
            background="#3b3f5c",
            foreground="#ffffff",
            borderwidth=0,
        )
        style.map("TButton", background=[("active", "#565a7a")])

        style.configure(
            "Status.TLabel",
            font=("Segoe UI", 13, "bold"),
            background="#27293d",
            foreground="#8be9fd",
            padding=8,
        )
        style.configure(
            "Header.TLabel",
            font=("Segoe UI", 16, "bold"),
            background="#1e1e2e",
            foreground="#ffffff",
        )
        style.configure(
            "Ear.TLabel",
            font=("Segoe UI", 10),
            background="#27293d",
            foreground="#cdd6f4",
        )

    def _build_layout(self):
        # --- Header ---
        header = ttk.Label(
            self.window, text="🚗  Driver Drowsiness Detection", style="Header.TLabel"
        )
        header.pack(side="top", pady=(12, 6))

        # --- Body: video (left) + control panel (right) ---
        body = ttk.Frame(self.window)
        body.pack(fill="both", expand=True, padx=12, pady=6)

        # Video feed
        video_frame = ttk.Frame(body, style="Control.TFrame")
        video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.video_label = tk.Label(
            video_frame, bg="#000000", text="Camera feed will appear here",
            fg="#8be9fd", font=("Segoe UI", 11),
        )
        self.video_label.pack(fill="both", expand=True, padx=8, pady=8)

        # Control panel
        control_frame = ttk.Frame(body, style="Control.TFrame", width=220)
        control_frame.pack(side="right", fill="y")
        control_frame.pack_propagate(False)

        ttk.Label(
            control_frame, text="Controls", font=("Segoe UI", 12, "bold"),
            background="#27293d", foreground="#ffffff",
        ).pack(pady=(14, 10))

        self.start_btn = ttk.Button(
            control_frame, text="▶  Start Detection", command=self.start_detection
        )
        self.start_btn.pack(fill="x", padx=16, pady=6)

        self.stop_btn = ttk.Button(
            control_frame, text="⏹  Stop Detection", command=self.stop_detection,
            state="disabled",
        )
        self.stop_btn.pack(fill="x", padx=16, pady=6)

        self.test_alarm_btn = ttk.Button(
            control_frame, text="🔔  Test Alarm", command=self.test_alarm
        )
        self.test_alarm_btn.pack(fill="x", padx=16, pady=6)

        self.quit_btn = ttk.Button(
            control_frame, text="✖  Quit", command=self.quit_app
        )
        self.quit_btn.pack(fill="x", padx=16, pady=(6, 16))

        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", padx=16, pady=8)

        # EAR readout
        self.ear_var = tk.StringVar(value="EAR: --")
        ttk.Label(
            control_frame, textvariable=self.ear_var, style="Ear.TLabel"
        ).pack(pady=(4, 12))

        # --- Status bar ---
        status_bar = ttk.Frame(self.window, style="Control.TFrame")
        status_bar.pack(side="bottom", fill="x")

        self.status_var = tk.StringVar(value="Status: Inactive")
        self.status_label = ttk.Label(
            status_bar, textvariable=self.status_var, style="Status.TLabel"
        )
        self.status_label.pack(pady=6)

    # ------------------------------------------------------------------
    # Detection control
    # ------------------------------------------------------------------
    def start_detection(self):
        if self.running:
            return

        self._set_status("Status: Initializing...", "#f1fa8c")

        # --- Placeholder: load dlib detector/predictor once ---
        # (Loaded lazily here so the app opens instantly; move to __init__
        # if you'd rather pay the load cost at startup.)
        if self.detector is None:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

        if self.alarm is None:
            self.alarm = Alarm()

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        time.sleep(0.3)  # brief camera warm-up

        if not self.cap.isOpened():
            self._set_status("Status: Camera error!", "#ff5555")
            self.cap = None
            return

        self.running = True
        self.frame_counter = 0
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._set_status("Status: Monitoring", "#50fa7b")

        self.update_frame()

    def stop_detection(self):
        self.running = False
        if self._after_id is not None:
            self.window.after_cancel(self._after_id)
            self._after_id = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.video_label.config(
            image="", text="Camera feed will appear here", fg="#8be9fd"
        )
        self.video_label.image = None

        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.ear_var.set("EAR: --")
        self._set_status("Status: Inactive", "#8be9fd")

    def test_alarm(self):
        """Trigger the alarm sound without needing an active detection loop."""
        if self.alarm is None:
            self.alarm = Alarm()
        self.alarm.trigger()
        self._set_status("Status: Test Alarm Triggered", "#ffb86c")
        # Revert the status label after a moment, but only if detection
        # hasn't changed state in the meantime.
        self.window.after(1500, self._restore_status_after_test)

    def _restore_status_after_test(self):
        if self.running:
            self._set_status("Status: Monitoring", "#50fa7b")
        else:
            self._set_status("Status: Inactive", "#8be9fd")

    def quit_app(self):
        self.stop_detection()
        self.window.destroy()

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------
    def update_frame(self):
        """Grab one frame, run detection, render it, and reschedule itself.

        Uses window.after() instead of a blocking while-loop so Tkinter's
        event loop (button clicks, window redraws) keeps running smoothly.
        """
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._set_status("Status: Camera read failed", "#ff5555")
            self.stop_detection()
            return

        frame = imutils.resize(frame, width=FRAME_WIDTH)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Placeholder: face + landmark detection ---
        faces = self.detector(gray, 0)

        drowsy_this_frame = False

        for face in faces:
            shape = self.predictor(gray, face)
            landmarks = shape_to_np(shape)

            # --- Placeholder: EAR calculation (ear_utils.py) ---
            ear, left_eye, right_eye = average_ear(
                landmarks, LEFT_EYE_POINTS, RIGHT_EYE_POINTS
            )

            cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

            self.ear_var.set(f"EAR: {ear:.3f}")

            if ear < EAR_THRESHOLD:
                self.frame_counter += 1
                if self.frame_counter >= EAR_CONSEC_FRAMES:
                    drowsy_this_frame = True
                    # --- Placeholder: alarm trigger (alarm.py) ---
                    self.alarm.trigger()
                    cv2.putText(
                        frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )
            else:
                self.frame_counter = 0

        # Update status label based on this frame's result
        if drowsy_this_frame:
            self._set_status("Status: DROWSY!", "#ff5555")
        elif self.running:
            self._set_status("Status: Monitoring", "#50fa7b")

        self._render_frame(frame)

        # Reschedule — ~30 FPS. Adjust the delay to taste / CPU budget.
        self._after_id = self.window.after(33, self.update_frame)

    def _render_frame(self, frame):
        """Convert a BGR OpenCV frame to a Tkinter-displayable image."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)

        self.video_label.config(image=photo, text="")
        self.video_label.image = photo  # keep a reference, avoid GC

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_status(self, text, color):
        self.status_var.set(text)
        self.status_label.configure(foreground=color)


def main():
    window = tk.Tk()
    app = DrowsinessDetectionApp(window)
    window.protocol("WM_DELETE_WINDOW", app.quit_app)
    window.mainloop()


if __name__ == "__main__":
    main()
