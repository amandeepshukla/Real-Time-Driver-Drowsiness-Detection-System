"""Central configuration for the Driver Drowsiness Detection system.
Tune these values based on lighting, camera, and user calibration."""
# --- Paths ---
SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
ALARM_SOUND_PATH = "assets/audio.wav"
# --- EAR (Eye Aspect Ratio) thresholds ---
# EAR drops sharply when eyes close. Typical open-eye EAR ~0.25-0.35.
EAR_THRESHOLD = 0.25
# Number of consecutive frames the EAR must stay below threshold
# before the audio is triggered. At ~20-30 FPS, 20 frames ~= 0.7-1s.
EAR_CONSEC_FRAMES = 20
# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
# --- dlib 68-point facial landmark indices for eyes ---
# (start, end) — end is exclusive, matching Python slicing
LEFT_EYE_POINTS = (42, 48)
RIGHT_EYE_POINTS = (36, 42)
# --- audio ---
ALARM_COOLDOWN_SECONDS = 3  # avoid re-triggering the audio every frame