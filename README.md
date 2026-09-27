# 🚗 Driver Drowsiness Detection System

A real-time system that watches your eyes through a webcam and sounds an alarm the moment it detects signs of drowsiness — built as a practical take on the classic Eye Aspect Ratio (EAR) approach used in driver-safety research. Now with an actual GUI instead of just a terminal window and a video feed popping up.

## Why I built this

Drowsy driving is one of those problems that's easy to underestimate until it's too late — a few seconds of microsleep at the wheel is all it takes. I wanted to build something that actually demonstrates real-time computer vision working end-to-end: camera → face detection → landmark tracking → a decision rule → an actionable alert. It's also a nice showcase of tying together OpenCV, dlib, and audio handling in one clean pipeline.

The original version just ran in a raw OpenCV window with a `q` to quit and not much else. Worked fine, but it didn't really *look* like a finished project, so I sat down and wrapped the whole thing in a proper Tkinter interface — start/stop buttons, a live EAR readout, a status bar, the works.

## How it works

1. **Capture** — OpenCV grabs frames straight from the webcam, unmodified.
2. **Detect** — `dlib`'s frontal face detector locates the face, then its 68-point shape predictor maps facial landmarks, including both eyes.
3. **Measure** — For each eye, the Eye Aspect Ratio (EAR) is calculated from 6 landmark points. EAR stays fairly constant while eyes are open and drops sharply when they close.
4. **Decide** — If EAR stays below a threshold for a set number of consecutive frames (not just a single blink), the system flags drowsiness.
5. **Alert** — A `pygame`-driven alarm plays in a background thread, so it never stutters the video loop.
6. **Display** — All of the above now runs inside a Tkinter window instead of a bare OpenCV frame, with the camera feed rendered live and controls sitting right next to it.

```
drowsiness_detection/
├── main.py              # original capture loop + detection + alarm trigger (CLI version)
├── gui.py                # Tkinter GUI wrapper around the same detection pipeline
├── ear_utils.py          # EAR math, landmark helpers
├── alarm.py               # threaded pygame alarm wrapper
├── config.py               # all the tunable knobs in one place
├── generate_alarm.py        # generates assets/alarm.wav (self-contained, no external audio needed)
├── requirements.txt
├── models/                  # drop shape_predictor_68_face_landmarks.dat here
└── assets/
    └── alarm.wav              # pre-generated alert tone
```

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/drowsiness-detection.git
cd drowsiness-detection
pip install -r requirements.txt
```

> Heads up: `dlib` compiles from source on some systems and needs CMake + a C++ compiler. If pip gives you trouble on Windows, `conda install -c conda-forge dlib` is usually the easier path.

### 2. Grab the landmark model

Not bundled in the repo since it's ~100MB. Download it and drop it in `models/`:

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat models/
```

### 3. Run it

You've got two options depending on how you want to use it:

**GUI version (recommended):**

```bash
python gui.py
```

This opens a proper window — hit **Start Detection** to fire up the camera and begin monitoring, **Test Alarm** if you just want to check the sound works without sitting in front of the webcam, and **Stop Detection** when you're done. The EAR value updates live in the side panel so you can actually see the number moving instead of guessing what's going on under the hood.

**Original CLI version:**

```bash
python main.py
```

- Green outline around each eye = landmarks being tracked live
- EAR value shown in the corner, updating every frame
- Red **"DROWSINESS ALERT!"** + alarm sound once eyes stay closed past the threshold
- `q` to quit

## Tuning it to you

Everyone's baseline EAR is a little different depending on eye shape, glasses, and lighting. If you're getting false alarms (or none at all), open `config.py`:

| Setting | What it controls |
|---|---|
| `EAR_THRESHOLD` | EAR below this counts as "eyes closed" |
| `EAR_CONSEC_FRAMES` | How many frames in a row before the alarm fires |
| `ALARM_COOLDOWN_SECONDS` | Minimum gap between repeated alarms |

Tip: run it once, watch the live EAR value with your eyes open vs. deliberately closed, and set the threshold somewhere in between. The GUI's side panel makes this a lot easier than squinting at the corner of a video frame.

## Tech stack

`Python` · `OpenCV` · `dlib` · `SciPy` · `pygame` · `Tkinter` · `Pillow`

## Possible next steps

- Swap `dlib` for MediaPipe FaceMesh for lighter-weight, GPU-free landmark detection
- Add yawning detection alongside EAR
- Log drowsiness events with timestamps, maybe show them in the GUI as a small history panel
- Package the GUI version as a standalone `.exe` / desktop app so it doesn't need a Python install to run

## Author

Built by **Amandeep** — BCA student, full-stack + cybersecurity background. Feel free to connect or drop feedback if you try this out.

## License

MIT — use it, fork it, improve it.
