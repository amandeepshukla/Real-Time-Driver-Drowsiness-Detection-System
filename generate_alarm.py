"""
One-time utility: generates a simple beeping audio.wav so the project
runs out of the box without needing an external audio file.

Run once:
    python generate_audio.py

Feel free to replace assets/audio.wav with your own sound afterward.
"""

import math
import struct
import wave

OUT_PATH = "assets/audio.wav"
SAMPLE_RATE = 44100
FREQ_HZ = 1000
BEEP_MS = 300
GAP_MS = 150
BEEPS = 3


def generate_beep_samples(freq, duration_ms, sample_rate):
    n_samples = int(sample_rate * duration_ms / 1000)
    return [
        int(32767 * 0.5 * math.sin(2 * math.pi * freq * (i / sample_rate)))
        for i in range(n_samples)
    ]


def generate_silence_samples(duration_ms, sample_rate):
    return [0] * int(sample_rate * duration_ms / 1000)


def main():
    samples = []
    for _ in range(BEEPS):
        samples.extend(generate_beep_samples(FREQ_HZ, BEEP_MS, SAMPLE_RATE))
        samples.extend(generate_silence_samples(GAP_MS, SAMPLE_RATE))

    with wave.open(OUT_PATH, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(
            b"".join(struct.pack("<h", s) for s in samples)
        )

    print(f"[INFO] Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
