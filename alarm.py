"""
Alarm handling via pygame.mixer.

Kept in its own thread so the audio call never blocks or interferes
with the OpenCV video loop / frame rate.
"""

import threading
import time

import pygame

from config import ALARM_SOUND_PATH, ALARM_COOLDOWN_SECONDS


class Alarm:
    def __init__(self, sound_path: str = ALARM_SOUND_PATH):
        pygame.mixer.init()
        self.sound = pygame.mixer.Sound(sound_path)
        self._last_played = 0.0
        self._lock = threading.Lock()

    def trigger(self):
        """Play the alarm in a background thread, respecting a cooldown."""
        now = time.time()
        with self._lock:
            if now - self._last_played < ALARM_COOLDOWN_SECONDS:
                return
            self._last_played = now

        threading.Thread(target=self._play, daemon=True).start()

    def _play(self):
        self.sound.play()

    def stop(self):
        self.sound.stop()
