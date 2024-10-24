import random

import pygame


class AudioGenerator:
    def __init__(self):
        # Initialise Pygame + Mixer
        pygame.init()
        pygame.mixer.init()

        # Load Flacs
        emotions = ["happy", "sad", "angry", "surprise", "disgust", "fear", "neutral"]
        self.audio_files = {
            emotion: [
                pygame.mixer.Sound(f"src/sounds/{emotion}_{i}.flac")
                for i in range(1, 4)
            ]
            for emotion in emotions
        }

        self.current_sound = None

    # Play/Fade Functions
    def play_audio(self, emotion, fade_in_ms=0):
        sound = random.choice(self.audio_files[emotion])
        sound.play(loops=4, fade_ms=fade_in_ms)
        self.current_sound = sound

    def fade_out_audio(self, fade_out_ms):
        if self.current_sound:
            self.current_sound.fadeout(fade_out_ms)
