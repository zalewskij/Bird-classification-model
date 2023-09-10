import numpy as np
import librosa

class Augmentation:
    def __init__(self, name, age):
        pass

    @staticmethod
    def add_white_noise(signal, noise_factor):
        noise = np.random.normal(0, signal.std(), signal.size)
        return signal + noise * noise_factor

    @staticmethod
    def time_stretch(signal, stretch_rate):
        return librosa.effects.time_stretch(signal, rate = stretch_rate)

    @staticmethod
    def pitch_scale(signal, sr, n_semitones):
        return librosa.effects.pitch_shift(signal, sr = sr, n_steps = n_semitones)

    @staticmethod
    def invert_polarity(signal):
        return signal * -1

    @staticmethod
    def random_gain(signal, min_gain_factor, maximum_gain_factor):
        gain_factor = np.random.uniform(min_gain_factor, maximum_gain_factor)
        return signal*gain_factor








