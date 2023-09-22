import numpy as np
import librosa
import math


class Augmentation:
    """
    Class implementing different augmentations techniques
    """

    @staticmethod
    def add_white_noise(signal, noise_factor):
        """
        Add gaussian/white noise to the signal
        Parameters
        ----------
        signal : np.array
            Sound signal
        noise_factor : float
            Bigger factor -> noisier signal

        Returns
        -------
        np.array
            Noisy signal
        """
        noise = np.random.normal(0, signal.std(), signal.size)
        return signal + noise * noise_factor

    @staticmethod
    def time_stretch(signal, stretch_rate):
        return librosa.effects.time_stretch(signal, rate=stretch_rate)

    @staticmethod
    def pitch_scale(signal, sr, n_semitones):
        return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_semitones)

    @staticmethod
    def invert_polarity(signal):
        return signal * -1

    @staticmethod
    def random_gain(signal, min_gain_factor, maximum_gain_factor):
        gain_factor = np.random.uniform(min_gain_factor, maximum_gain_factor)
        return signal * gain_factor

    @staticmethod
    def add_background_noise(signal, noise, noise_factor):
        if len(noise) < len(signal):
            noise = np.tile(noise, (math.ceil(len(signal) / len(noise))))
        if len(noise) > len(signal):
            start = np.random.randint(len(noise) - len(signal))
            noise = noise[start:start+len(signal)]
        return signal + noise * noise_factor
