import random

import numpy as np
import torch
from birdclassification.preprocessing.augmentation import (invert_polarity, add_white_noise, time_stretch,
                                                           random_gain, add_background_noise, time_shift, random_chunk)
from torchaudio.functional import pitch_shift


class InvertPolarity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, waveform: torch.Tensor):
        return invert_polarity(waveform)


class AddWhiteNoise(torch.nn.Module):
    def __init__(self, min_factor, max_factor):
        super().__init__()
        self.noise_factor = random.uniform(min_factor, max_factor)

    def forward(self, waveform: torch.Tensor):
        return add_white_noise(waveform, self.noise_factor)


class TimeStretch(torch.nn.Module):
    def __init__(self, min_rate, max_rate):
        super().__init__()
        self.stretch_rate = random.uniform(min_rate, max_rate)

    def forward(self, waveform: torch.Tensor):
        return time_stretch(waveform, self.stretch_rate)


class PitchShifting(torch.nn.Module):
    def __init__(self, sr, min_semitones, max_semitones):
        super().__init__()
        self.n_semitones = random.randint(min_semitones, max_semitones)
        self.sr = sr

    def forward(self, waveform: torch.Tensor):
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        return pitch_shift(waveform=waveform, sample_rate=self.sr, n_steps=self.n_semitones)


class RandomGain(torch.nn.Module):
    def __init__(self, min_factor, max_factor):
        super().__init__()
        self.gain_factor = random.uniform(min_factor, max_factor)

    def forward(self, waveform: torch.Tensor):
        return random_gain(waveform, self.gain_factor)


class AddBackgroundNoise(torch.nn.Module):
    def __init__(self, noise, min_factor, max_factor):
        super().__init__()
        self.noise = noise
        self.noise_factor = random.uniform(min_factor, max_factor)

    def forward(self, waveform: torch.Tensor):
        return add_background_noise(waveform, self.noise, self.noise_factor)


class TimeShift(torch.nn.Module):
    def __init__(self, min_factor, max_factor):
        super().__init__()
        self.shift_factor = random.uniform(min_factor, max_factor)

    def forward(self, waveform: torch.Tensor):
        return time_shift(waveform, self.shift_factor)


class RandomChunk(torch.nn.Module):
    def __init__(self, sr, min_factor, max_factor):
        super().__init__()
        self.sr = sr
        self.chunk_size = random.uniform(min_factor, max_factor)

    def forward(self, waveform: torch.Tensor):
        return random_chunk(waveform, self.sr, self.chunk_size)
