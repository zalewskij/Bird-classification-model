import math
import os
import random
import torch
import torchaudio
from torchaudio.functional import pitch_shift, bandpass_biquad
from torch.utils.data import Dataset
from torchaudio.transforms import Resample


class BackgroundNoises(Dataset):
    def __init__(self, noises_dir):
        self.recording_dir = noises_dir
        self.noises = os.listdir(self.recording_dir)
        self.target_sr = 32000

    def __len__(self):
        return len(self.noises)

    def __getitem__(self, idx):
        file = os.path.join(self.recording_dir, self.noises[idx])
        audio, sr = torchaudio.load(file)
        if sr != self.target_sr:
            resampler = Resample(orig_freq=sr, new_freq=self.target_sr)
            audio = resampler(audio)
        return audio


class AddBackgroundNoise(torch.nn.Module):
    def __init__(self, min_factor, max_factor, noises_dir):
        super().__init__()
        self.min = min_factor
        self.max = max_factor
        self.noises = BackgroundNoises(noises_dir)

    def forward(self, waveform: torch.Tensor):
        noise_factor = random.uniform(self.min, self.max)

        n = random.randint(0, self.noises.__len__())
        noise = self.noises[n]
        if noise.size()[1] < waveform.size()[1]:
            noise = torch.cat([noise] * math.ceil(waveform.size()[1] / noise.size()[1]), dim=1)
        if noise.size()[1] > waveform.size()[1]:
            start = torch.randint(0, noise.size()[1] - waveform.size()[1], (1,))
            noise = noise[:, start:start + waveform.size()[1]]
        return waveform + noise * noise_factor


class InvertPolarity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, waveform: torch.Tensor):
        return waveform * -1


class AddWhiteNoise(torch.nn.Module):
    def __init__(self, min_factor, max_factor):
        super().__init__()
        self.min = min_factor
        self.max = max_factor

    def forward(self, waveform: torch.Tensor):
        noise_factor = random.uniform(self.min, self.max)

        noise = torch.normal(0, waveform.std(), waveform.shape)
        noisy_signal = waveform + noise * noise_factor

        return noisy_signal


class PitchShifting(torch.nn.Module):
    def __init__(self, sr, min_semitones, max_semitones):
        super().__init__()
        self.min = min_semitones
        self.max = max_semitones
        self.sr = sr

    def forward(self, waveform: torch.Tensor):
        n_semitones = random.randint(self.min, self.max)

        return pitch_shift(waveform=waveform, sample_rate=self.sr, n_steps=n_semitones)


class RandomGain(torch.nn.Module):
    def __init__(self, min_factor, max_factor):
        super().__init__()
        self.min = min_factor
        self.max = max_factor

    def forward(self, waveform: torch.Tensor):
        gain_factor = random.uniform(self.min, self.max)

        return waveform * gain_factor


class TimeShift(torch.nn.Module):
    def __init__(self, min_factor, max_factor):
        super().__init__()
        self.min = min_factor
        self.max = max_factor

    def forward(self, waveform: torch.Tensor):
        shift_factor = random.uniform(self.min, self.max)

        max_shift = int(waveform.size()[1] * shift_factor)
        waveform = torch.roll(waveform, random.randint(-max_shift, max_shift))
        return waveform


# FILTERING
class BandPass(torch.nn.Module):
    def __init__(self, sr, central_freq, Q=0.707):
        super().__init__()
        self.sr = sr
        self.central_freq = central_freq
        self.Q = Q

    def forward(self, waveform: torch.Tensor):
        waveform = bandpass_biquad(waveform, self.sr, self.central_freq, self.Q)
        return waveform


# Currently NOT used
class RandomChunk(torch.nn.Module):
    def __init__(self, sr, min_factor, max_factor):
        super().__init__()
        self.sr = sr
        self.chunk_size = random.uniform(min_factor, max_factor)

    def forward(self, waveform: torch.Tensor):
        seconds = int(self.chunk_size * waveform.size()[1] / self.sr)
        seconds *= self.sr
        start = random.randint(0, waveform.size()[1] - seconds)
        signal = waveform[:, start: start + seconds]
        print(waveform.shape)
        return signal
