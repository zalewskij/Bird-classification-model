import math
import random
import torch
from birdclassification.preprocessing.noises_dataset import NoisesDataset
from torchaudio.functional import pitch_shift, bandpass_biquad


class AddBackgroundNoise(torch.nn.Module):
    """
       Adds another audio to the signal
    """

    def __init__(self, min_factor, max_factor, df=None, noises_dir=''):
        """
        Parameters
        ----------
        min_factor : float
            Lower-bound for a noise factor
        max_factor : float
            Upper-bound for a noise factor
        noises_dir: string
            Path to a directory with environmental noises
        """
        super().__init__()
        self.min = min_factor
        self.max = max_factor
        self.noises = NoisesDataset(df=df, recordings__dir=noises_dir) if noises_dir != '' and df is not None else None


    def forward(self, waveform: torch.Tensor):
        """
        Parameters
        ----------
        waveform: torch.Tensor
            Sound signal

        Returns
        ----------
        torch.Tensor
            Signal with background noise added
        """
        if self.noises is None:
            return waveform

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
    """
    Inverts the polarity of the signal
    """

    def __init__(self):
        super().__init__()

    def forward(self, waveform: torch.Tensor):
        """
        Parameters
        ----------
        waveform : torch.Tensor
            Sound signal

        Returns
        -------
        torch.Tensor
            Inverted signal
        """
        return waveform * -1


class AddWhiteNoise(torch.nn.Module):
    """
    Add gaussian/white noise to the signal
    """

    def __init__(self, min_factor, max_factor):
        """
        Parameters
        ----------
        min_factor : float
            Lower-bound for a noise factor
        max_factor : float
            Upper-bound for a noise factor
        """
        super().__init__()
        self.min = min_factor
        self.max = max_factor

    def forward(self, waveform: torch.Tensor):
        """
        Parameters
        ----------
        waveform: torch.Tensor
            Sound signal

        Returns
        ----------
        torch.Tensor
            Signal with white noise added
        """
        noise_factor = random.uniform(self.min, self.max)

        noise = torch.normal(0, waveform.std(), waveform.shape)
        noisy_signal = waveform + noise * noise_factor

        return noisy_signal


class PitchShifting(torch.nn.Module):
    """
    Changes the pitch of the signal
    """

    def __init__(self, sr, min_semitones, max_semitones):
        """
        Parameters
        ----------
        sr : float
            Sampling rate of the signal
        min_semitones : float
            Lower-bound for the number of semitones to shift
        max_semitones : float
            Upper-bound for the number of semitones to shift
        """
        super().__init__()
        self.min = min_semitones
        self.max = max_semitones
        self.sr = sr

    def forward(self, waveform: torch.Tensor):
        """
        Parameters
        ----------
        waveform: torch.Tensor
            Sound signal

        Returns
        ----------
        torch.Tensor
            Altered signal
        """
        n_semitones = random.randint(self.min, self.max)
        return pitch_shift(waveform=waveform, sample_rate=self.sr, n_steps=n_semitones)


class RandomGain(torch.nn.Module):
    """
    Scales the amplitude of the signal
    """

    def __init__(self, min_factor, max_factor):
        """
        Parameters
        ----------
        min_factor : float
            Lower-bound for scaling factor
        max_factor : float
            Upper-bound for scaling factor
        """
        super().__init__()
        self.min = min_factor
        self.max = max_factor

    def forward(self, waveform: torch.Tensor):
        """
        Parameters
        ----------
        waveform: torch.Tensor
            Sound signal

        Returns
        ----------
        torch.Tensor
            Scaled signal
        """
        gain_factor = random.uniform(self.min, self.max)

        return waveform * gain_factor


class TimeShift(torch.nn.Module):
    """
    Shifts the audio in time
    """

    def __init__(self, min_factor, max_factor):
        """
        Parameters
        ----------
        min_factor : float
            Lower-bound for shifting factor
        max_factor : float
            Upper-bound for shifting factor
        """
        super().__init__()
        self.min = min_factor
        self.max = max_factor

    def forward(self, waveform: torch.Tensor):
        """
        Parameters
        ----------
        waveform: torch.Tensor
            Sound signal

        Returns
        ----------
        torch.Tensor
            Shifted signal
        """
        shift_factor = random.uniform(self.min, self.max)

        max_shift = int(waveform.size()[1] * shift_factor)
        waveform = torch.roll(waveform, random.randint(-max_shift, max_shift))
        return waveform


# FILTERING
class BandPass(torch.nn.Module):
    """
    BandPass biquad filter
    """

    def __init__(self, sr, central_freq, Q=0.707):
        """
        Parameters
        ----------
        sr : float
            Sampling rate of the signal
        central_freq : float
            Central frequency of the signal
        Q: float
            Q-factor
        """
        super().__init__()
        self.sr = sr
        self.central_freq = central_freq
        self.Q = Q

    def forward(self, waveform: torch.Tensor):
        """
        Parameters
        ----------
        waveform: torch.Tensor
            Sound signal

        Returns
        ----------
        torch.Tensor
            Filtered signal
        """
        waveform = bandpass_biquad(waveform, self.sr, self.central_freq, self.Q)
        return waveform


# Currently NOT used
class RandomChunk(torch.nn.Module):
    """
    Obtains a random part of the signal
    """

    def __init__(self, sr, min_factor, max_factor):
        """
        Parameters
        ----------
        sr: float
            Sampling rate of the signal
        min_factor : float
            Lower-bound for chunk size
        max_factor : float
            Upper-bound for chunk size
        """
        super().__init__()
        self.sr = sr
        self.chunk_size = random.uniform(min_factor, max_factor)

    def forward(self, waveform: torch.Tensor):
        """
        Parameters
        ----------
        waveform: torch.Tensor
            Sound signal

        Returns
        ----------
        torch.Tensor
            Random part of the signal
        """
        seconds = int(self.chunk_size * waveform.size()[1] / self.sr)
        seconds *= self.sr
        start = random.randint(0, waveform.size()[1] - seconds)
        signal = waveform[:, start: start + seconds]
        print(waveform.shape)
        return signal
