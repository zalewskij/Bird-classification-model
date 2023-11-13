import random
import librosa
import math
import numpy as np
import torch

from birdclassification.preprocessing.utils import timer


@timer
def add_white_noise(signal, noise_factor):
    """
    Add gaussian/white noise to the signal
    Parameters
    ----------
    signal : torch.Tensor
        Sound signal
    noise_factor : float
        Bigger factor -> noisier signal

    Returns
    -------
    torch.Tensor
        Noisy signal
    """
    noise = torch.normal(0, signal.std(), signal.shape)
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal)
    noisy_signal = signal + noise * noise_factor
    return noisy_signal


@timer
def time_stretch(signal, stretch_rate):
    """
    Stretches the signal
    Parameters
    ----------
    signal : torch.Tensor
        Sound signal
    stretch_rate : float
        Bigger factor -> longer signal

    Returns
    -------
    torch.Tensor
        Stretched signal
    """
    return librosa.effects.time_stretch(signal, rate=stretch_rate)


@timer
def pitch_shift(signal, sr, n_semitones):
    """
    Changes the pitch of the signal
    Parameters
    ----------
    signal : torch.Tensor
        Sound signal
    sr : float
        Sampling rate of the signal
    n_semitones: float
        Number of semitones to shift

    Returns
    -------
    torch.Tensor
        Altered signal
    """
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_semitones)


@timer
def invert_polarity(signal):
    """
    Inverts the polarity of the signal
    Parameters
    ----------
    signal : torch.Tensor
        Sound signal

    Returns
    -------
    torch.Tensor
        Inverted signal
    """
    return signal * -1


@timer
def random_gain(signal, gain_factor):
    """
    Scales the amplitude of the signal
    Parameters
    ----------
    signal : torch.Tensor
        Sound signal
    gain_factor : float
         Scaling factor

    Returns
    -------
    torch.Tensor
        Scaled signal
    """
    return signal * gain_factor


@timer
def add_background_noise(signal, noise, noise_factor):
    """
    Adds another audio to the signal
    Parameters
    ----------
    signal : torch.Tensor
        Sound signal
    noise : np.array
        Second sound signal - the one to be added
    noise_factor : float
        Bigger factor -> louder background signal

    Returns
    -------
    torch.Tensor
        Obtained signal
    """
    if len(noise) < len(signal):
        noise = np.tile(noise, (math.ceil(len(signal) / len(noise))))
    if len(noise) > len(signal):
        start = np.random.randint(len(noise) - len(signal))
        noise = noise[start:start + len(signal)]
    return signal + noise * noise_factor


@timer
def time_shift(signal, shift_factor):
    """
    Shifts the audio in time
    Parameters
    ----------
    signal : torch.Tensor
        Sound signal

    Returns
    -------
    torch.Tensor
        Shifted signal
    """
    max_shift = int(len(signal) * shift_factor)
    signal = torch.roll(signal, random.randint(-max_shift, max_shift))
    return signal


@timer
def random_chunk(signal, sr, chunk_size):
    """
    Obtains a random part of the signal
    Parameters
    ----------
    signal : torch.Tensor
        Sound signal
    sr : float
        Sampling rate of the signal
    chunk_size: float
        Proportion of the audio wanted

    Returns
    -------
    torch.Tensor
        Random part of the signal
    """
    seconds = int(chunk_size * signal.size()[1]/sr)
    seconds *= sr
    start = random.randint(0, signal.size()[1] - seconds)
    signal = signal[:, start: start + seconds]
    return signal


@timer
def partial_time_and_pitch_stretching(signal, sr, min_duration, max_duration, time_sd, pitch_sd, time_mean, pitch_mean):
    """
    Random partial stretching in time and frequency
    Parameters
    ----------
    signal : np.array
        Sound signal
    sr : float
        Sampling rate of the signal
    min_duration: float
        Minimal length of a segment
    max_duration: float
        Maximal length of a segment
    time_sd: float
        Standard deviation for choosing the rate of a time stretch
    pitch_sd: float
        Standard deviation for choosing the number of semitones
    time_mean: float
        Mean for choosing the rate of a time stretch
    pitch_mean: float
        Mean for choosing the number of semitones

    Returns
    -------
    np.array
        Altered signal
    """
    start = 0
    new_signal = []
    min_segment = int(min_duration * sr)
    max_segment = int(max_duration * sr)
    while start < len(signal):
        length = random.randint(min_segment, max_segment)
        segment = signal[start:min(start + length, len(signal))]

        time_stretch_rate = np.random.normal(time_mean, time_sd)
        segment = librosa.effects.time_stretch(segment, rate=time_stretch_rate)

        n_semitones = np.random.normal(pitch_mean, pitch_sd)
        segment = librosa.effects.pitch_shift(segment, sr=sr, n_steps=n_semitones)

        start = min(start + length, len(signal))
        new_signal.extend(segment)

    return new_signal
