import numpy as np
import librosa
import math
from utils import timer


@timer
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


@timer
def time_stretch(signal, stretch_rate):
    """
    Stretches the signal
    Parameters
    ----------
    signal : np.array
        Sound signal
    stretch_rate : float
        Bigger factor -> longer signal

    Returns
    -------
    np.array
        Stretched signal
    """
    return librosa.effects.time_stretch(signal, rate=stretch_rate)


@timer
def pitch_scale(signal, sr, n_semitones):
    """
    Changes the pitch of the signal
    Parameters
    ----------
    signal : np.array
        Sound signal
    sr : float
        Sampling rate of the signal
    n_semitones: float
        Number of semitones to shift

    Returns
    -------
    np.array
        Altered signal
    """
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_semitones)


@timer
def invert_polarity(signal):
    """
    Inverts the polarity of the signal
    Parameters
    ----------
    signal : np.array
        Sound signal

    Returns
    -------
    np.array
        Inverted signal
    """
    return signal * -1


@timer
def random_gain(signal, min_gain_factor, maximum_gain_factor):
    """
    Scales the amplitude of the signal
    Parameters
    ----------
    signal : np.array
        Sound signal
    min_gain_factor : float
        Lower-bound for the scaling
    maximum_gain_factor: float
        Upper-bound for the scaling

    Returns
    -------
    np.array
        Scaled signal
    """
    gain_factor = np.random.uniform(min_gain_factor, maximum_gain_factor)
    return signal * gain_factor


@timer
def add_background_noise(signal, noise, noise_factor):
    """
    Adds another audio to the signal
    Parameters
    ----------
    signal : np.array
        Sound signal
    noise : np.array
        Second sound signal - the one to be added
    noise_factor : float
        Bigger factor -> louder background signal

    Returns
    -------
    np.array
        Obtained signal
    """
    if len(noise) < len(signal):
        noise = np.tile(noise, (math.ceil(len(signal) / len(noise))))
    if len(noise) > len(signal):
        start = np.random.randint(len(noise) - len(signal))
        noise = noise[start:start + len(signal)]
    return signal + noise * noise_factor


@timer
def time_shift(signal):
    """
    Shifts the audio in time
    Parameters
    ----------
    signal : np.array
        Sound signal

    Returns
    -------
    np.array
        Shifted signal
    """
    max_shift = int(len(signal) * 0.1)
    signal = np.roll(signal, np.random.randint(-max_shift, max_shift))
    return signal


@timer
def random_chunk(signal, sr, chunk_size):
    """
    Obtains a random part of the signal
    Parameters
    ----------
    signal : np.array
        Sound signal
    sr : float
        Sampling rate of the signal
    chunk_size: int
        Number of seconds wanted

    Returns
    -------
    np.array
        Random part of the signal
    """
    chunk_size *= sr
    start = np.random.randint(0, len(signal) - chunk_size)
    signal = signal[start: start + chunk_size]
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
        length = np.random.randint(min_segment, max_segment)
        segment = signal[start:min(start + length, len(signal))]

        time_stretch_rate = np.random.normal(time_mean, time_sd)
        segment = librosa.effects.time_stretch(segment, rate=time_stretch_rate)

        n_semitones = np.random.normal(pitch_mean, pitch_sd)
        segment = librosa.effects.pitch_shift(segment, sr=sr, n_steps=n_semitones)

        start = min(start + length, len(signal))
        new_signal.extend(segment)

    return np.array(new_signal)
