import numpy as np
from scipy import signal


def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_signal = signal.sosfiltfilt(sos, data)
    return filtered_signal


def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data)
    return filtered_data


def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data)
    return filtered_data
