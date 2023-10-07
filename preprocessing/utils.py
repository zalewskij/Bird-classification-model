from PIL import Image
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import librosa
from scipy.signal import hilbert
from visualization.plots import plot_selected_waveform


def cut_around_index(y, center, length):
    half_slice_width = int(length / 2)
    left_index = center - half_slice_width
    right_index = center + half_slice_width

    if left_index < 0:
        left_index = 0
        right_index = 2 * half_slice_width
    elif right_index >= len(y):
        right_index = len(y)
        left_index = right_index - 2 * half_slice_width

    return y[left_index:right_index]


def save_array_as_image(filename, array):
    img = Image.fromarray(np.uint8(array * 255), 'L')
    img.save(filename)


def get_loudest_index(y, n_fft, hop_length):
    # https://stackoverflow.com/questions/73208147/identifying-the-loudest-part-of-an-audio-track-and-cropping-librosa-or-torchaud
    clip_rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    clip_rms = clip_rms.squeeze()
    peak_rms_index = clip_rms.argmax()
    return peak_rms_index * hop_length + int(n_fft / 2)


def matched_filter(x, sr, filter):
    """
    https://en.wikipedia.org/wiki/Matched_filter
    Filter to detect the presence of the template in the unknown signal
    Parameters
    ----------
    x : np.array
        Unknown signal
    sr : int
        Sampling rate
    filter: np.array
        Template signal
    """
    h = np.flip(filter)
    a = 1
    filter_response = signal.lfilter(h, a, x)
    filter_delay = int((len(h) - 1) / 2)
    analytic_signal = hilbert(filter_response)
    envelope = np.abs(analytic_signal)

    b = np.ones(1000)
    b = b / len(b)
    envelope_1 = signal.filtfilt(b, a, envelope)

    threshold = 0.0175
    ind_stop = np.where((envelope_1[0:-2] >= threshold) & (envelope_1[1:-1] < threshold))
    env_shift = np.empty(len(envelope_1))
    env_shift[0] = np.NaN
    env_shift[1:] = envelope_1[0:-1]
    ind_start = np.where((envelope_1 >= threshold) & (env_shift < threshold))
    time_start = ind_start[0] - filter_delay
    time_stop = ind_stop[0] - filter_delay
    start_stop = np.sort(np.concatenate((time_start, time_stop)))

    plot_selected_waveform(x, start_stop)
