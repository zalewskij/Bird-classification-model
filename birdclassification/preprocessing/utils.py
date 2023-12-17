import torch
from PIL import Image
import numpy as np
from scipy import signal
import librosa
from scipy.signal import hilbert
from birdclassification.visualization.plots import plot_selected_waveform
from time import time


def timer(func):
    """
    Function to measure time of execution of the function
    Parameters
    ----------
    func
        Function to be wrapped

    Returns
    -------
        Function func with added functionality of printing its execution time
    """
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f'Function: {func.__name__!r}, execution time: {end - start}')
        return result

    return wrapper


def cut_around_index_librosa(y, center, length):
    """
    Return a fragment of the signal centered at given index with given length
    Parameters
    ----------
    y : np.array
        Signal represented as an array
    center : int
        index in the array which will be the center of the fragment
    length : int
        length of the fragment

    Returns
    -------
    np.array
        Fragment of the signal represented as an array
    """
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


def cut_around_index(y, center, length):
    """
    Return a fragment of the signal centered at given index with given length
    Parameters
    ----------
    y : Tensor
        Signal represented as a tensor
    center : int
        index in the array which will be the center of the fragment
    length : int
        length of the fragment

    Returns
    -------
    Tensor
        Fragment of the signal represented as a tensor
    """
    half_slice_width = int(length / 2)
    left_index = center - half_slice_width
    right_index = center + half_slice_width

    if left_index < 0:
        left_index = 0
        right_index = 2 * half_slice_width
    elif right_index >= y.size()[1]:
        right_index = y.size()[1]
        left_index = right_index - 2 * half_slice_width

    return y[:, left_index:right_index]


def array_to_image(array):
    return Image.fromarray(np.uint8(array * 255), 'L')


def convert_waveform_to_loudness(y, n_fft, hop_length):
    """
    https://stackoverflow.com/questions/73208147/identifying-the-loudest-part-of-an-audio-track-and-cropping-librosa-or-torchaud
    Converts a signal (sound) to the loudness representation 
    Parameters
    ----------
    y : np.array
        Signal represented as an array
    n_fft : int
        frames per window
    hop_length : int
        overlap of windows

    Returns
    -------
    np.array
        Loundness of the signal
    """
    clip_rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    return clip_rms.squeeze()


def get_loudest_index(y, n_fft, hop_length):
    """
    Finds the loudest part of a signal
    Parameters
    ----------
    y : np.array
        Signal represented as an array
    n_fft : int
        frames per window
    hop_length : int
        overlap of windows

    Returns
    -------
    int
        Index of the loudest part
    """
    clip_rms = convert_waveform_to_loudness(y, n_fft, hop_length)
    return clip_rms.argmax() * hop_length + int(n_fft / 2)


def get_thresholded_fragments_librosa(y, sr, n_fft, hop_length, sample_length, threshold):
    """
    Selects the fragments of a given length from the signal, based on their loudness
    Parameters
    ----------
    y : np.array
        Signal represented as an array
    sr : int
        sampling rate
    n_fft : int
        frames per window
    hop_length : int
        overlap of windows
    sample_length: int
        length of the created samples

    Returns
    -------
    list[np.array]
        List of fragments of the given signal
    """
    clip_rms = convert_waveform_to_loudness(y, n_fft, hop_length)
    peak_rms_index = clip_rms.argmax()

    sample_length_for_rms = int(sample_length * sr / hop_length)

    best_sample_start_rms = peak_rms_index - int(sample_length_for_rms / 2)
    start = best_sample_start_rms - int(best_sample_start_rms / sample_length_for_rms) * sample_length_for_rms
    number_of_potential_samples = int((len(clip_rms) - start) / sample_length_for_rms)

    potential_samples = [x * sample_length_for_rms + start for x in range(number_of_potential_samples)]

    loudness_threshold = threshold * clip_rms[peak_rms_index]
    samples = [x for x in potential_samples if clip_rms[x:x + sample_length_for_rms].max() > loudness_threshold]

    chosen_indexes = [i * hop_length + int(n_fft / 2) for i in samples]
    return [y[left_index:left_index + (sample_length * sr)] for left_index in chosen_indexes]


def get_thresholded_fragments(y, sr, n_fft, hop_length, sample_length, threshold = 0.7):
    """
    Selects the fragments of a given length from the signal, based on their loudness
    Parameters
    ----------
    y : Tensor
        Signal represented as a tensor
    sr : int
        sampling rate
    n_fft : int
        frames per window
    hop_length : int
        overlap of windows
    sample_length: int
        length of the created samples

    Returns
    -------
    list[Tensor]
        List of fragments of the given signal
    """
    clip_rms = convert_waveform_to_loudness(y, n_fft, hop_length)
    peak_rms_index = clip_rms.argmax()

    sample_length_for_rms = int(sample_length * sr / hop_length)

    best_sample_start_rms = peak_rms_index - int(sample_length_for_rms / 2)
    start = best_sample_start_rms - int(best_sample_start_rms / sample_length_for_rms) * sample_length_for_rms
    number_of_potential_samples = int((len(clip_rms) - start) / sample_length_for_rms)

    potential_samples = [x * sample_length_for_rms + start for x in range(number_of_potential_samples)]

    loudness_threshold = threshold * clip_rms[peak_rms_index]
    samples = [x for x in potential_samples if clip_rms[x:x + sample_length_for_rms].max() > loudness_threshold]

    chosen_indexes = [i * hop_length + int(n_fft / 2) for i in samples]
    return [y[:, left_index:left_index + (sample_length * sr)] for left_index in chosen_indexes]


def matched_filter(x, sr, filter):
    """
    https://en.wikipedia.org/wiki/Matched_filter
    Filter to detect the presence of the template in the unknown signal
    Parameters
    ----------
    x : np.array
        Unknown signal
    sr : int
        sampling rate
    filter: np.array
        Template signal
    Returns
    -------
    None
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

    plot_selected_waveform(x, sr, start_stop)


def mix_down(waveform):
    """
    Convert stereo to mono using a mean of two signals.
    https://github.com/pytorch/audio/issues/363
    Parameters
    ----------
    waveform: torch.Tensor

    Returns
    -------
    waveform: torch.Tensor
        Mono waveform
    """
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def right_pad(waveform, minimal_length):
    """
    Right padding of signal if the signal is shorter the desired length
    Parameters
    ----------
    waveform: torch.Tensor
    minimal_length: int
    Returns
    -------
    waveform: torch.Tensor
        Padded waveform
    """
    length_signal = waveform.shape[1]
    if length_signal < minimal_length:
        missing_samples = minimal_length - length_signal
        last_dim_padding = (0, missing_samples)
        waveform = torch.nn.functional.pad(waveform, last_dim_padding)
    return waveform
