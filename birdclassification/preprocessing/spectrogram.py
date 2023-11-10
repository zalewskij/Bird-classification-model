import librosa
import numpy as np
import torchaudio

from .utils import *


def generate_mel_spectrogram(y, sr, n_fft, hop_length, number_of_bands = 64, fmin = 150, fmax = 15000):
    """
    Generates mel spectrogram for a given signal
    Parameters
    ----------
    y : Tensor
        Signal for generating spectrogram
    sr : int
        sampling rate
    n_fft : int
        frames per window
    hop_length : int
        overlap of windows
    number_of_bands : int
        number of mel bands
    fmin : int
        minimal frequency for spectrogram
    fmax : int
        maximal frequency for spectrogram

    Returns
    -------
    Tensor
        Spectrogram represented as a 2d array
    """
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, f_min=fmin, f_max=fmax, n_mels=number_of_bands)
    M = transform(y)[0]
    M_db = librosa.power_to_db(M, ref=np.max)
    normalized_M_db = (M_db + 80) / 80
    return normalized_M_db


def export_and_save_spectrogram(input_filename, output_filename):
    """
    Generates and saves as an image mel spectrogram created from the loudest three seconds of a given recoding
    Parameters
    ----------
    input_filename : string
        Name of the input file (recording)
    output_filename : string
        Name of the output file (image)
    """
    sr = 32000 # sampling rate
    n_fft = 512 # frames per window
    hop_length = 3 * 128 # overlap
    sample_length = 3 # length of sample for spectrogram

    y, rec_sr = torchaudio.load(input_filename)
    if rec_sr != sr:
        y = torchaudio.transforms.Resample(orig_freq=rec_sr, new_freq=sr)(y)

    peak = get_loudest_index(y, n_fft, hop_length)
    y = cut_around_index(y, peak, sr * sample_length)
    spectrogram = generate_mel_spectrogram(y, sr, n_fft, hop_length)
    img = array_to_image(spectrogram)
    img.save(output_filename)


def export_spectrograms_with_threshold(input_filename, threshold = 0.7):
    """
    Generates mel spectrograms from a given recording, taking into account fragments louder than a given threshold 
    Parameters
    ----------
    input_filename : string
        Name of the input file (recording)

    Returns
    -------
    list[Image]
        Spectrogram represented as an image
    """
    sr = 32000 # sampling rate
    n_fft = 512 # frames per window
    hop_length = 3 * 128 # overlap
    sample_length = 3 # length of sample for spectrogram

    y, rec_sr = torchaudio.load(input_filename)
    if rec_sr != sr:
        y = torchaudio.transforms.Resample(orig_freq=rec_sr, new_freq=sr)(y)

    samples = get_thresholded_fragments(y, sr, n_fft, hop_length, sample_length, threshold)
    spectrograms = [generate_mel_spectrogram(sample, sr, n_fft, hop_length) for sample in samples]
    return [array_to_image(x) for x in spectrograms]
