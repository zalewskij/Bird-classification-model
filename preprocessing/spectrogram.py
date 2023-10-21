import librosa
import numpy as np

from utils import *

def generate_mel_spectrogram(y, sr, n_fft, hop_length, number_of_bands = 64, fmin = 150, fmax = 15000):
    """
    Generates mel spectrogram for a given signal
    Parameters
    ----------
    y : np.array
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
    np.array
        Spectrogram represented as a 2d array
    """
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=number_of_bands, fmin=fmin, fmax=fmax)
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
    sr = 48000 # sampling rate
    n_fft = 512 # frames per window
    hop_length = 3 * 128 # overlap
    sample_length = 3 # length of sample for spectrogram

    y, _ = librosa.load(input_filename, sr=sr)
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
    sr = 48000 # sampling rate
    n_fft = 512 # frames per window
    hop_length = 3 * 128 # overlap
    sample_length = 3 # length of sample for spectrogram

    y, _ = librosa.load(input_filename, sr=sr)
    samples = get_thresholded_fragments(y, sr, n_fft, hop_length, sample_length, threshold)
    spectrograms = [generate_mel_spectrogram(sample, sr, n_fft, hop_length) for sample in samples]
    return [array_to_image(x) for x in spectrograms]