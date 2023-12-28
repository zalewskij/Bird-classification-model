import librosa
import numpy as np
import torchaudio

from .utils import *


class ShiftSpectrogram(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    x = torch.squeeze(x, 1)
    maximum = torch.amax(x, dim=(1,2))
    return (x - maximum[:, None, None])


def generate_mel_spectrogram_seq(y, sr, n_fft, hop_length, number_of_bands = 64, fmin = 150, fmax = 15000, device = 'cpu'):
    T2 = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, f_min=fmin, f_max=fmax, n_mels=number_of_bands),
        torchaudio.transforms.AmplitudeToDB(top_db=80),
        ShiftSpectrogram()
    )
    T2.to(device)
    return T2(y)


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
    return M_db


def normalize_spectrogram_for_display(spectrogram):
    """
    Normalized spectrogram for displaying it a black and white image
    Parameters
    ----------
    Tensor
        Spectrogram represented as a 2d array

    Returns
    -------
    Tensor
        Spectrogram represented as a 2d array
    """
    return (spectrogram + 80) / 80


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
