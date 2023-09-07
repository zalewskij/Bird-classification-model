from PIL import Image
import numpy as np
import librosa

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
    img = Image.fromarray(np.uint8(array * 255) , 'L')
    img.save(filename)

def get_loudest_index(y, n_fft, hop_length):
    # https://stackoverflow.com/questions/73208147/identifying-the-loudest-part-of-an-audio-track-and-cropping-librosa-or-torchaud
    clip_rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    clip_rms = clip_rms.squeeze()
    peak_rms_index = clip_rms.argmax()
    return peak_rms_index * hop_length + int(n_fft/2)
