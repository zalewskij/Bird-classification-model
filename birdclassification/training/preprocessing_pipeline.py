import torch
import random
from birdclassification.preprocessing.augmentations import InvertPolarity, AddWhiteNoise, PitchShifting, RandomGain, \
    TimeShift, AddBackgroundNoise, BandPass
from birdclassification.preprocessing.spectrogram import generate_mel_spectrogram
from birdclassification.preprocessing.utils import get_loudest_index, cut_around_index
import pickle
from birdclassification.preprocessing.utils import mix_down, right_pad
from birdclassification.preprocessing.utils import timer
from noisereduce.torchgate import TorchGate as TG


class PreprocessingPipeline(torch.nn.Module):
    """
        Pipeline for preprocessing the recordings
    """
    def __init__(self, noises_dir):
        """
        Parameters
        ----------
        noises_dir: string
            Path to a directory with environmental noises
        """
        super().__init__()

        self.parameters = {
            'white_noise_min': 0.1,
            'white_noise_max': 0.8,
            'random_gain_min': 0.5,
            'random_gain_max': 1.5,
            'time_shift_min': 0.1,
            'time_shift_max': 0.3,
            'add_background_min': 0.1,
            'add_background_max': 0.5,
            'pitch_shift_min': 1,
            'pitch_shift_max': 10,
            'n_fft': 512,
            'sr': 32000,
            'hop_length': 384,
            'sample_length': 3,
            'number_of_bands': 64,
            'fmin': 150,
            'fmax': 15000,
            'central_freq': 10000
        }

        # self.augmentations = [
        #     InvertPolarity(),
        #     AddWhiteNoise(min_factor=self.parameters['white_noise_min'], max_factor=self.parameters['white_noise_max']),
        #     RandomGain(min_factor=self.parameters['random_gain_min'], max_factor=self.parameters['random_gain_max']),
        #     TimeShift(min_factor=self.parameters['time_shift_min'], max_factor=self.parameters['time_shift_max']),
        #     AddBackgroundNoise(min_factor=self.parameters['add_background_min'], max_factor=self.parameters['add_background_max'], noises_dir=noises_dir),
        #     PitchShifting(sr=self.parameters['sr'], min_semitones=self.parameters['pitch_shift_min'], max_semitones=self.parameters['pitch_shift_max'])]

        self.augmentations = []
        self.probabilities = [0.5 for i in range(len(self.augmentations))]
        self.bandpass = BandPass(sr=self.parameters['sr'], central_freq=self.parameters['central_freq'])
        self.noise_reduction = TG(sr=self.parameters['sr'], nonstationary=False)

        self.get_spectrogram = generate_mel_spectrogram
        self.get_loudest_index = get_loudest_index
        self.cut_around_largest_index = cut_around_index
        self.mix_down = mix_down
        self.right_pad = right_pad

    def save(self, filepath):
        """
        Parameters
        ----------
        filepath
            Path to save the class instance
        """
        file = open(filepath, 'wb')
        pickle.dump(self, file)
        file.close()

    def load(self, filepath):
        """
        Parameters
        ----------
        filepath
            Path from where to load the class instance
        """
        file = open(filepath, 'rb')
        element = pickle.load(file)
        file.close()
        return element

    # @timer
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform: torch.Tensor
            Sound signal

        Returns
        -------
        audio: torch.Tensor
            Preprocessed audio in form of a spectrogram
        """
        waveform = self.mix_down(waveform)
        waveform = self.right_pad(waveform, minimal_length=self.parameters['sample_length'] * self.parameters['sr'])

        # select loudest 3 second chunk
        peak = get_loudest_index(waveform, self.parameters['n_fft'], self.parameters['hop_length'])
        waveform = cut_around_index(waveform, peak, self.parameters['sr'] * self.parameters['sample_length'])

        # augmentations
        if self.augmentations:
            n = random.randint(0, min(3, len(self.augmentations)))
            selected = random.choices(list(self.augmentations), weights=self.probabilities, k=n)
            aug = torch.nn.Sequential(*selected)
            waveform = aug(waveform)

        # generate spectrogram
        spectrogram = self.get_spectrogram(waveform, self.parameters['sr'], self.parameters['n_fft'],
                                           self.parameters['hop_length'])
        return spectrogram