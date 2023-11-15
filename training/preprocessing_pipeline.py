import torch
import random
from birdclassification.preprocessing.augmentations import InvertPolarity, AddWhiteNoise, PitchShifting, RandomGain, \
    TimeShift, AddBackgroundNoise
from birdclassification.preprocessing.spectrogram import generate_mel_spectrogram
from birdclassification.preprocessing.utils import get_loudest_index, cut_around_index
import pickle
from birdclassification.preprocessing.utils import mix_down


class PreprocessingPipeline(torch.nn.Module):
    def __init__(self, noises_dir):
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
            'fmax': 15000
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

        self.get_spectrogram = generate_mel_spectrogram
        self.get_loudest_index = get_loudest_index
        self.cut_around_largest_index = cut_around_index
        self.mix_down = mix_down

    def save(self, filepath):
        file = open(filepath, 'wb')
        pickle.dump(self, file)
        file.close()

    def load(self, filepath):
        file = open(filepath, 'rb')
        element = pickle.load(file)
        file.close()
        return element

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = self.mix_down(waveform)

        # select loudest 3 second chunk
        peak = get_loudest_index(waveform, self.parameters['n_fft'], self.parameters['hop_length'])
        waveform = cut_around_index(waveform, peak, self.parameters['sr'] * self.parameters['sample_length'])

        # augmentations
        if self.augmentations:
            n = random.randint(0, len(self.augmentations))
            selected = random.choices(list(self.augmentations), weights=self.probabilities, k=n)
            print(selected)
            aug = torch.nn.Sequential(*selected)
            waveform = aug(waveform)

        # generate spectrogram
        spectrogram = self.get_spectrogram(waveform, self.parameters['sr'], self.parameters['n_fft'], self.parameters['hop_length'])

        return spectrogram
