import torch
import random
from birdclassification.preprocessing.augmentations import InvertPolarity, AddWhiteNoise, PitchShifting, RandomGain, \
    TimeShift, AddBackgroundNoise, BandPass
from birdclassification.preprocessing.spectrogram import generate_mel_spectrogram, generate_mel_spectrogram_seq
from birdclassification.preprocessing.utils import get_loudest_index, cut_around_index, get_thresholded_fragments
from birdclassification.preprocessing.utils import mix_down, right_pad
from noisereduce.torchgate import TorchGate as TG


class PreprocessingPipeline(torch.nn.Module):
    """
        Pipeline for preprocessing the recordings
    """
    def __init__(self, noises_df=None, noises_dir='', random_fragment=False, device='cpu'):
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
            'central_freq': 10000,
            'random_fragment': random_fragment 
        }

        # self.augmentations = [
        #     InvertPolarity().to(device),
        #     AddWhiteNoise(min_factor=self.parameters['white_noise_min'], max_factor=self.parameters['white_noise_max'], device=device).to(device),
        #     RandomGain(min_factor=self.parameters['random_gain_min'], max_factor=self.parameters['random_gain_max']).to(device),
        #     TimeShift(min_factor=self.parameters['time_shift_min'], max_factor=self.parameters['time_shift_max']).to(device),
        #     AddBackgroundNoise(min_factor=self.parameters['add_background_min'], max_factor=self.parameters['add_background_max'], df=noises_df, noises_dir=noises_dir).to(device),
        #     # PitchShifting(sr=self.parameters['sr'], hop_length=self.parameters['hop_length'], n_fft=self.parameters['n_fft'], min_semitones=self.parameters['pitch_shift_min'], max_semitones=self.parameters['pitch_shift_max'])
        # ]

        self.device = device

        self.augmentations = []
        self.probabilities = [0.5 for i in range(len(self.augmentations))]
        self.bandpass = BandPass(sr=self.parameters['sr'], central_freq=self.parameters['central_freq'])
        self.noise_reduction = TG(sr=self.parameters['sr'], nonstationary=False)

        self.get_spectrogram = generate_mel_spectrogram
        self.get_thresholded_fragments = get_thresholded_fragments
        self.get_loudest_index = get_loudest_index
        self.cut_around_largest_index = cut_around_index
        self.mix_down = mix_down
        self.right_pad = right_pad

    def forward(self, waveform: torch.Tensor, use_augmentations = True) -> torch.Tensor:
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

        if self.augmentations and use_augmentations:
            n = random.randint(0, min(3, len(self.augmentations)))
            selected = random.choices(list(self.augmentations), weights=self.probabilities, k=n)
            aug = torch.nn.Sequential(*selected)
            waveform = aug(waveform)

        spectrogram = generate_mel_spectrogram_seq(y=waveform, sr=32000, n_fft=512, hop_length=384, device=self.device)
        spectrogram = torch.unsqueeze(spectrogram, dim=1)

        return spectrogram