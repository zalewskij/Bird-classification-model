import torch
import random
from birdclassification.preprocessing.utils import get_loudest_index, cut_around_index, get_thresholded_fragments
from birdclassification.preprocessing.utils import mix_down, right_pad


class DatasetPipeline(torch.nn.Module):
    """
        Pipeline for preprocessing the recordings while creating the dataset
    """
    def __init__(self, random_fragment=False):
        """
        Parameters
        ----------
        random_fragment: bool
            Is the fragment of the recoding choosen randomly
        """
        super().__init__()

        self.parameters = {
            'n_fft': 512,
            'sr': 32000,
            'hop_length': 384,
            'sample_length': 3,
            'random_fragment': random_fragment 
        }

        self.get_thresholded_fragments = get_thresholded_fragments
        self.get_loudest_index = get_loudest_index
        self.cut_around_largest_index = cut_around_index
        self.mix_down = mix_down
        self.right_pad = right_pad

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform: torch.Tensor
            Sound signal

        Returns
        -------
        audio: torch.Tensor
            Preprocessed audio in form of a waveform
        """
        waveform = self.mix_down(waveform)
        waveform = self.right_pad(waveform, minimal_length=self.parameters['sample_length'] * self.parameters['sr'])

        if self.parameters['random_fragment']:
            #select random 3 second chunk from the loudest ones
            waveform = random.choice(self.get_thresholded_fragments(waveform, self.parameters['sr'], self.parameters['n_fft'],
                                            self.parameters['hop_length'], sample_length=self.parameters['sample_length'], threshold=0.7))
        else:
            # select loudest 3 second chunk
            peak = self.get_loudest_index(waveform, self.parameters['n_fft'], self.parameters['hop_length'])
            waveform = self.cut_around_largest_index(waveform, peak, self.parameters['sr'] * self.parameters['sample_length'])

        return waveform