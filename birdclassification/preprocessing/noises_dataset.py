from torch.utils.data import Dataset
import torchaudio
from birdclassification.visualization.plots import plot_torch_spectrogram, plot_torch_waveform
from pathlib import Path


class NoisesDataset(Dataset):
    def __init__(self, df, recordings__dir, sample_rate=32000):
        df['filepath'] = df.apply(lambda x: Path(recordings__dir, x['folder'], f"{str(x['filename'])}.ogg"), axis=1)
        self.sample_rate = sample_rate
        self.filepath = df['filepath'].to_numpy()

    def __len__(self):
        """
        Returns
        -------
        Size of the dataset: int
        """
        return self.filepath.size

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int
            Index of the recording

        Returns
        -------
        audio: torch.Tensor
            recording waveform
        """
        audio, sr = torchaudio.load(self.filepath[idx])
        return audio

    def get_filepath(self, idx):
        """
        Parameters
        ----------
        idx: int
            Index of the recording

        Returns
        -------
        Filepath of the recording: str
        """
        return self.filepath[idx]

    def visualize_dataset(self, idx, n):
        """
        Function to plot waveform, spectrogram and play recording

        Parameters
        ----------
        idx: index of the first recording
        n: number of samples to plot

        Returns
        -------
        None
        """
        for i in range(idx, idx + n):
            audio, sr = torchaudio.load(self.filepath[i])
            print("Shape of audio:", audio.shape)
            plot_torch_waveform(audio, sr)
            plot_torch_spectrogram(audio)
            print("-------------------------------------------------------------")
