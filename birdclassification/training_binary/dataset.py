from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from birdclassification.training.preprocessing_pipeline import PreprocessingPipeline
import torchaudio
from birdclassification.preprocessing.utils import timer
from birdclassification.visualization.plots import plot_torch_spectrogram, plot_torch_waveform
from pathlib import Path


class BinaryDataset(Dataset):
    def __init__(self, df, not_bird_dir, bird_dir, noises_dir, sample_rate=32000, device="cpu"):
        df['filepath'] = df.apply(lambda x: Path(bird_dir, x['folder'], f"{str(x['filename'])}.mp3")
                                  if x['isBird'] == 1 else Path(not_bird_dir, x['folder'], f"{str(x['filename'])}.ogg"),
                                  axis=1)
        self.sample_rate = sample_rate
        self.filepath = df['filepath'].to_numpy()
        self.label = df['isBird'].to_numpy()
        self.device = device
        self.preprocessing_pipeline = PreprocessingPipeline(noises_df=None, noises_dir=noises_dir).to(device)

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
        label:
            label of the recording
        """
        audio, sr = torchaudio.load(self.filepath[idx])
        label = self.label[idx]
        audio = self.preprocessing_pipeline(audio)
        return audio, label

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
            label = self.label[i]
            print("Shape of audio:", audio.shape)
            if label == 0:
                print("Label: Not Bird")
            else:
                print("Label: Bird")

            plot_torch_waveform(audio, sr)
            plot_torch_spectrogram(audio)
            print("-------------------------------------------------------------")
