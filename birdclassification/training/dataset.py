from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from birdclassification.training.dataset_pipeline import DatasetPipeline
import torchaudio
from birdclassification.preprocessing.utils import timer
from birdclassification.visualization.plots import plot_torch_spectrogram, plot_torch_waveform
from pathlib import Path


class Recordings30(Dataset):
    def __init__(self, df, recording_dir, device="cpu", random_fragment=False):
        """
        Parameters
        ----------
        df: pd.DataFrame
            dataframe of recordings
        recording_dir: str
            filepath to the directory with recordings
        noises_df: pd.DataFrame
            dataframe of noises
        noises_dir: str
            filepath to the directory with noises
        device: str
            cpu or cuda depending on the used device
        """
        df['filepath'] = df.apply(lambda x: Path(recording_dir, x['Latin name'], f"{str(x['id'])}.mp3"), axis=1)
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['Latin name'])

        self.filepath = df['filepath'].to_numpy()
        self.label = df['label'].to_numpy()
        self.device = device
        self.recording_dir = recording_dir
        self.dataset_pipeline = DatasetPipeline(random_fragment=random_fragment).to(device)
        self.le_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))

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
        audio = self.dataset_pipeline(audio)
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

    def get_mapping(self):
        """
        Returns
        -------
        Dictionary mapping number to Latin name
        """
        return self.le_name_mapping

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
            print("Label: ", self.get_mapping()[label])
            plot_torch_waveform(audio, sr)
            plot_torch_spectrogram(audio)
