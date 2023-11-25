from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from training.preprocessing_pipeline import PreprocessingPipeline
import torchaudio
from birdclassification.preprocessing.utils import timer
from birdclassification.visualization.plots import plot_torch_spectrogram, plot_torch_waveform


class Recordings30(Dataset):
    def __init__(self, df, recording_dir, noises_dir, sample_rate=32000, device="cpu"):
        df['filepath'] = df.apply(lambda x: f"{recording_dir}{x['Latin name']}/{str(x['id'])}.mp3", axis=1)
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['Latin name'])

        self.sample_rate = sample_rate
        self.filepath = df['filepath'].to_numpy()
        self.label = df['label'].to_numpy()
        self.device = device
        self.recording_dir = recording_dir
        self.preprocessing_pipeline = PreprocessingPipeline(noises_dir).to(device)
        self.le_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))

    def __len__(self):
        return self.filepath.size

    def __getitem__(self, idx):
        #print(self.get_filepath(idx))
        audio, sr = torchaudio.load(self.filepath[idx])
        # audio = audio.to(self.device)
        label = self.label[idx]
        #label = label.to(self.device)
        audio = self.preprocessing_pipeline(audio)
        return audio, label

    def get_filepath(self, idx):
        return self.filepath[idx]

    def get_mapping(self):
        return self.le_name_mapping

    def visualize_dataset(self, idx, n):
        """
        Parameters
        ----------
        idx: index of the first recording
        n: number of samples to plot
        """
        for i in range(idx, idx + n):
            audio, sr = torchaudio.load(self.filepath[i])
            label = self.label[i]
            print("Shape of audio:", audio.shape)
            print("Label: ", self.get_mapping()[label])
            plot_torch_waveform(audio, sr)
            plot_torch_spectrogram(audio)


