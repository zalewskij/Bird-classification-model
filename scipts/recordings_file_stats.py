import pandas as pd
from birdclassification.preprocessing.filtering import filter_recordings_30
from birdclassification.training.dataset import Recordings30
from birdclassification.preprocessing.utils import get_thresholded_fragments, mix_down, right_pad
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
import torchaudio


TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 123
BASE_PATH = Path(__file__).resolve().parent.parent
#RECORDINGS_DIR = Path('/mnt/d/recordings_30')
RECORDINGS_DIR = Path("/media/jacek/E753-A120/recordings_30")
NOISES_DIR = Path('/media/jacek/E753-A120/NotBirds')
WRITER_DIR = Path(__file__).resolve().parent / "logs"
MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_models" / "cnn_1.pt"

SAMPLE_RATE = 32000
BATCH_SIZE = 32
NUM_WORKERS = 8

LEARNING_RATE = 0.0001
EPOCHS = 20

parameters = {
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
        }

df = filter_recordings_30(BASE_PATH / "data" / "xeno_canto_recordings.csv", BASE_PATH/ "data" / "bird-list-extended.csv")
noises_df = pd.read_csv(Path('../data/noises.csv'))

train_df, test_val_df = train_test_split(df, stratify=df['Latin name'], test_size=0.2, random_state = SEED)
val_df, test_df = train_test_split(test_val_df, stratify=test_val_df['Latin name'], test_size=0.5, random_state = SEED)

train_ds = Recordings30(train_df, recording_dir=RECORDINGS_DIR, device = DEVICE, random_fragment=True)
val_ds = Recordings30(val_df, recording_dir=RECORDINGS_DIR, device = DEVICE)
test_ds = Recordings30(test_df, recording_dir=RECORDINGS_DIR, device = DEVICE)

train_dl  = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
val_dl  = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


df = pd.DataFrame(columns = ['id', 'Latin name', 'length', 'fragments_0.7'])
for i in range(10):
    path = str(train_ds.get_filepath(i)).split('/')
    id = path[-1].split('.')[0]
    latin_name = path[-2]
    waveform, label = torchaudio.load(train_ds.get_filepath(i))
    length = waveform.size()[1]/SAMPLE_RATE
    waveform = mix_down(waveform)
    waveform = right_pad(waveform, minimal_length=parameters['sample_length'] * parameters['sr'])
    n_fragments = len(get_thresholded_fragments(waveform, SAMPLE_RATE, parameters['n_fft'], parameters['hop_length'], parameters['sample_length'], threshold=0.7))
    new_row = {'id': id, 'Latin name': latin_name, 'length': length, 'fragments_0.7': int(n_fragments)}
    df = df._append(new_row, ignore_index=True)

print(df)

