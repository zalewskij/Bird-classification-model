from birdclassification.training.preprocessing_pipeline import PreprocessingPipeline
import torch
from torch import nn
from torchvision.models import resnet34
from birdclassification.preprocessing.filtering import filter_recordings_30
from birdclassification.preprocessing.spectrogram import generate_mel_spectrogram_seq
from birdclassification.training.dataset import Recordings30
from birdclassification.training.training_utils import train_one_epoch
from birdclassification.training.validation_metrics import calculate_metric
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from pathlib import Path
from datetime import datetime
import sys
from time import time


TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 123
BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent
RECORDINGS_DIR = Path('/mnt/d/recordings_30')
NOISES_DIR = Path('') # Path('D:\\JAcek\\noises_dir')
WRITER_DIR = Path(__file__).resolve().parent / "logs"
MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_models" / "resnet_1.pt"

SAMPLE_RATE = 32000
BATCH_SIZE = 32
NUM_WORKERS = 8

LEARNING_RATE = 0.0001
EPOCHS = 5

df = filter_recordings_30(BASE_PATH / "data" / "xeno_canto_recordings.csv",
                          BASE_PATH / "data" / "bird-list-extended.csv")
noises_df = None

train_df, test_val_df = train_test_split(df, stratify=df['Latin name'], test_size=0.2, random_state=SEED)
val_df, test_df = train_test_split(test_val_df, stratify=test_val_df['Latin name'], test_size=0.5,
                                   random_state=SEED)

train_ds = Recordings30(train_df, recording_dir=RECORDINGS_DIR, device=DEVICE)
val_ds = Recordings30(val_df, recording_dir=RECORDINGS_DIR, device=DEVICE)
test_ds = Recordings30(test_df, recording_dir=RECORDINGS_DIR, device=DEVICE)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


resnet = resnet34(pretrained=True)
resnet.fc = nn.Linear(512, 50)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet = resnet.to(DEVICE)

preprocessing_pipeline = PreprocessingPipeline(device=DEVICE, random_fragment=True, noises_dir=NOISES_DIR, noises_df=noises_df).to(DEVICE)

optimizer = torch.optim.Adam(resnet.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

best_vloss = sys.float_info.max

print("--------------------")
print(f"TIMESTAMP: {TIMESTAMP}")
print(f"DEVICE: {DEVICE}")
print(f"SEED: {SEED}")
print(f"BASE_PATH: {BASE_PATH}")
print(f"RECORDINGS_DIR: {RECORDINGS_DIR}")
print(f"NOISES_DIR: {NOISES_DIR}")
print(f"WRITER_DIR: {WRITER_DIR}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"SAMPLE_RATE: {SAMPLE_RATE}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"NUM_WORKERS: {NUM_WORKERS}")
print(f"LEARNING_RATE: {LEARNING_RATE}")
print(f"EPOCHS: {EPOCHS}")
print("--------------------")

writer = SummaryWriter(WRITER_DIR)
epoch_number = 0
start_time = time()

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    epoch_start_time = time()

    # Make sure gradient tracking is on, and do a pass over the data
    resnet.train()
    avg_loss = train_one_epoch(epoch_number, preprocessing_pipeline, writer, train_dl, optimizer, loss_fn, resnet, DEVICE, start_time)

    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    resnet.eval()
    running_vloss = 0.0

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_dl):
            vinputs, vlabels = vdata
            vinputs = preprocessing_pipeline(vinputs.to(DEVICE), use_augmentations=False)
            voutputs = resnet(vinputs.to(DEVICE, dtype=torch.float32))
            vloss = loss_fn(voutputs, vlabels.long().to(DEVICE))
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print("#############################################################")
    print("Epoch results:")
    print(f'Loss train {avg_loss} valid loss: {avg_vloss}')
    validation_precision_score = calculate_metric(resnet, val_dl, device=DEVICE, preprocessing_pipeline=preprocessing_pipeline, metric=lambda x, y: precision_score(x, y, average='macro'))
    print(f'Validation macro avarage precision: {validation_precision_score}')
    print(f'Epoch execution time {time() - epoch_start_time}')
    print("#############################################################\n\n")

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       epoch_number + 1)

    writer.add_scalars('Macro_averaged_precision_score',
                       {'Validation': validation_precision_score},
                       epoch_number + 1)

    writer.flush()

    # Track best performance, and save the model's state
    best_vloss = avg_vloss
    model_path = f'model_{TIMESTAMP}_{epoch_number}'
    torch.save(resnet.state_dict(), MODEL_PATH.parent / model_path)

    epoch_number += 1

torch.save(resnet.state_dict(), MODEL_PATH)
