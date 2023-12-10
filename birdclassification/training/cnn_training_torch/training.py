import pandas as pd
from birdclassification.training.dataset import BinaryDataset
from birdclassification.training.cnn_training_torch.CNN_model import CNNNetwork
from birdclassification.training.training_utils import train_one_epoch
from birdclassification.training.validation_metrics import calculate_metric
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import precision_score
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
from time import time

def main():
    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    SEED = 123
    BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent
    BIRD_RECORDINGS_DIR = Path('/mnt/d/JAcek/recordings_30')
    NOT_BIRD_RECORDINGS_DIR = Path('/mnt/d/JAcek/NotBirds')
    NOISES_DIR = Path('')
    WRITER_DIR = Path('logs')
    MODEL_PATH = Path(__file__).resolve().parent / "saved_models" / "binary_classificator.pt"

    SAMPLE_RATE = 32000
    NUM_SAMPLES = SAMPLE_RATE * 1
    BATCH_SIZE = 32
    NUM_WORKERS = 8

    LEARNING_RATE = 0.0001
    EPOCHS = 5

    df = pd.read_csv(Path("../../../data/data.csv"))

    train_df, test_val_df = train_test_split(df, stratify=df['isBird'], test_size=0.2, random_state=SEED)
    val_df, test_df = train_test_split(test_val_df, stratify=test_val_df['isBird'], test_size=0.5, random_state=SEED)

    train_ds = BinaryDataset(train_df, bird_dir=BIRD_RECORDINGS_DIR, not_bird_dir=NOT_BIRD_RECORDINGS_DIR, noises_dir=NOISES_DIR, sample_rate=SAMPLE_RATE,
                            device=DEVICE)
    val_ds = BinaryDataset(val_df, bird_dir=BIRD_RECORDINGS_DIR, not_bird_dir=NOT_BIRD_RECORDINGS_DIR, noises_dir=NOISES_DIR, sample_rate=32000, device=DEVICE)
    test_ds = BinaryDataset(test_df, bird_dir=BIRD_RECORDINGS_DIR, not_bird_dir=NOT_BIRD_RECORDINGS_DIR, noises_dir=NOISES_DIR, sample_rate=32000, device=DEVICE)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.sort(df['isBird'].unique()),
                                                      y=df.loc[:, 'isBird']
                                                      )

    ##########

    cnn = CNNNetwork().to(DEVICE)
    summary(cnn, (1, 64, 251))

    cnn.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    writer = SummaryWriter(WRITER_DIR)
    epoch_number = 0

    best_vloss = sys.float_info.max

    print("--------------------")
    print(f"TIMESTAMP: {TIMESTAMP}")
    print(f"DEVICE: {DEVICE}")
    print(f"SEED: {SEED}")
    print(f"BASE_PATH: {BASE_PATH}")
    print(f"BIRD RECORDINGS_DIR: {BIRD_RECORDINGS_DIR}")
    print(f"NOT BIRD RECORDINGS_DIR: {NOT_BIRD_RECORDINGS_DIR}")
    print(f"NOISES_DIR: {NOISES_DIR}")
    print(f"WRITER_DIR: {WRITER_DIR}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"SAMPLE_RATE: {SAMPLE_RATE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"LEARNING_RATE: {LEARNING_RATE}")
    print(f"EPOCHS: {EPOCHS}")
    print("--------------------")


    start_time = time()

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        epoch_start_time = time()

        # Make sure gradient tracking is on, and do a pass over the data
        cnn.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, train_dl, optimizer, loss_fn, cnn, DEVICE, start_time)

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        cnn.eval()
        running_vloss = 0.0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_dl):
                vinputs, vlabels = vdata
                vinputs = torch.unsqueeze(vinputs, dim=1).to(DEVICE)
                voutputs = cnn(vinputs)
                vloss = loss_fn(voutputs, vlabels.to(DEVICE))
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print("#############################################################")
        print("Epoch results:")
        print(f'Loss train {avg_loss} valid loss: {avg_vloss}')
        validation_precision_score = calculate_metric(cnn, val_dl, device = DEVICE,
                                                      metric=lambda x, y: precision_score(x, y, average='macro'))
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
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'model_{TIMESTAMP}_{epoch_number}'
            torch.save(cnn.state_dict(), model_path)

        epoch_number += 1

    torch.save(cnn.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    main()