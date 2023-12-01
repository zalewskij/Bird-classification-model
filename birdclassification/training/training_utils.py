from time import time
import torch


def train_one_epoch(epoch_index, tb_writer, training_loader, optimizer, loss_fn, model, device, start_time):
    """
    Function to train model in epoch

    Parameters
    ----------
    epoch_index: int
    tb_writer: TensorBoard writer object
    training_loader: torch.DataLoader
    optimizer: torch.optim Optimizer
    loss_fn: torch nn Loss
    model: nn.Module
    device: str
        cpu or cuda

    Returns
    -------
    last_loss: float
        Value of the loss in the epoch
    """
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs = torch.unsqueeze(inputs, dim=1)
        #labels = labels.to(torch.int64)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs.to(device))

        # Compute the loss and its gradients
        loss = loss_fn(outputs.to(device), labels.long().to(device))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10  # loss per batch
            print(f'Loss after batch {i + 1}: {last_loss:.4f}; time elapsed: {time() - start_time:.2f}')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss