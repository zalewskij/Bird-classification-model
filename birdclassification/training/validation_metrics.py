from birdclassification.preprocessing.spectrogram import generate_mel_spectrogram_seq
import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report


def calculate_metric(model, val_loader, preprocessing_pipeline, device, metric = None):
    """
    Calculate metric score for DataLoader instance
    (e.g.: macro/micro precision/recall/f1 score)

    Parameters
    ----------
    model: nn.Module
        model used to predict labels
    val_loader: torch.utils.data.DataLoader
        DataLoader used to calculate validation score

    metric:
        Examples of possible metrics:
        precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
        f1: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        recall: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html

    Returns
    -------
    Calculated metric: float
    """

    true_labels, predicted_labels = get_true_and_predicted_labels(model, val_loader, preprocessing_pipeline, device)
    return metric(true_labels, predicted_labels)


def get_true_and_predicted_labels(model, val_loader, preprocessing_pipeline, device):
    """
    Get true and predicted labels

    Parameters
    ----------
    model: nn.Module
        model used to predict labels
    val_loader: torch.utils.data.DataLoader
        DataLoader used to calculate validation score


    Returns
    -------
    true_labels, predicted_labels: torch.Tensor, torch.Tensor
    """

    model.to(device)
    model.eval()
    true_labels = torch.Tensor().to(device)
    predicted_labels = torch.Tensor().to(device)
    with torch.no_grad():
        for input, labels in val_loader:
            spectrogram = preprocessing_pipeline(input.to(device), use_augmentations=False)
            validation_output = model(spectrogram)
            predictions = torch.max(validation_output, dim=1)[1].data.squeeze()
            predicted_labels = torch.cat((predicted_labels, predictions))
            true_labels = torch.cat((true_labels, labels.to(device)))

    return true_labels.cpu(), predicted_labels.cpu()