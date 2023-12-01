import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report


def calculate_metric(model, val_loader, device, metric = None):
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

    model.to(device)
    model.eval()
    true_labels = torch.Tensor().to(device)
    predicted_labels = torch.Tensor().to(device)
    with torch.no_grad():
        for images, labels in val_loader:
            images = torch.unsqueeze(images, dim=1)
            validation_output = model(images.to(device))
            predictions = torch.max(validation_output, dim=1)[1].data.squeeze()
            predicted_labels = torch.cat((predicted_labels, predictions))
            true_labels = torch.cat((true_labels, labels.to(device)))

    return metric(true_labels.cpu(), predicted_labels.cpu())