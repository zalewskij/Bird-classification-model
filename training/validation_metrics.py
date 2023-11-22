import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report


def calculate_metric(model, val_loader, average = 'macro', metric = None):
    """
    Calculate metric score for DataLoader instance
    (macro/micro precision/recall/f1 score)

    Parameters
    ----------
    model: nn.Module
        model used to predict labels
    val_loader: torch.utils.data.DataLoader
        DataLoader used to calculate validation score

    average: str
        Calculation mode, details about possible values:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

    metric:
        Possible metrics:
        precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
        f1: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        recall: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html

    """

    model.eval()
    true_labels = torch.Tensor()
    predicted_labels = torch.Tensor()
    with torch.no_grad():
        for images, labels in val_loader:
            images = torch.unsqueeze(images, dim=1)
            validation_output = model(images)
            predictions = torch.max(validation_output, dim = 1)[1].data.squeeze()
            predicted_labels = torch.cat((predicted_labels, predictions))
            true_labels = torch.cat((true_labels, labels))

    return metric(true_labels, predicted_labels, average=average)


def data_loader_balanced_accuracy(model, val_loader):
    """
    Calculate balanced accuracy score (macro)
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html

    Parameters
    ----------
    model: nn.Module
        model used to predict labels
    val_loader: torch.utils.data.DataLoader
        DataLoader used to calculate validation score
    """
    model.eval()
    true_labels = torch.Tensor()
    predicted_labels = torch.Tensor()
    with torch.no_grad():
        for images, labels in val_loader:
            images = torch.unsqueeze(images, dim=1)
            validation_output = model(images)

            predictions = torch.max(validation_output, dim=1)[1].data.squeeze()
            predicted_labels = torch.cat((predicted_labels, predictions))
            true_labels = torch.cat((true_labels, labels))

    return balanced_accuracy_score(true_labels, predicted_labels)

def data_loader_accuracy(model, val_loader):
    """
    Calculate accuracy score (micro)
    https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score

    Parameters
    ----------
    model: nn.Module
        model used to predict labels
    val_loader: torch.utils.data.DataLoader
        DataLoader used to calculate validation score
    """
    model.eval()
    true_labels = torch.Tensor()
    predicted_labels = torch.Tensor()
    with torch.no_grad():
        for images, labels in val_loader:
            images = torch.unsqueeze(images, dim=1)
            validation_output = model(images)
            predictions = torch.max(validation_output, dim=1)[1].data.squeeze()
            predicted_labels = torch.cat((predicted_labels, predictions))
            true_labels = torch.cat((true_labels, labels))

    return accuracy_score(true_labels, predicted_labels)

def get_confusion_matrix(model, val_loader):
    """
    Parameters
    ----------
    model: nn.Module
        model used to predict labels
    val_loader: torch.utils.data.DataLoader
        DataLoader used to calculate validation score

    Returns
    -------
    np.array
    Confusion matrix
    """
    true_labels = torch.Tensor()
    predicted_labels = torch.Tensor()
    with torch.no_grad():
        for images, labels in val_loader:
            images = torch.unsqueeze(images, dim=1)
            validation_output = model(images)
            predictions = torch.max(validation_output, dim=1)[1].data.squeeze()
            predicted_labels = torch.cat((predicted_labels, predictions))
            true_labels = torch.cat((true_labels, labels))

    return confusion_matrix(true_labels, predicted_labels)


def get_classification_report(model, val_loader):
    """
    Parameters
    ----------
    model: nn.Module
       model used to predict labels
    val_loader: torch.utils.data.DataLoader
       DataLoader used to calculate validation score

    Returns
    -------
    """
    true_labels = torch.Tensor()
    predicted_labels = torch.Tensor()
    with torch.no_grad():
        for images, labels in val_loader:
            images = torch.unsqueeze(images, dim=1)
            validation_output = model(images)
            predictions = torch.max(validation_output, dim=1)[1].data.squeeze()
            predicted_labels = torch.cat((predicted_labels, predictions))
            true_labels = torch.cat((true_labels, labels))

    print(classification_report(true_labels, predicted_labels))