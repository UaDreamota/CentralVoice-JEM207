import csv
from pathlib import Path
from typing import List

import torch
from torchmetrics import Accuracy, ConfusionMatrix

from scripts.utils.datasets import CremadPrecompDataset, CREMA_ROOT, get_dev_transform


def evaluate_predictions(log_dir: str) -> tuple[float, list[float]]:
    """Evaluate predictions saved in ``predictions.csv`` inside ``log_dir``.

    Parameters
    ----------
    log_dir : str
        Path to the run directory containing ``predictions.csv``.

    Returns
    -------
    tuple[float, list[float]]
        Overall accuracy and per-class accuracy list.
    """
    pred_path = Path(log_dir) / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file '{pred_path}' not found.")

    preds: List[int] = []
    with pred_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds.append(int(row["prediction"]))

    test_ds = CremadPrecompDataset(
        root=CREMA_ROOT,
        split="test",
        train_transform=None,
        dev_transform=get_dev_transform(),
    )
    labels = [label for _, label in test_ds]

    if len(preds) != len(labels):
        raise ValueError(
            f"Number of predictions ({len(preds)}) does not match "
            f"number of test samples ({len(labels)})"
        )

    preds_t = torch.tensor(preds)
    labels_t = torch.tensor(labels)

    acc_metric = Accuracy(task="multiclass", num_classes=6)
    accuracy = acc_metric(preds_t, labels_t).item()

    class_acc_metric = Accuracy(task="multiclass", num_classes=6, average="none")
    class_accuracy = class_acc_metric(preds_t, labels_t).tolist()

    cm_metric = ConfusionMatrix(task="multiclass", num_classes=6)
    confusion = cm_metric(preds_t, labels_t)
    print("Confusion matrix:\n", confusion)
    print(f"Accuracy: {accuracy:.4f}")
    print("Class accuracy:", class_accuracy)

    return accuracy, class_accuracy