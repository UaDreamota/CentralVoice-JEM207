# scripts/utils/eval_pred.py

import csv
from pathlib import Path
from typing import List

import torch
from torchmetrics import Accuracy, ConfusionMatrix, F1Score
import pandas as pd

from scripts.utils.datasets import CremadPrecompDataset, CREMA_ROOT, get_dev_transform


CLASS_NAMES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]

def evaluate_predictions(log_dir: str, class_names = CLASS_NAMES) -> tuple[float, list[float], list[int], list[int]]:
    """Evaluate predictions saved in ``predictions.csv`` inside ``log_dir``.

    Parameters
    ----------
    log_dir : str
        Path to the run directory containing ``predictions.csv``.

    Returns
    -------
    tuple[float, list[float], list[int], list[int]]
        Overall accuracy, per-class accuracy list, raw predictions, raw labels.
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
    per_class_acc = dict(zip(class_names, class_accuracy))

    f1_metric = F1Score(task='multiclass', num_classes=6, average='macro')
    f1_macro = f1_metric(preds_t, labels_t).item()

    cm_metric = ConfusionMatrix(task="multiclass", num_classes=6)
    confusion = cm_metric(preds_t, labels_t)
    cm_df = pd.DataFrame(confusion.numpy(), index=class_names, columns=class_names)

    print("Confusion matrix (rows=true, cols=pred):")
    print(cm_df)
    print(f"Evaluation - overall accuracy: {accuracy:.4f}")
    print("Evaluation - per-class recall (normalized):")
    for name in class_names:
        print(f"  {name}: {per_class_acc[name]:.4f}")
    print(f"Macro F1-score: {f1_macro:.4f}")

    return accuracy, class_accuracy, preds, labels