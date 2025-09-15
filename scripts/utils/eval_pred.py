# scripts/utils/eval_pred.py

import csv
from pathlib import Path
from typing import List

import torch
from torchmetrics import Accuracy, ConfusionMatrix, F1Score
import pandas as pd

from scripts.utils.datasets import CremadPrecompDataset, CREMA_ROOT, get_dev_transform


CLASS_NAMES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]


def evaluate_predictions(log_dir: str, class_names=CLASS_NAMES, verbose: bool = True) -> tuple[float, list[float], list[int], list[int]]:
    """
    Evaluate predictions stored in `predictions.csv` inside a run directory.

    Parameters
    ----------
    log_dir : str | PathLike
        Path to the run directory containing `predictions.csv`.
    class_names : Sequence[str], optional
        Ordered class labels corresponding to numeric targets. Defaults to the module-level `CLASS_NAMES`.
    verbose : bool, default True
        If True, prints the confusion matrix and metric summaries to stdout.

    Returns
    -------
    accuracy : float
        Overall accuracy across the test set.
    per_class_recall : list of float
        Per-class recall (class-wise accuracy) in the same order as `class_names`.
    preds : list of int
        Model predicted class indices.
    labels : list of int
        Ground-truth class indices.

    Raises
    ------
    FileNotFoundError
        If `predictions.csv` does not exist at the specified location.
    ValueError
        If the number of predictions does not match the number of test samples.

    Notes
    -----
    Per-class values are recall (true positives divided by support) because
    `torchmetrics.Accuracy` with `average='none'` reports class-wise correct / support.
    A macro F1-score and a confusion matrix are computed for logging only and are not returned.
    The function assumes the ordering of `predictions.csv` matches the dataset iteration order.
    """
    pred_path = Path(log_dir) / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file '{pred_path}' not found.")

    preds: List[int] = []
    with pred_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "prediction" not in reader.fieldnames:
            raise ValueError("Missing 'prediction' column in predictions.csv.")
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
            f"Number of predictions ({len(preds)}) does not match number of test samples ({len(labels)})"
        )

    preds_t = torch.tensor(preds)
    labels_t = torch.tensor(labels)

    acc_metric = Accuracy(task="multiclass", num_classes=len(class_names))
    accuracy = acc_metric(preds_t, labels_t).item()

    class_acc_metric = Accuracy(task="multiclass", num_classes=len(class_names), average="none")
    class_accuracy = class_acc_metric(preds_t, labels_t).tolist()
    per_class_acc = dict(zip(class_names, class_accuracy))

    f1_metric = F1Score(task="multiclass", num_classes=len(class_names), average="macro")
    f1_macro = f1_metric(preds_t, labels_t).item()

    cm_metric = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
    confusion = cm_metric(preds_t, labels_t)
    cm_df = pd.DataFrame(confusion.numpy(), index=class_names, columns=class_names)

    if verbose:
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm_df)
        print(f"Evaluation - overall accuracy: {accuracy:.4f}")
        print("Evaluation - per-class recall (normalized):")
        for name in class_names:
            print(f"  {name}: {per_class_acc[name]:.4f}")
        print(f"Macro F1-score: {f1_macro:.4f}")

    return accuracy, class_accuracy, preds, labels