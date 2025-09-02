# scripts/utils/viz.py

# ─────────────────────────────────────────────────────────────
# Visualization helpers for data and metrics 
# ─────────────────────────────────────────────────────────────
import torch
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from torchmetrics import ConfusionMatrix

# ─────────────────────────────────────────────────────────────
# plot_class_distribution
# ─────────────────────────────────────────────────────────────

def plot_class_distribution(
    labels_csv: Path,
    outdir: Path,
    *,
    label: str = "emotion",
    split: str = "split",
    title_prefix: str = "Class distribution",
) -> dict[str, Path]:
    """
    Origin: Matplotlib (pyplot); pandas (groupby/crosstab).

    Notes
    -----
    - Overall chart: % normalized by the total number of samples.
    - By-split chart: % normalized within each split (column-wise).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(labels_csv)

    if label not in df.columns:
        raise ValueError(
            f"Label column '{label}' not found in {labels_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    out_paths: dict[str, Path] = {}

    # ---------- Overall distribution ----------
    counts = df[label].value_counts().sort_index()
    total = int(counts.sum())
    pct = (counts / total) * 100.0

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index.astype(str), counts.values)

    ax.set_title(f"{title_prefix} – overall")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count, Proportion")

    ymax = counts.max()
    off_hi = max(1.2, 0.06 * ymax)   # absolute (higher)
    off_lo = max(0.3, 0.015 * ymax)  # percent (lower)
    ax.set_ylim(0, ymax + off_hi * 2)

    for i, b in enumerate(bars):
        h = b.get_height()
        x = b.get_x() + b.get_width() / 2
        ax.text(x, h + off_hi, f"{int(h)}", ha="center", va="bottom", fontsize=9)
        ax.text(x, h + off_lo, f"{pct.iloc[i]:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    p_overall = outdir / "00_class_distribution_overall.png"
    fig.savefig(p_overall, dpi=300)
    plt.close(fig)
    out_paths["overall"] = p_overall

    # ---------- By-split distribution (grouped bars with inner spacing) ----------
    if split in df.columns:
        piv = pd.crosstab(df[label], df[split]).loc[counts.index]  # keep same class order

        import numpy as np

        classes = piv.index.astype(str).tolist()
        splits = piv.columns.astype(str).tolist()
        vals = piv.to_numpy()                        # shape: (n_classes, n_splits)
        n_classes, n_splits = vals.shape
        col_sums = piv.sum(axis=0).to_numpy()        # per-split totals

        x = np.arange(n_classes, dtype=float)

        group_width = 0.82       # total width allocated to each class group (<1 leaves gap between groups)
        inner_gap = 0.06         # gap *between* bars within a group (in axis units)
        bar_w = (group_width - inner_gap * (n_splits - 1)) / max(n_splits, 1)

        fig, ax = plt.subplots(figsize=(10, 5))

        ymax = vals.max() if vals.size else 1.0
        off_hi = max(1.2, 0.06 * ymax)   # absolute (higher)
        off_lo = max(0.3, 0.015 * ymax)  # percent (lower)
        ax.set_ylim(0, ymax + off_hi * 2)

        bars_by_split = []
        for j in range(n_splits):
            # center group at x, offset each split bar with inner gaps
            offset = -group_width / 2 + j * (bar_w + inner_gap) + bar_w / 2
            b = ax.bar(x + offset, vals[:, j], width=bar_w, label=splits[j])
            bars_by_split.append(b)

            # annotate each bar (absolute above, percent slightly lower)
            denom = col_sums[j] if col_sums[j] > 0 else 1.0
            for i in range(n_classes):
                h = vals[i, j]
                if h <= 0:
                    continue
                x_pos = x[i] + offset
                pct_val = (h / denom) * 100.0
                ax.text(x_pos, h + off_hi, f"{int(h)}", ha="center", va="bottom", fontsize=8)
                ax.text(x_pos, h + off_lo, f"{pct_val:.1f}%", ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{title_prefix} – by split")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count, Proportions")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=0)
        ax.legend(title=split)

        fig.tight_layout()
        p_split = outdir / "01_class_distribution_by_split.png"
        fig.savefig(p_split, dpi=300)
        plt.close(fig)
        out_paths["by_split"] = p_split

    return out_paths

def plot_training_history(
    history_csv: Path,
    outdir: Path,
    *,
    title_prefix: str | None = None,
) -> dict[str, Path]:
    """Visualize train and dev loss, accuracy and macro F1 over epochs.
    The CSV file is expected to be in long format with the following columns::
        epoch, split, loss, acc, macro_f1, lr, wall_time
    Parameters
    ----------
    history_csv:
        Path to the CSV file produced during training.
    outdir:
        Directory where the figures will be saved.
    title_prefix:
        Prefix for each figure title.
    Returns
    -------
    dict[str, Path]
        Mapping from metric name (``loss``, ``accuracy``, ``macro_f1``)
        to the path of the saved PNG figure.
    Notes
    -----
    - Early stopping epoch is assumed to be the final epoch recorded.
    - The best dev accuracy is highlighted on the accuracy plot.
    - The model name is inferred from ``history_csv``'s parent directory and
      included in figure titles and filenames.
    """
    df = pd.read_csv(history_csv)
    outdir.mkdir(parents=True, exist_ok=True)
    model_name = history_csv.parent.parent.stem
    if title_prefix is None:
        title_prefix = f"{model_name} training history"
    required_cols = {"epoch", "split", "loss", "acc", "macro_f1"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {history_csv}."
        )
    train_df = df[df["split"] == "train"].sort_values("epoch")
    dev_df = df[df["split"] == "dev"].sort_values("epoch")
    early_stop_epoch = int(df["epoch"].max())
    best_dev_idx = dev_df["acc"].idxmax()
    best_dev_epoch = int(dev_df.loc[best_dev_idx, "epoch"])
    best_dev_value = float(dev_df.loc[best_dev_idx, "acc"])
    out_paths: dict[str, Path] = {}
    def _plot(col: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(train_df["epoch"], train_df[col], label="train")
        ax.plot(dev_df["epoch"], dev_df[col], label="dev")
        ax.axvline(early_stop_epoch, color="red", linestyle="--", label="early stop")
        ax.axvline(best_dev_epoch, color="green", linestyle=":", label="best dev acc")
        if col == "acc":
            ax.scatter([best_dev_epoch], [best_dev_value], color="green", zorder=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} – {ylabel}")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        filename = f"{model_name}_training_history_{col}.png"
        out_path = outdir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        out_paths[col] = out_path
    _plot("loss", "Loss")
    _plot("acc", "Accuracy")
    _plot("macro_f1", "Macro F1")

    return out_paths

def plot_confusion_matrix(
    outdir: Path,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    *,
    model_name: str | None = None,
) -> Path:
    """Plot a 6-class confusion matrix from prediction CSV.
    The CSV is expected to contain a ``prediction`` column with integer labels
    corresponding to the CREMA-D emotions.
    Parameters
    ----------
    outdir:
        Directory where the confusion matrix figure will be saved.
    predictions:
        Tensor of predicted class indices or logits/probabilities (N or N×C).
    labels:
        Tensor of true class indices (N).
    model_name:
        Model name that was trained. 
    Returns
    -------
    Path
        Path to the saved PNG image.
    """

    cm_metric = ConfusionMatrix(task="multiclass", num_classes=6)
    cm = cm_metric(predictions, labels).numpy()

    slug = str(model_name).replace(" ", "_")
    classes = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(model_name)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    out_path = outdir / f"{slug}_confusion_matrix.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    
    return out_path



