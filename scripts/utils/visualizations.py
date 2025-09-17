# scripts/utils/vizualizations.py
"""
VISUALIZATION HELPERS for data and metrics
"""

from __future__ import annotations
import torch

from pathlib import Path
from typing import Sequence, Optional
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch 
from torchmetrics.classification import ConfusionMatrix

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
    fig.savefig(p_overall, dpi=500)
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

        ax.set_title(f"{title_prefix} - by split")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=0)
        ax.legend(title=split)

        fig.tight_layout()
        p_split = outdir / "01_class_distribution_by_split.png"
        fig.savefig(p_split, dpi=500)
        plt.close(fig)
        out_paths["by_split"] = p_split

    return out_paths

# ─────────────────────────────────────────────────────────────
# plot_training_history
# ─────────────────────────────────────────────────────────────

def plot_training_history(
    history_csv: Path,
    outdir: Path,
    *,
    title_prefix: str | None = None,
) -> dict[str, Path]:
    """Visualize train & dev loss/accuracy/macro F1 over epochs with integer x-axis."""
    df = pd.read_csv(history_csv)
    outdir.mkdir(parents=True, exist_ok=True)

    model_name = history_csv.parent.parent.stem
    if title_prefix is None:
        title_prefix = model_name

    required_cols = {"epoch", "split", "loss", "acc", "macro_f1"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {history_csv}.")

    train_df = df[df["split"] == "train"].sort_values("epoch")
    dev_df   = df[df["split"] == "dev"].sort_values("epoch")

    # epochs for integer ticks/limits
    epochs = np.sort(df["epoch"].unique().astype(int))
    early_stop_epoch = int(df["epoch"].max())

    out_paths: dict[str, Path] = {}

    def _plot(col: str, ylabel: str) -> None:
        # widen figure a bit based on number of epochs (capped)
        width = min(18, 8 + 0.06 * len(epochs))  # tweak if you want more/less space
        fig, ax = plt.subplots(figsize=(width, 5))

        ax.plot(train_df["epoch"], train_df[col], label="train")
        ax.plot(dev_df["epoch"],   dev_df[col],   label="dev")

        # best dev point per metric (loss=min, others=max)
        idx = dev_df[col].idxmin() if col == "loss" else dev_df[col].idxmax()
        best_epoch = int(dev_df.loc[idx, "epoch"])
        best_value = float(dev_df.loc[idx, col])

        # vertical markers
        ax.axvline(early_stop_epoch, color="red",   linestyle="--", label="early stop")
        ax.axvline(
            best_epoch,
            color="green",
            linestyle=":",
            label=f"best dev {ylabel.lower()} = {best_value:.4f}",
        )

        # green dot at best dev point
        ax.scatter([best_epoch], [best_value], color="green", zorder=5)

        # labels/title
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} – {ylabel}")

        # integer ticks + full list of epochs
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if epochs.size == 1:
            ax.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
            ax.set_xticks([int(epochs[0])])
        else:
            start = 0
            end = int(epochs.max())
            ax.set_xlim(start - 0.5, end + 0.5)
            ticks = np.arange(start, end + 1, 5, dtype=int)
            ax.set_xticks(ticks)

        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        filename = f"{model_name}_{col}.png"
        out_path = outdir / filename
        fig.savefig(out_path, dpi=500)
        plt.close(fig)
        out_paths[col] = out_path

    _plot("loss", "Loss")
    _plot("acc", "Accuracy")
    _plot("macro_f1", "Macro F1")

    return out_paths


# ─────────────────────────────────────────────────────────────
# plot_confusion_matrix
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    outdir: Path,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    *,
    model_name: Optional[str] = None,
    pred_names: Sequence[str] = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD"),
    true_names: Optional[Sequence[str]] = None,
    reverse_true: bool = False,
    hspace: float = 0.6,  
) -> Path:
    """
    Row-normalized confusion matrix (%) + per-class recall bar strip (TRUE order).

    - x-axis (top heatmap) shows PREDICTED labels (pred_names).
    - y-axis (heatmap) and the bar chart below use TRUE labels (true_names),
      optionally reversed with reverse_true=True.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    preds = torch.as_tensor(predictions)
    targs = torch.as_tensor(labels)
    if preds.ndim > 1:
        preds = preds.argmax(dim=1)
    if targs.ndim > 1:
        targs = targs.argmax(dim=1)

    num_classes = len(pred_names)
    if true_names is None:
        true_names = list(pred_names)
    if reverse_true:
        true_names = list(true_names)[::-1]

    # 1) Raw CM (counts) in canonical index order 0..C-1
    cm_raw = ConfusionMatrix(task="multiclass", num_classes=num_classes)(preds, targs).cpu().numpy()

    # 2) Row-normalize (true-label normalization)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.where(row_sums > 0, cm_raw / row_sums, 0.0)  # rows=true, cols=pred

    # 3) Reorder for display
    col_idx = [pred_names.index(nm) for nm in pred_names]              # predicted order (usually 0..C-1)
    row_idx = [pred_names.index(nm) for nm in true_names]              # true order (custom/reversed ok)
    cm_disp = cm_norm[np.ix_(row_idx, col_idx)]

    # Per-class recall (TRUE order chosen for display)
    per_class_recall = np.array([cm_norm[pred_names.index(nm), pred_names.index(nm)] for nm in true_names])

    # ---- Figure: heatmap + recall bars ----
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[12, 2.6], hspace=hspace)
    ax = fig.add_subplot(gs[0])
    # sharex keeps bar alignment with columns; we will override tick labels below
    ax_bar = fig.add_subplot(gs[1], sharex=ax)

    im = ax.imshow(cm_disp * 100.0, cmap="Blues", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized (%)", rotation=90)

    # Heatmap ticks/labels
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(pred_names, rotation=45, ha="right")
    ax.set_yticklabels(true_names)
    ax.set_xlabel("Predicted emotions")
    ax.set_ylabel("True emotions")
    if model_name:
        ax.set_title(model_name)

    # Cell annotations; WHITE text only when cell value >= 50%
    fixed_thresh = 50.0
    for i in range(num_classes):
        for j in range(num_classes):
            val_pct = cm_disp[i, j] * 100.0
            ax.text(
                j, i, f"{val_pct:.0f}%",
                ha="center", va="center",
                color="white" if val_pct >= fixed_thresh else "black",
                fontsize=9,
            )

    # Recall bars (TRUE order on x-axis)
    x = np.arange(num_classes)
    bars = ax_bar.bar(x, per_class_recall, width=0.7, color="#2b7bbb", alpha=0.85)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Recall")

    # Show emotion labels on the bar chart's x-axis (TRUE order)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(true_names, rotation=0)  # rotate if you want
    ax_bar.set_xlabel("True label")
    ax_bar.grid(axis="y", alpha=0.25)

    # Annotate bars with percentages
    for b in bars:
        h = b.get_height()
        ax_bar.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h*100:.0f}%", ha="center", va="bottom", fontsize=8)

    slug = (model_name or "model").replace(" ", "_")
    out_path = outdir / f"{slug}_confusion_matrix.png"
    fig.savefig(out_path, dpi=500)
    plt.close(fig)

    return out_path