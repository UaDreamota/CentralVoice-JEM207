# scripts/utils/viz.py

# ─────────────────────────────────────────────────────────────
# Visualization helpers for data and metrics 
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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

