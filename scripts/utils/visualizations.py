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

    Args
    ----
    labels_csv : Path
        Path to the labels CSV with a label column (e.g., "emotion").
    outdir : Path
        Directory to save images (created if missing).
    label : str, default "emotion"
        Name of the label column in the CSV.
    split : str, default "split"
        Name of the split/partition column; if absent, per-split plot is skipped.
    title_prefix : str, default "Class distribution"
        Prefix for figure titles (prepended to '– overall' / '– by split').

    Returns
    -------
    dict[str, Path]
        Mapping {'overall': <png_path>, 'by_split': <png_path_if_made>}.

    Usage
    -----
    # 1) Typical (label='emotion', split='split')
    paths = plot_class_distribution(Path("data/labels.csv"), Path("reports/data_overview"))

    # 2) Custom column names
    paths = plot_class_distribution(Path("labels.csv"), Path("reports"), label="target", split="partition")

    # 3) No split column present (only overall plot returned)
    paths = plot_class_distribution(Path("labels.csv"), Path("reports"), split="fold")
    """
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(labels_csv)

    if label not in df.columns:
        raise ValueError(
            f"Label column '{label}' not found in {labels_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    out_paths: dict[str, Path] = {}

    # Overall distribution
    counts = df[label].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(f"{title_prefix} – overall")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p_overall = outdir / "00_class_distribution_overall.png"
    fig.savefig(p_overall, dpi=150)
    plt.close(fig)
    out_paths["overall"] = p_overall

    # By-split distribution (if split column exists)
    if split in df.columns:
        piv = pd.crosstab(df[label], df[split]).loc[counts.index]  # keep same class order
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        piv.plot(kind="bar", ax=ax)
        ax.set_title(f"{title_prefix} – by split")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.legend(title=split)
        fig.tight_layout()
        p_split = outdir / "01_class_distribution_by_split.png"
        fig.savefig(p_split, dpi=150)
        plt.close(fig)
        out_paths["by_split"] = p_split

    return out_paths
