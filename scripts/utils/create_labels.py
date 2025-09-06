import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

LABEL_FIELDS = ["path", "actor_id", "sentence", "emotion", "level"]
SPLIT_FIELDS = LABEL_FIELDS + ["split"]

def parse_filename(filename: str) -> Tuple[str, str, str, str]:
    """Parse CREMA-D style filename -> (actor_id, sentence, emotion, level)."""
    name, _ = os.path.splitext(filename)
    parts = name.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected filename format: {filename}")
    return tuple(parts)  # type: ignore

def collect_files(root: str) -> List[str]:
    paths = []
    for entry in os.scandir(root):
        if entry.is_file() and entry.name.lower().endswith((".wav", ".mp3", ".npy")):
            paths.append(entry.path)
    return sorted(paths)

def _counts_for_group(n: int, ratios=(0.8, 0.1, 0.1)) -> Tuple[int, int, int]:
    """Integer counts (train, dev, test) that sum to n for given ratios."""
    rt, rd, rte = ratios
    t = int(n * rt)
    d = int(n * rd)
    te = int(n * rte)
    # distribute any rounding remainder
    remainder = n - (t + d + te)
    order = sorted([("train", rt), ("dev", rd), ("test", rte)],
                   key=lambda kv: kv[1], reverse=True)
    for k, _ in order:
        if remainder <= 0:
            break
        if k == "train":
            t += 1
        elif k == "dev":
            d += 1
        else:
            te += 1
        remainder -= 1
    return t, d, te

def split_dataset_stratified_by_emotion(rows: List[List[str]],
                                        ratios=(0.8, 0.1, 0.1),
                                        seed: int = 0) -> List[List[str]]:
    """Stratified 80/10/10 split within each emotion, then merged."""
    rng = random.Random(seed)

    # group rows by emotion (index 3)
    groups: Dict[str, List[List[str]]] = {}
    for r in rows:
        groups.setdefault(r[3], []).append(r)

    split_rows: List[List[str]] = []
    for emotion, items in groups.items():
        items = items[:]  # copy
        rng.shuffle(items)
        n_train, n_dev, n_test = _counts_for_group(len(items), ratios)

        # assign labels within the group
        idx = 0
        for r in items[idx:idx + n_train]:
            split_rows.append(r + ["train"])
        idx += n_train
        for r in items[idx:idx + n_dev]:
            split_rows.append(r + ["dev"])
        idx += n_dev
        for r in items[idx:idx + n_test]:
            split_rows.append(r + ["test"])

    return split_rows

def main() -> None:
    parser = argparse.ArgumentParser(description="Create label CSV from filenames")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/processed",
        help="Directory containing WAV/MP3/NPY files",
    )
    parser.add_argument(
        "-o", "--out",
        default="data/labels.csv",
        help="Path to save label CSV",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for splitting",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    files = collect_files(str(data_dir))

    rows: List[List[str]] = []
    for audio in files:
        actor_id, sentence, emotion, level = parse_filename(os.path.basename(audio))
        feature_path = Path(audio).with_suffix(".npy").resolve()
        rel_path = feature_path.relative_to(repo_root)
        rows.append([str(rel_path), actor_id, sentence, emotion, level])

    rows = split_dataset_stratified_by_emotion(rows, ratios=(0.8, 0.1, 0.1), seed=args.seed)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SPLIT_FIELDS)
        writer.writerows(rows)

    # extra copy as before
    meta_path = repo_root / "mela.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SPLIT_FIELDS)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} entries to {out_path}")

if __name__ == "__main__":
    main()
