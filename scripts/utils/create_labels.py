import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple


LABEL_FIELDS = ["path", "actor_id", "sentence", "emotion", "level"]
SPLIT_FIELDS = LABEL_FIELDS + ["split"]


def parse_filename(filename: str) -> Tuple[str, str, str, str]:
    """Parse CREMA-D style filename.

    Returns actor_id, sentence, emotion, level.
    """
    name, _ = os.path.splitext(filename)
    parts = name.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected filename format: {filename}")
    return tuple(parts)  # type: ignore


def collect_files(root: str) -> List[str]:
    paths = []
    for entry in os.scandir(root):
        if entry.is_file() and entry.name.lower().endswith((".wav", ".mp3")):
            paths.append(entry.path)
    return sorted(paths)


def split_dataset(rows: List[List[str]], dev_ratio: float = 0.1) -> List[List[str]]:
    """Assign a train/dev/test split according to sentence rules."""
    rng = random.Random(0)

    groups: dict[str, List[List[str]]] = defaultdict(list)
    for row in rows:
        groups[row[2]].append(row)

    test_only = {"TSI", "WSI"}
    half_dev = {"DFA", "ITS"}

    split_rows: List[List[str]] = []

    for sentence, items in groups.items():
        items = items[:]
        rng.shuffle(items)

        if sentence in test_only:
            for r in items:
                split_rows.append(r + ["test"])
            continue

        if sentence in half_dev:
            dev_count = len(items) // 2
            dev_items = items[:dev_count]
            rest = items[dev_count:]

            if rest:
                split_rows.append(rest.pop(0) + ["test"])

            for r in dev_items:
                split_rows.append(r + ["dev"])
            for r in rest:
                split_rows.append(r + ["train"])
            continue

        dev_count = max(1, int(len(items) * dev_ratio))
        dev_items = items[:dev_count]
        rest = items[dev_count:]

        if rest:
            split_rows.append(rest.pop(0) + ["test"])

        for r in dev_items:
            split_rows.append(r + ["dev"])
        for r in rest:
            split_rows.append(r + ["train"])

    return split_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create label CSV from filenames")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/unprocessed/crema-d/AudioWAV",
        help="Directory containing WAV or MP3 files",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="data/labels.csv",
        help="Path to save label CSV",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    files = collect_files(str(data_dir))
    rows = []
    for audio in files:
        actor_id, sentence, emotion, level = parse_filename(os.path.basename(audio))
        feature_path = Path(audio).with_suffix(".npy").resolve()
        rel_path = feature_path.relative_to(repo_root)
        rows.append([str(rel_path), actor_id, sentence, emotion, level])

    rows = split_dataset(rows)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SPLIT_FIELDS)
        writer.writerows(rows)

    meta_path = repo_root / "meta.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SPLIT_FIELDS)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} entries to {out_path}")


if __name__ == "__main__":
    main()

