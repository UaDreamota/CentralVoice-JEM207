# scripts/utils/dataset.py

"""Dataset utility for the CREMA-D speech-sentiment project
   *using *pre-computed* MFCC feature matrices stored as .npy files*.
"""

import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CremadPrecompDataset(Dataset):
    """Return **pre-computed** MFCC tensors and integer emotion labels.

    Parameters
    ----------
    root : str | Path
        Directory that contains the metadata CSV **plus** two sub-folders:

        * ``audio/``   – the original WAV clips (optional, never touched here)
        * ``mfcc/``    – one ``.npy`` file per clip, same basename as the WAV

    split : {"train", "dev", "test"}
        Partition to load (must match the ``split`` column in the CSV).

    meta_file : str, default "meta.csv"
        Filename of the metadata table.

    transform : callable, optional
        Optional mapping ``np.ndarray -> np.ndarray`` that is *applied
        **after** loading* (e.g. normalisation, padding, etc.).
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        meta_file: str = "meta.csv",
        transform: Optional[callable] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split.lower()
        self.meta_file = meta_file
        self.transform = transform  # may be None

        meta_path = self.root / self.meta_file
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file '{meta_path}' not found.")

        self.items: list[Tuple[Path, int]] = []
        with meta_path.open(newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("split", "").lower() != self.split:
                    continue

                # ---------- expected file locations -------------------------
                wav_file = self.root / "audio" / row["file"]          # not used
                npy_file = self.root / "mfcc" / Path(row["file"]).with_suffix(".npy")
                # ------------------------------------------------------------

                if not npy_file.exists():
                    raise FileNotFoundError(f"Feature file '{npy_file}' not found.")
                self.items.append((npy_file, int(row["emotion"])))

        if not self.items:
            raise RuntimeError(
                f"No samples for split '{self.split}' found in '{meta_path}'."
            )

    # ------------------------------------------------------------- Dataset API
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        npy_path, label = self.items[idx]

        feats_np: np.ndarray = np.load(npy_path)  # shape expected: (1, 1, C, T)

        if self.transform is not None:
            feats_np = self.transform(feats_np)

        if feats_np.ndim != 4:
            raise ValueError(
                "Each '.npy' must contain an array shaped (1, 1, C, T); "
                f"got {feats_np.shape} from '{npy_path}'."
            )

        feats = torch.from_numpy(feats_np).float()
        return feats, label

    # ------------------------------------------------------------- utilities
    def describe(self) -> str:
        return (
            f"CremadPrecompDataset(root='{self.root}', split='{self.split}', "
            f"size={len(self)})"
        )


# ----------------------------------------------------------------- __main__

def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Quick dataset sanity check")
    parser.add_argument("--root", required=True,
                        help="Folder with mfcc/, audio/ and meta.csv")
    parser.add_argument("--split", default="train",
                        choices=["train", "dev", "test"],
                        help="Dataset partition to inspect")
    parser.add_argument("--index", type=int, default=0,
                        help="Zero-based sample index")
    args = parser.parse_args()

    dataset = CremadPrecompDataset(args.root, split=args.split)
    features, label = dataset[args.index]
    print(dataset.describe())
    print(
        f"Sample {args.index}: features shape={tuple(features.shape)}, label={label}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
