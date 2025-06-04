# utils/dataset.py
"""Dataset utilities for the CREMA-D speech-sentiment project.

A *single* public class is exported: :class:`CremadDataset`.
It
1. reads a CSV metadata file sitting next to an ``audio/`` directory,
2. selects rows whose ``split`` column equals the argument (``train``/``dev``/``test``),
3. loads each waveform with :pyfunc:`utils.audio_fearures.load_audio`,
4. converts the signal to MFCCs via :pyfunc:`utils.audio_fearures.extract_mfcc` (or any feature function you inject),
5. returns a **tuple** ``(features : torch.FloatTensor, label : int)`` directly usable by a PyTorch ``DataLoader``.

The module also provides a small ``main`` for ad‑hoc inspection:

```bash
python -m utils.dataset --root data/cremad --split dev --index 42
```
"""

import argparse
import csv
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Relative import from the same utils package
from .audio_fearures import load_audio, extract_mfcc  # noqa: E402


class CremadDataset(Dataset):
    """Return MFCC features and integer emotion labels from CREMA‑D.

    Parameters
    ----------
    root : str or Path
        Directory containing *both* the metadata CSV and the ``audio/``
        directory with WAV clips.
    split : {"train", "dev", "val", "test"}
        Partition to load. The string must exactly match the value in the
        ``split`` column of the CSV.
    meta_file : str, default "meta.csv"
        Filename of the metadata table.
    feature_fn : Callable, optional
        Custom function ``(signal : np.ndarray, sr : int) -> np.ndarray`` that
        converts a mono waveform to any time‐frequency representation.
        Defaults to 40‑coefficient MFCCs via
        :pyfunc:`utils.audio_fearures.extract_mfcc`.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        meta_file: str = "meta.csv",
        feature_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split.lower()
        self.meta_file = meta_file
        self.feature_fn = feature_fn or (
            lambda sig, sr: extract_mfcc(sig, sr, n_mfcc=40)
        )

        meta_path = self.root / self.meta_file
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file '{meta_path}' not found.")

        self.items: list[Tuple[Path, int]] = []
        with meta_path.open(newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("split", "").lower() != self.split:
                    continue
                wav_file = self.root / "audio" / row["file"]
                if not wav_file.exists():
                    raise FileNotFoundError(f"Audio file '{wav_file}' not found.")
                self.items.append((wav_file, int(row["emotion"])))

        if not self.items:
            raise RuntimeError(
                f"No samples for split '{self.split}' found in '{meta_path}'."
            )

    # ------------------------------------------------------------------ Dataset
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        wav_path, label = self.items[idx]
        signal, sr = load_audio(str(wav_path), sr=16_000)  # ensures mono, 16 kHz
        feats_np = self.feature_fn(signal, sr)

        if feats_np.ndim != 4:
            raise ValueError(
                "feature_fn must return an array shaped (1, 1, C, T); "
                f"got {feats_np.shape} instead."
            )

        feats = torch.from_numpy(feats_np).float()
        return feats, label

    # ----------------------------------------------------------- convenience API
    def describe(self) -> str:
        """"""
        return (
            f"CremadDataset(root='{self.root}', split='{self.split}', "
            f"size={len(self)})"
        )



# --------------------------------------------------------------------- __main__

def _main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Quick dataset sanity check")
    parser.add_argument("--root", required=True, help="Folder with audio/ and meta.csv")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"],
                        help="Dataset partition to inspect")
    parser.add_argument("--index", type=int, default=0, help="Zero‑based sample index")
    args = parser.parse_args()

    dataset = CremadDataset(args.root, split=args.split)
    features, label = dataset[args.index]
    print(dataset.describe())
    print(
        f"Sample {args.index}: features shape={tuple(features.shape)}, label={label}")


if __name__ == "__main__":  # pragma: no cover
    _main()
