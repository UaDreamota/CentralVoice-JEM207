# scripts/utils/dataset.py

"""
Dataset utility for the CREMA-D speech-sentiment project
*using *pre-computed* MFCC feature matrices stored as .npy files*.
"""

import argparse
import random
import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchaudio.transforms import FrequencyMasking, TimeMasking

# Repository root and dataset location
REPO_ROOT = Path(__file__).resolve().parents[2]
CREMA_ROOT = REPO_ROOT / "data"

# ─────────────────────────────────────────────────────────────
### DATASET
# ─────────────────────────────────────────────────────────────

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
        meta_file: str = "labels.csv",
        train_transform: Optional[callable] = None,
        dev_transform: Optional[callable] = None
    ) -> None:
        self.root = Path(root)
        self.split = split.lower()
        self.meta_file = meta_file
        self.train_transform = train_transform
        self.dev_transform = dev_transform


        meta_path = self.root / self.meta_file
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file '{meta_path}' not found.")

        self.items: list[Tuple[Path, int]] = []
        with meta_path.open(newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("split", "").lower() != self.split:
                    continue

                # --------------- resolve feature path ---------------------
                # Older label CSVs used a 'file' column containing just the
                # basename; newer ones store a 'path' column relative to the
                # repository root. Support both for backward compatibility.
                file_field = row.get("file") or row.get("path")
                if file_field is None:
                    raise KeyError("Metadata missing 'file'/'path' column")

                if row.get("file"):
                    npy_file = self.root / "mfcc" / Path(file_field).with_suffix(".npy")
                else:
                    p = Path(file_field)
                    npy_file = p if p.is_absolute() else REPO_ROOT / p

                if not npy_file.exists():
                    raise FileNotFoundError(f"Feature file '{npy_file}' not found.")

                emotion_raw = row["emotion"]
                try:
                    label = int(emotion_raw)
                except ValueError:
                    EMOTION_MAP = {
                        "ANG": 0,
                        "DIS": 1,
                        "FEA": 2,
                        "HAP": 3,
                        "NEU": 4,
                        "SAD": 5,
                    }
                    if emotion_raw not in EMOTION_MAP:
                        raise ValueError(f"Unknown emotion label '{emotion_raw}'")
                    label = EMOTION_MAP[emotion_raw]

                self.items.append((npy_file, label))

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

        if self.split == "train" and self.train_transform:
            feats = self.train_transform(feats_np)
        elif self.split in {"dev", "val", "test"} and self.dev_transform:
            feats = self.dev_transform(feats_np)
        else:
            # No transform provided for this split; just convert to a tensor
            # to keep the downstream code happy
            feats = torch.tensor(feats_np, dtype=torch.float)

        if feats.ndim != 4:
            raise ValueError(
                "Each '.npy' must contain an array shaped (1, 1, C, T); "
                f"got {feats_np.shape} from '{npy_path}'."
            )

        # Remove the dummy leading dimension so DataLoader yields
        # tensors of shape (B, 1, 40, 218) instead of (B, 1, 1, 40, 218).
        if feats.shape[0] != 1:
            raise ValueError(
                f"Expected leading dimension of size 1, got {feats.shape[0]} from '{npy_path}'."
            )
        feats = feats.squeeze(0)

        return feats, label

    # ------------------------------------------------------------- utilities
    def describe(self) -> str:
        return (
            f"CremadPrecompDataset(root='{self.root}', split='{self.split}', "
            f"size={len(self)})"
        )
    
# ─────────────────────────────────────────────────────────────
### DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────

# ------------- Train transformation -------------------------------------------

class TrainTransform:
    """
    Training transform for MFCC tensors shaped (1, C, 40, T_var) -> (1, C, 40, time_length).
    Applies (in order): pad/truncate → *paper-style* augmentation (optional) →
    local z-score normalization → SpecAugment (freq/time masking).

    Paper-style augmentation (feature-space approximation):
      • One of {time-shift, white-noise} chosen uniformly per sample.
      • Time-shift range: ±time_shift_ms (converted to frames via hop_ms).
      • White noise strength: alpha ∈ [0, noise_abs_max] scaled by sample std.

    Args:
      time_length (int): target number of time frames (default 218).
      freq_mask_param (int): SpecAugment max masked freq bins (default 15).
      time_mask_param (int): SpecAugment max masked time frames (default 25).
      enable_paper_aug (bool): turn paper-style augmentation on/off (default True).
      time_shift_ms (int): max absolute time-shift in milliseconds (default 350).
      hop_ms (float): analysis hop in ms to convert ms→frames (default 16.0).
      noise_abs_max (float): max white-noise factor in [0, 0.2] (default 0.2).
      shift_mode (str): 'zero' (zero-fill the wrapped region) or 'roll' (circular).
      p_aug (float): probability to apply augmentation to a given sample (default 1.0).
      rng (random.Random | None): RNG for reproducibility.

    Returns:
      torch.Tensor: (1, C, 40, time_length), float32.
    """
    def __init__(self,
                 time_length: int = 218,
                 freq_mask_param: int = 15,
                 time_mask_param: int = 25,
                 *,
                 enable_paper_aug: bool = True,
                 time_shift_ms: int = 350,
                 hop_ms: float = 16.0,
                 noise_abs_max: float = 0.2,
                 shift_mode: str = "zero",
                 p_aug: float = 1.0,
                 rng: Optional[random.Random] = None):
        self.time_length = time_length
        self.freq_mask = FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = TimeMasking(time_mask_param=time_mask_param)

        self.enable_paper_aug = enable_paper_aug
        self.frames_max = int(round(time_shift_ms / hop_ms))  # ≈ 22 for 350/16
        self.noise_abs_max = float(noise_abs_max)
        self.shift_mode = shift_mode
        self.p_aug = float(p_aug)
        if rng is None:
            self.rng = random.Random()
            # Mirror the state of Python's global RNG so that calling
            # `random.seed(...)` in the training script is sufficient to make
            # the augmentation deterministic across runs.
            self.rng.setstate(random.getstate())
        else:
            self.rng = rng

    def _time_shift_inplace(self, x: torch.Tensor, k: int) -> None:
        """Shift along last dim by k frames; zero-fill the wrapped chunk if shift_mode == 'zero'."""
        if k == 0:
            return
        T = x.shape[-1]
        # roll first (cheap), then optionally zero the wrapped region
        x.copy_(torch.roll(x, shifts=k, dims=-1))
        if self.shift_mode == "zero":
            if k > 0:
                x[..., :k] = 0
            else:
                x[..., T + k:] = 0  # k is negative

    def _white_noise_inplace(self, x: torch.Tensor, alpha: float) -> None:
        """Add zero-mean Gaussian noise scaled by alpha * std(x)."""
        if alpha <= 0:
            return
        std = x.std()
        if torch.isfinite(std) and std > 0:
            x.add_(torch.randn_like(x) * (alpha * std))

    def __call__(self, mfcc):
        # 1) to torch.float32
        if not isinstance(mfcc, torch.Tensor):
            mfcc = torch.tensor(mfcc, dtype=torch.float)
        if mfcc.ndim != 4:
            raise ValueError(f"Expected 4D tensor (1, C, H_var, T_var), got {mfcc.shape}.")
        B, C, H, T = mfcc.shape
        if H != 40:
            raise ValueError(f"Expected H=40 (MFCC bins), got H={H}.")

        # 2) pad/truncate in time
        if T < self.time_length:
            mfcc = F.pad(mfcc, (0, self.time_length - T))
        elif T > self.time_length:
            mfcc = mfcc[..., :self.time_length]

        # 3) paper-style augmentation (feature-space), before normalization
        if self.enable_paper_aug and self.rng.random() < self.p_aug:
            # choose augmentation uniformly
            if self.rng.random() < 0.5:
                # time-shift: uniform integer in [-frames_max, frames_max]
                k = self.rng.randint(-self.frames_max, self.frames_max)
                if k != 0:
                    for b in range(B):
                        for c in range(C):
                            self._time_shift_inplace(mfcc[b, c], k)
            else:
                # white noise: alpha ~ U(0, noise_abs_max)
                alpha = self.rng.random() * self.noise_abs_max
                self._white_noise_inplace(mfcc, alpha)

        # 4) local z-score normalization (over all pixels of the MFCC image)
        mean = mfcc.mean()
        std = mfcc.std()
        mfcc = (mfcc - mean) / (std + 1e-9)

        # 5) SpecAugment masking (operate per [B, C] slice on 2D [40, T])
        B, C, H, T = mfcc.shape
        for b in range(B):
            for c in range(C):
                slice2d = mfcc[b, c, :, :]
                slice2d = self.freq_mask(slice2d)
                slice2d = self.time_mask(slice2d)
                mfcc[b, c, :, :] = slice2d

        return mfcc


class DevTransform:
    """
    Validation/Test transform: pad/truncate → local z-score normalization.
    No augmentation applied.
    """
    def __init__(self, time_length: int = 218):
        self.time_length = time_length

    def __call__(self, mfcc):
        if not isinstance(mfcc, torch.Tensor):
            mfcc = torch.tensor(mfcc, dtype=torch.float)
        if mfcc.ndim != 4:
            raise ValueError(f"Expected 4D tensor (1, C, H_var, T_var), got {mfcc.shape}.")

        B, C, H, T = mfcc.shape
        if H != 40:
            raise ValueError(f"Expected H=40 (MFCC bins), got H={H}.")

        if T < self.time_length:
            mfcc = F.pad(mfcc, (0, self.time_length - T))
        elif T > self.time_length:
            mfcc = mfcc[..., :self.time_length]

        mean = mfcc.mean()
        std = mfcc.std()
        mfcc = (mfcc - mean) / (std + 1e-9)
        return mfcc




    
# ------------- Train transformation wrapper -------------------------------------------

def get_train_transform(time_length: int = 218,
                        freq_mask_param: int = 15,
                        time_mask_param: int = 25,
                        **paper_aug_kwargs):
    """
    Factory for TrainTransform. Pass paper-style augmentation options via **paper_aug_kwargs.
    Examples: enable_paper_aug=True, time_shift_ms=350, noise_abs_max=0.2, shift_mode='zero'.
    """
    return TrainTransform(
        time_length=time_length,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        **paper_aug_kwargs
    )

# ------------- Dev transformation wrapper -------------------------------------------

def get_dev_transform(time_length: int = 218):
    """
    Returns an instance of DevTransform with the specified time_length.

    Args:
    -----
    time_length : int
        Fixed number of time‐frames after padding/truncation (default: 218).

    Returns:
    --------
    DevTransform
    """
    return DevTransform(time_length=time_length)
    
# ─────────────────────────────────────────────────────────────
### DATALOADERS
# ─────────────────────────────────────────────────────────────
    
def create_dataloaders(
    batch_size: int,
    time_length: int = 218,
    freq_mask_param: int = 15,
    time_mask_param: int = 25,
    *,
    seed: int | None = None,
    rng: Optional[random.Random] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, dev, and test dataloaders for CREMA-D using
    CremadPrecompDataset. Pass the appropriate transforms into each.
    """
    if rng is not None and seed is not None:
        raise ValueError("Pass either 'seed' or 'rng', not both.")

    train_rng: random.Random
    if rng is not None:
        train_rng = rng
    elif seed is not None:
        train_rng = random.Random(seed)
    else:
        train_rng = random.Random()
        train_rng.setstate(random.getstate())
    # 1. Instantiate the transforms
    train_transform = get_train_transform(
        time_length=time_length,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        rng=train_rng,
    )
    dev_transform = get_dev_transform(time_length=time_length)

    # 2. Create dataset instances (notice the corrected class name)
    train_ds = CremadPrecompDataset(
        root=CREMA_ROOT,
        split="train",
        train_transform=train_transform,
        dev_transform=None  # not used for train split
    )
    dev_ds = CremadPrecompDataset(
        root=CREMA_ROOT,
        split="dev",
        train_transform=None,
        dev_transform=dev_transform
    )
    test_ds = CremadPrecompDataset(
        root=CREMA_ROOT,
        split="test",
        train_transform=None,
        dev_transform=dev_transform
    )
    
    num_workers = 0 if torch.cuda.is_available() else 4
    pin_memory = True if torch.cuda.is_available() else False
    persistent_workers = True if num_workers > 0 else False
    # 3. Wrap each in a DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    dev_dl   = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    return train_dl, dev_dl, test_dl


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
