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
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchaudio.transforms import FrequencyMasking, TimeMasking

# Repository root and dataset location
REPO_ROOT = Path(__file__).resolve().parents[2]
CREMA_ROOT = REPO_ROOT / "data" / "processed"

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
        elif self.split in {"dev", "val"} and self.dev_transform:
            feats = self.dev_transform(feats_np)

        if feats.ndim != 4:
            raise ValueError(
                "Each '.npy' must contain an array shaped (1, 1, C, T); "
                f"got {feats_np.shape} from '{npy_path}'."
            )

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
    Transformation for training that expects input of shape (1, C, H_var, T_var)
    and outputs a tensor of shape (1, C, 40, 218), with SpecAugment‐style
    frequency & time masking.

    Parameters:
    -----------
    time_length : int
        Desired time‐frame length (218).

    freq_mask_param : int
        Maximum number of consecutive frequency bins to mask (e.g., 15). 
        Used by torchaudio.transforms.FrequencyMasking.

    time_mask_param : int
        Maximum number of consecutive time frames to mask (e.g., 25). 
        Used by torchaudio.transforms.TimeMasking.
    """
    def __init__(self, time_length: int = 218,
                 freq_mask_param: int = 15,
                 time_mask_param: int = 25):
        self.time_length = time_length
        # FrequencyMasking will zero out up to freq_mask_param consecutive rows
        self.freq_mask = FrequencyMasking(freq_mask_param=freq_mask_param)
        # TimeMasking will zero out up to time_mask_param consecutive columns
        self.time_mask = TimeMasking(time_mask_param=time_mask_param)

    def __call__(self, mfcc):
        """
        Apply padding/truncation, normalization, and masking.

        Parameters:
        -----------
        mfcc : torch.Tensor or numpy.ndarray
            Input MFCC tensor or array of shape (1, C, H_var, T_var).
            - 1: “batch” dimension (always 1 here)
            - C: number of channels (often 1 for a single MFCC “image”)
            - H_var: number of MFCC coefficients (should be 40)
            - T_var: variable time‐frames (to be padded/truncated to 218)

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (1, C, 40, 218), dtype=torch.float,
            normalized (zero mean, unit variance) and with random freq/time
            masks applied (training‐only augmentation).
        """
        # 1. Ensure torch.Tensor type, cast to float if coming from NumPy
        if not isinstance(mfcc, torch.Tensor):
            mfcc = torch.tensor(mfcc, dtype=torch.float)

        # 2. Confirm we have exactly 4 dimensions
        #    If someone accidentally passed shape (C, H, T), this will raise an error.
        if mfcc.ndim != 4:
            raise ValueError(f"Expected 4D tensor (1, C, H_var, T_var), got shape {mfcc.shape}.")

        # 3. Extract dimensions
        #    B = batch size (should be 1)
        #    C = channels (often 1 for MFCC)
        #    H = height (number of MFCC bins, should be 40)
        #    T = original time‐frames (variable)
        B, C, H, T = mfcc.shape

        # 4. If H != 40, you may decide to raise an error (since H must be 40 here)
        if H != 40:
            raise ValueError(f"Expected H=40 (MFCC bins), but got H={H}.")

        # 5. Pad or truncate along the time axis (dimension index -1)
        if T < self.time_length:
            pad_amount = self.time_length - T
            # F.pad with a 2‐element tuple (pad_left, pad_right) pads only last dimension.
            # Here, pad=(0, pad_amount) → adds pad_amount zeros on the right side of the W dimension.
            mfcc = F.pad(mfcc, (0, pad_amount))
            # After padding, new shape is (1, C, 40, 218)
        elif T > self.time_length:
            # Truncate to the first `time_length` frames
            mfcc = mfcc[..., :self.time_length]
            # After truncation, shape is (1, C, 40, 218)

        # 6. Normalize per sample: compute mean and std over all values in the 4D tensor
        #    (batch=1, so effectively over that single sample).
        mean = mfcc.mean()
        std = mfcc.std()
        mfcc = (mfcc - mean) / (std + 1e-9)  # add epsilon to avoid division by zero

        # 7. Apply SpecAugment‐style masks on each (H, W) slice for every channel.
        #    FrequencyMasking and TimeMasking from torchaudio expect a 2D or 3D
        #    spectrogram: (freq, time) or (batch, freq, time). They do not natively
        #    operate on a 4D tensor. Thus, we iterate over batch and channels.
        for b in range(B):
            for c in range(C):
                # Extract the (H, W) slice: shape (40, 218)
                slice_2d = mfcc[b, c, :, :]
                # 7a. Frequency masking: zero out up to freq_mask_param consecutive rows
                slice_2d = self.freq_mask(slice_2d)
                # 7b. Time masking: zero out up to time_mask_param consecutive columns
                slice_2d = self.time_mask(slice_2d)
                # Write back the masked slice
                mfcc[b, c, :, :] = slice_2d

        return mfcc

# ------------- Dev transformation -------------------------------------------
class DevTransform:
    """
    Transformation for validation/dev that expects input of shape (1, C, H_var, T_var)
    and outputs a tensor of shape (1, C, 40, 218), with only padding/truncation
    and normalization—no speculative masking.

    Parameters:
    -----------
    time_length : int
        Desired time‐frame length (218).
    """
    def __init__(self, time_length: int = 218):
        self.time_length = time_length

    def __call__(self, mfcc):
        """
        Apply padding/truncation and normalization (no augmentation).

        Parameters:
        -----------
        mfcc : torch.Tensor or numpy.ndarray
            Input MFCC of shape (1, C, H_var, T_var).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (1, C, 40, 218), dtype=torch.float,
            normalized (zero‐mean, unit‐variance).
        """
        # 1. Ensure torch.Tensor type
        if not isinstance(mfcc, torch.Tensor):
            mfcc = torch.tensor(mfcc, dtype=torch.float)

        # 2. Confirm 4D shape
        if mfcc.ndim != 4:
            raise ValueError(f"Expected 4D tensor (1, C, H_var, T_var), got shape {mfcc.shape}.")

        B, C, H, T = mfcc.shape
        if H != 40:
            raise ValueError(f"Expected H=40 (MFCC bins), but got H={H}.")

        # 3. Pad or truncate along the time axis
        if T < self.time_length:
            pad_amount = self.time_length - T
            mfcc = F.pad(mfcc, (0, pad_amount))
        elif T > self.time_length:
            mfcc = mfcc[..., :self.time_length]

        # 4. Normalize per sample
        mean = mfcc.mean()
        std = mfcc.std()
        mfcc = (mfcc - mean) / (std + 1e-9)

        return mfcc
    
# ------------- Train transformation wrapper -------------------------------------------

def get_train_transform(time_length: int = 218,
                        freq_mask_param: int = 15,
                        time_mask_param: int = 25):
    """
    Returns an instance of TrainTransform with the specified hyperparameters.

    Args:
    -----
    time_length : int
        Fixed number of time‐frames after padding/truncation (default: 218).
    freq_mask_param : int
        Maximum width of frequency‐axis mask (default: 15).
    time_mask_param : int
        Maximum width of time‐axis mask (default: 25).

    Returns:
    --------
    TrainTransform
    """
    return TrainTransform(
        time_length=time_length,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param
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
    time_mask_param: int = 25
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, dev, and test dataloaders for CREMA-D using
    CremadPrecompDataset. Pass the appropriate transforms into each.
    """
    # 1. Instantiate the transforms
    train_transform = get_train_transform(
        time_length=time_length,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param
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

    # 3. Wrap each in a DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    dev_dl   = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

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
