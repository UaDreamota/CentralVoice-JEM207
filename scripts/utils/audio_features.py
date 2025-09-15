#utils/audio_features.py
"""
Script to extract audio features and all the attached helpers
"""

import argparse
import os
from pathlib import Path

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"

def load_audio(path: str, sr: int = 16000):
    """Load an audio file with the given sample rate."""
    signal, sample_rate = librosa.load(path, sr=sr)
    return signal, sample_rate


def extract_mfcc(signal: np.ndarray, sr: int = 16000, *, n_mfcc: int = 40,
                  n_fft: int = 512, hop_length: int = 256,
                  window: str = "hann") -> np.ndarray:
    """Compute MFCC features from a waveform.

    Pipeline:
      1) Power mel-spectrogram (power=2.0)
      2) Log scale (dB)
      3) DCT to obtain MFCCs

    Returns
    -------
    numpy.ndarray
        Array shaped (1, 1, n_mfcc, frames), dtype float32.
    """
    # 1) mel power spectrogram
    S = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        power=2.0,
        n_mels=max(n_mfcc, 40),  # ensure enough mel bands for the DCT
    )
    # 2) log scale (dB)
    S_db = librosa.power_to_db(S, ref=np.max)

    # 3) MFCCs from log-mel
    mfcc = librosa.feature.mfcc(
        S=S_db,
        n_mfcc=n_mfcc,
        dct_type=2,
        norm="ortho",
    )

    features = mfcc[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)
    assert features.shape[2] == n_mfcc, (
        f"Expected {n_mfcc} coefficients, got {features.shape[2]}"
    )
    return features


def collect_audio_files(path: str) -> list[str]:
    """Return a sorted list of WAV/MP3 files under ``path``.

    If ``path`` is itself a file, a single-item list is returned.
    """
    if os.path.isfile(path):
        return [path]

    files = []
    for entry in os.scandir(path):
        if entry.is_file() and entry.name.lower().endswith((".wav", ".mp3")):
            files.append(entry.path)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MFCC features")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default="../../data/unprocessed/crema-d/AudioWAV",
        help="Path to a WAV/MP3 file or a directory of audio files",
    )
    parser.add_argument(
        "--out",
        "-o",
        help=(
            "Output file (.npy) for a single input or directory to save multiple"
            " files. Defaults to saving into data/processed."
        ),
    )
    args = parser.parse_args()

    paths = collect_audio_files(args.audio_path)
    if not paths:
        parser.error(f"No audio files found at {args.audio_path}")

    save_dir = None
    single_out = None
    if args.out:
        if len(paths) > 1 or os.path.isdir(args.out):
            save_dir = args.out
            os.makedirs(save_dir, exist_ok=True)
        else:
            single_out = args.out

    for audio_path in paths:
        wav, sr = load_audio(audio_path, sr=16000)
        feats = extract_mfcc(wav, sr)

        if save_dir:
            out_path = os.path.join(save_dir, Path(audio_path).stem + ".npy")
            np.save(out_path, feats)
            print(f"Saved MFCCs to {out_path} with shape {feats.shape}")
        elif single_out:
            np.save(single_out, feats)
            print(f"Saved MFCCs to {single_out} with shape {feats.shape}")
        elif len(paths) > 1 or os.path.isdir(args.audio_path):
            processed_dir = Path("/data/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            out_path = processed_dir / f"{Path(audio_path).stem}.npy"
            np.save(out_path, feats)
            print(f"Saved MFCCs to {out_path} with shape {feats.shape}")
        else:
            print(feats)
            print("Shape:", feats.shape)


if __name__ == "__main__":
    main()
