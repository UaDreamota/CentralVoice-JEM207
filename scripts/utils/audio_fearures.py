# Utilities for loading audio and extracting MFCC features.

import argparse
import numpy as np
import librosa


def load_audio(path: str, sr: int = 16000):
    signal, sample_rate = librosa.load(path, sr=sr)
    return signal, sample_rate


def extract_mfcc(signal: np.ndarray, sr: int = 16000, *, n_mfcc: int = 40,
                  n_fft: int = 512, hop_length: int = 256,
                  window: str = "hann") -> np.ndarray:


    #Returns an array shaped (1, 1, coeff, frames) suitable for PyTorch models.

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
    )
    mfcc = librosa.power_to_db(mfcc, ref=np.max)

    features = mfcc[np.newaxis, np.newaxis, :, :]
    assert features.shape[2] == n_mfcc, (
        f"Expected {n_mfcc} coefficients, got {features.shape[2]}")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MFCC features")
    parser.add_argument("audio_path", help="Path to a WAV or MP3 file")
    parser.add_argument(
        "--out",
        "-o",
        help="Optional path to save features as .npy. If omitted, prints the shape",
    )
    args = parser.parse_args()

    wav, sr = load_audio(args.audio_path, sr=16000)
    feats = extract_mfcc(wav, sr)

    if args.out:
        np.save(args.out, feats)
        print(f"Saved MFCCs to {args.out} with shape {feats.shape}")
    else:
        print(feats)
        print("Shape:", feats.shape)


if __name__ == "__main__":
    main()