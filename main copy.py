# main.py

# ─────────────────────────────────────────────────────────────
### IMPORTS
# ─────────────────────────────────────────────────────────────

import os
import subprocess
import sys
from pathlib import Path

from scripts.utils.data_downloader import download_data
import zipfile
import tarfile


FOLDER_URL_DATA = "https://drive.google.com/drive/folders/1rWylF6dUeP2D8k39gMPGEUY4yBhyXGee?usp=sharing"
FOLDER_URL_TEST = "https://drive.google.com/drive/folders/1-M4YQKUbNfAz-IZSGcSUguu7aqAa3AYf?usp=sharing"


REPO_ROOT = Path(__file__).resolve().parent
DATA_AUDIO_DIR = REPO_ROOT / "data" / "unprocessed" / "crema-d" / "AudioWAV"
LABEL_FILE = REPO_ROOT / "data" / "labels.csv"
UNPROCESSED_ROOT = REPO_ROOT / "data" / "unprocessed"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"


# ─────────────────────────────────────────────────────────────
### SMALL HELPERS
# ─────────────────────────────────────────────────────────────

def _run_script(script: str, *args: str) -> None:
    cmd = [sys.executable, script, *args]
    subprocess.run(cmd, check=True)


def ensure_features(source_dir: Path) -> None:
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    if any(p.suffix == ".npy" for p in PROCESSED_ROOT.iterdir()):
        print("Features already extracted. Skipping feature extraction.")
        return

    print("Extracting MFCC features…")
    script = str(REPO_ROOT / "scripts" / "utils" / "audio_features.py")
    _run_script(script, str(source_dir), "--out", str(PROCESSED_ROOT))


def ensure_labels(_: Path) -> None:
    if LABEL_FILE.exists():
        print("Label CSV already exists. Skipping label generation.")
        return

    print("Creating label CSV…")
    script = str(REPO_ROOT / "scripts" / "utils" / "create_labels.py")
    _run_script(script, str(PROCESSED_ROOT))


def extract_archives(directory: Path) -> None:
    for archive in directory.rglob("*"):
        if archive.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}:
            print(f"Extracting {archive}…")
            try:
                if archive.suffix.lower() == ".zip":
                    with zipfile.ZipFile(archive, "r") as zf:
                        zf.extractall(archive.parent)
                else:
                    with tarfile.open(archive, "r:*") as tf:
                        tf.extractall(archive.parent)
            except Exception as e:
                print(f"Failed to extract {archive}: {e}")


# ─────────────────────────────────────────────────────────────
### MODEL SELECTION
# ─────────────────────────────────────────────────────────────

def _resolve_model_script(human_choice: str) -> Path:
    """
    Map user's choice to an existing model script.

    Folders (as in your screenshot):
      - scripts/models/cbam/cbam.py
      - scripts/models/no_cbam/no_cbam.py  (also try .../baseline.py if that’s your filename)
      - scripts/models/baseline/baseline.py
    """
    choice = human_choice.strip().lower()

    if "cbam" in choice:
        candidates = [REPO_ROOT / "scripts" / "models" / "cbam" / "cbam.py"]
    elif "baseline" in choice:
        candidates = [REPO_ROOT / "scripts" / "models" / "baseline" / "baseline.py"]
    else:
        # default: plain CNN (no attention)
        candidates = [REPO_ROOT / "scripts" / "models" / "no_cbam" / "no_cbam.py"]

    for p in candidates:
        if p.exists():
            return p

    # If nothing matched, fail clearly with what we tried.
    tried = " | ".join(str(p.relative_to(REPO_ROOT)) for p in candidates)
    raise FileNotFoundError(f"Could not find a model script for '{human_choice}'. Tried: {tried}")

# ─────────────────────────────────────────────────────────────
### MULTI-SELECT PARSER (tiny helper)
# ─────────────────────────────────────────────────────────────
def _parse_model_list(text: str) -> list[str]:
    """
    Parse a human string like:
      "CNN+CBAM, baseline", "cbam and baseline", "baseline + cnn"
    → returns canonical ids in order without duplicates:
      ["cbam", "baseline"] or ["baseline", "no_cbam"], etc.
    """
    s = text.lower()
    for ch in [",", "+", "/", "|"]:
        s = s.replace(ch, " ")
    s = s.replace(" and ", " ")
    tokens = s.split()

    out: list[str] = []
    for t in tokens:
        if "cbam" in t and "cbam" not in out:
            out.append("cbam")
        elif "base" in t and "baseline" not in out:
            out.append("baseline")
        elif ("cnn" in t or "no_cbam" in t or "no-cbam" in t or "nocbam" in t) and "no_cbam" not in out:
            out.append("no_cbam")

    return out or ["no_cbam"]  # sensible default


def train_model(model_script: Path) -> None:
    print(f"Training model via: {model_script.relative_to(REPO_ROOT)}")
    _run_script(str(model_script))


# ─────────────────────────────────────────────────────────────
### MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    # 1) Data presence / optional download
    if not DATA_AUDIO_DIR.exists() or not any(DATA_AUDIO_DIR.iterdir()):
        print("Wait! Data is missing.")
        download_question = (
            input("Do you wish to download the data? [y/n]: ").strip().lower()
        )
        if download_question == "y":
            test_data_question = (
                input(
                    "Do you wish to download the full dataset or the test? [full/test]: "
                )
                .strip()
                .lower()
            )
            if test_data_question == "test":
                print("Creating 50 file batches. This may take a while...")
                download_data(FOLDER_URL_TEST, str(UNPROCESSED_ROOT))
                extract_archives(UNPROCESSED_ROOT)
            elif test_data_question == "full":
                print("Creating 50 file batches. This may take a while...")
                download_data(FOLDER_URL_DATA, UNPROCESSED_ROOT)
        else:
            print("You may now only run the inference script, as the data is missing.")
            return
    else:
        print("Data already exists. Skipping download.")

    if not DATA_AUDIO_DIR.exists():
        print("ERROR: expected audio directory", DATA_AUDIO_DIR)
        return

    # 2) Features + labels
    ensure_features(DATA_AUDIO_DIR)
    ensure_labels(PROCESSED_ROOT)

    # 3) Train?
    training_question = input("Do you wish to train the model? [y/n]: ").strip().lower()
    if training_question == "y":
        model_choice = input("Which models do you wish to train? [CNN+CBAM, CNN, baseline]: ").strip()
        models = _parse_model_list(model_choice)
        print("Selected models (in order):", ", ".join(models))

        for m in models:
            try:
                model_script = _resolve_model_script(m)
            except FileNotFoundError as e:
                print(e)
                print(f"Skipping '{m}'.")
                continue
            train_model(model_script)
    else:
        print("Training model terminated.")



if __name__ == "__main__":
    main()
