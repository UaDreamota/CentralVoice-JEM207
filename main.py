
import os
import subprocess
import sys
from pathlib import Path

from scripts.utils.data_downloader import download_data
import zipfile
import tarfile



FOLDER_URL_DATA = (
    "https://drive.google.com/drive/folders/1rWylF6dUeP2D8k39gMPGEUY4yBhyXGee?usp=sharing"
)
FOLDER_URL_TEST = (
    "https://drive.google.com/drive/folders/1-M4YQKUbNfAz-IZSGcSUguu7aqAa3AYf?usp=sharing"
)



REPO_ROOT = Path(__file__).resolve().parent
DATA_AUDIO_DIR = REPO_ROOT / "data" / "unprocessed" / "crema-d" / "AudioWAV"
LABEL_FILE = REPO_ROOT / "data" / "labels.csv"
UNPROCESSED_ROOT = REPO_ROOT / "data" / "unprocessed"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"


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


def ensure_labels(audio_dir: Path) -> None:
    if LABEL_FILE.exists():
        print("Label CSV already exists. Skipping label generation.")
        return

    print("Creating label CSV…")
    script = str(REPO_ROOT / "scripts" / "utils" / "create_labels.py")
    _run_script(script, str(audio_dir))

def train_model() -> None:
    print("Training model...")
    script = str(REPO_ROOT / "scripts" / "models" /  "model_baseline.py")
    _run_script(script)


def extract_archives(directory: Path) -> None:
    for archive in directory.rglob('*'):
        if archive.suffix.lower() in {'.zip', '.tar', '.gz', '.tgz'}:
            print(f"Extracting {archive}…")
            try:
                if archive.suffix.lower() == '.zip':
                    with zipfile.ZipFile(archive, 'r') as zf:
                        zf.extractall(archive.parent)
                else:
                    with tarfile.open(archive, 'r:*') as tf:
                        tf.extractall(archive.parent)
            except Exception as e:
                print(f"Failed to extract {archive}: {e}")

def main() -> None:
    if not DATA_AUDIO_DIR.exists() or not any(DATA_AUDIO_DIR.iterdir()):
        print("Wait! Data is missing.")
        download_question = input("Do you wish to download the data? [y/n]: ").strip().lower()
        if download_question == "y":
            test_data_question = input(
                "Do you wish to download the full dataset or the test? [full/test]: "
            ).strip().lower()
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

    ensure_features(DATA_AUDIO_DIR)
    ensure_labels(PROCESSED_ROOT)

    training_question = input("Do you wish to train the model? [y/n]: ").strip().lower()
    if training_question == "y":
        print("Training model...")
        train_model()
    else:
        print("Training model terminated.")

if __name__ == "__main__":
    main()