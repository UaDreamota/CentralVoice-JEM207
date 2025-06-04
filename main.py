import os
import subprocess
import sys
from pathlib import Path

from scripts.utils.data_downloader import download_data




FOLDER_URL_DATA = (
    "https://drive.google.com/drive/folders/1rWylF6dUeP2D8k39gMPGEUY4yBhyXGee?usp=sharing"
)
FOLDER_URL_TEST = (
    "https://drive.google.com/drive/folders/1-M4YQKUbNfAz-IZSGcSUguu7aqAa3AYf?usp=sharing"
)
OUTPUT_DIR = "data/unprocessed"

REPO_ROOT = Path(__file__).resolve().parent
DATA_AUDIO_DIR = REPO_ROOT / "data" / "unprocessed" / "crema-d" / "AudioWAV"
LABEL_FILE = REPO_ROOT / "data" / "labels.csv"


def _run_script(script: str, *args: str) -> None:
    cmd = [sys.executable, script, *args]
    subprocess.run(cmd, check=True)


def ensure_features(audio_dir: Path) -> None:
    if any(p.suffix == ".npy" for p in audio_dir.iterdir()):
        print("Features already extracted. Skipping feature extraction.")
        return

    print("Extracting MFCC features…")
    script = str(REPO_ROOT / "scripts" / "utils" / "audio_features.py")
    _run_script(script, str(audio_dir))


def ensure_labels(audio_dir: Path) -> None:
    if LABEL_FILE.exists():
        print("Label CSV already exists. Skipping label generation.")
        return

    print("Creating label CSV…")
    script = str(REPO_ROOT / "scripts" / "utils" / "create_labels.py")
    _run_script(script, str(audio_dir))

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
                download_data(FOLDER_URL_TEST, OUTPUT_DIR)
            elif test_data_question == "full":
                print("Creating 50 file batches. This may take a while...")
                download_data(FOLDER_URL_DATA, OUTPUT_DIR)
        else:
            print("You may now only run the inference script, as the data is missing.")
            return
    else:
        print("Data already exists. Skipping download.")

    if not DATA_AUDIO_DIR.exists():
        print("ERROR: expected audio directory", DATA_AUDIO_DIR)
        return

    ensure_features(DATA_AUDIO_DIR)
    ensure_labels(DATA_AUDIO_DIR)




if __name__ == "__main__":
    main()