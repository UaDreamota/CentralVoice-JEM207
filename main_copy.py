# main.py

# ─────────────────────────────────────────────────────────────
### IMPORTS
# ─────────────────────────────────────────────────────────────

import os
import subprocess
import sys
from pathlib import Path

from scripts.utils.data_downloader import download_data
from scripts.utils.eval_pred import evaluate_predictions
from scripts.utils.visualizations import plot_class_distribution, plot_training_history, plot_confusion_matrix
import zipfile
import tarfile


FOLDER_URL_DATA = "https://drive.google.com/drive/folders/1rWylF6dUeP2D8k39gMPGEUY4yBhyXGee?usp=sharing"
FOLDER_URL_TEST = "https://drive.google.com/drive/folders/1-M4YQKUbNfAz-IZSGcSUguu7aqAa3AYf?usp=sharing"


REPO_ROOT = Path(__file__).resolve().parent
DATA_AUDIO_DIR = REPO_ROOT / "data" / "unprocessed" / "crema-d" / "AudioWAV"
LABEL_FILE = REPO_ROOT / "data" / "labels.csv"
UNPROCESSED_ROOT = REPO_ROOT / "data" / "unprocessed"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
SCOPE_MARKER = UNPROCESSED_ROOT / "crema-d" / ".download_scope"  


# ─────────────────────────────────────────────────────────────
### SMALL HELPERS
# ─────────────────────────────────────────────────────────────

def _run_script(script: str, *args: str) -> None:
    """
    Run a Python script in a subprocess using the current interpreter.

    Parameters
    ----------
    script : str
        Absolute or repository-relative path to a Python script file.
    *args : str
        Additional command-line arguments passed to the script.

    Raises
    ------
    subprocess.CalledProcessError
        If the invoked script exits with a non-zero status.

    Notes
    -----
    - Uses `sys.executable` to ensure the same interpreter/environment.
    - Stdout/stderr are not captured; they stream directly to this process.
    """
    cmd = [sys.executable, script, *args]
    subprocess.run(cmd, check=True)


def _count_audio_wavs() -> int:
    """
    Count .wav files recursively under the expected audio directory.

    Returns
    -------
    int
        Number of files ending with `.wav` found under `DATA_AUDIO_DIR`.

    Notes
    -----
    Returns 0 if `DATA_AUDIO_DIR` does not exist.
    """
    if not DATA_AUDIO_DIR.exists():
        return 0
    return sum(1 for _ in DATA_AUDIO_DIR.rglob("*.wav"))


def _write_scope_marker(scope: str) -> None:
    """
    Write a scope marker file indicating which subset is present.

    Parameters
    ----------
    scope : str
        Data scope label to persist, typically ``'full'`` or ``'test'``.

    Notes
    -----
    The marker is written to ``SCOPE_MARKER`` and used to infer whether
    the repository contains the full dataset or a test subset on future runs.
    Any I/O errors are caught and reported as warnings, without raising.
    """
    try:
        SCOPE_MARKER.parent.mkdir(parents=True, exist_ok=True)
        SCOPE_MARKER.write_text(scope.strip())
    except Exception as e:
        print(f"Warning: could not write scope marker: {e}")


def _read_scope_marker() -> str | None:
    """
    Read the persisted data scope marker.

    Returns
    -------
    str or None
        The normalized scope string (``'full'`` or ``'test'``) if the marker
        exists and is readable, otherwise ``None``.

    Notes
    -----
    Any I/O errors are suppressed and result in ``None``.
    """
    try:
        if SCOPE_MARKER.exists():
            return SCOPE_MARKER.read_text().strip().lower()
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────
### EXCTRACTION (zip, features, labels)
# ─────────────────────────────────────────────────────────────
def extract_archives(directory: Path) -> None:
    """
    Extract supported archives found under a directory tree.

    Parameters
    ----------
    directory : pathlib.Path
        Root directory to scan recursively for archives.

    Notes
    -----
    - Supported: `.zip` and tarballs (`.tar`, `.tar.gz`, `.tgz`).
    - Archives are extracted in-place to their parent directory. Errors are
      caught and logged.
    """
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


def ensure_features(source_dir: Path) -> None:
    """
    Ensure MFCC features are extracted for the available audio.

    Parameters
    ----------
    source_dir : pathlib.Path
        Directory containing raw ``.wav`` audio files.

    Notes
    -----
    - If the number of existing ``.npy`` feature files in ``PROCESSED_ROOT``
      is greater than or equal to the number of ``.wav`` files, extraction
      is skipped.
    - Otherwise, runs ``scripts/utils/audio_features.py`` to generate features
      into ``PROCESSED_ROOT``.
    """
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    n_feats = sum(1 for _ in PROCESSED_ROOT.glob("*.npy"))
    n_wavs = _count_audio_wavs()

    if n_feats > 0 and n_wavs > 0 and n_feats >= n_wavs:
        print("Features already extracted. Skipping feature extraction.")
        return

    print("Extracting MFCC features…")
    script = str(REPO_ROOT / "scripts" / "utils" / "audio_features.py")
    _run_script(script, str(source_dir), "--out", str(PROCESSED_ROOT))

def ensure_labels(_: Path) -> None:
    """
    Ensure the labels CSV is present.

    Parameters
    ----------
    _ : pathlib.Path
        Unused. Kept for a symmetric signature with feature extraction.

    Notes
    -----
    If ``LABEL_FILE`` already exists, label generation is skipped.
    Otherwise, runs ``scripts/utils/create_labels.py`` to build it based on
    processed features.
    """
    if LABEL_FILE.exists():
        print("Label CSV already exists. Skipping label generation.")
        return

    print("Creating label CSV…")
    script = str(REPO_ROOT / "scripts" / "utils" / "create_labels.py")
    _run_script(script, str(PROCESSED_ROOT))

# ─────────────────────────────────────────────────────────────
### MODEL SELECTION (by user) AND TRAINING
# ─────────────────────────────────────────────────────────────
def _resolve_model_script(human_choice: str) -> Path:
    """
    Resolve a human-friendly model choice to a concrete script path.

    Parameters
    ----------
    human_choice : str
        User input describing the model, e.g. ``'cbam'``, ``'baseline'``,
        or ``'cnn'``.

    Returns
    -------
    pathlib.Path
        Absolute path to the selected model script.

    Raises
    ------
    FileNotFoundError
        If no matching model script is found.

    Notes
    --------
    Folders:
      - scripts/models/cbam/cbam.py
      - scripts/models/no_cbam/no_cbam.py
      - scripts/models/baseline/baseline.py
    """
    choice = human_choice.strip().lower()

    if "cbam" in choice:
        candidates = [REPO_ROOT / "scripts" / "models" / "cbam" / "cbam.py"]
    elif "baseline" in choice:
        candidates = [REPO_ROOT / "scripts" / "models" / "baseline" / "baseline.py"]
    else:
        # default: plain CNN (no attention)
        candidates = [
            REPO_ROOT / "scripts" / "models" / "no_cbam" / "no_cbam.py",
            REPO_ROOT / "scripts" / "models" / "no_cbam" / "baseline.py",  # alternate filename
        ]

    for p in candidates:
        if p.exists():
            return p

    tried = " | ".join(str(p.relative_to(REPO_ROOT)) for p in candidates)
    raise FileNotFoundError(f"Could not find a model script for '{human_choice}'. Tried: {tried}")

def _parse_model_list(text: str) -> list[str]:
    """
    Parse a free-form string of model choices into canonical identifiers.

    Parameters
    ----------
    text : str
        A string like ``"CNN+CBAM, baseline"`` or ``"cbam and baseline"``.

    Returns
    -------
    list of str
        Ordered, de-duplicated canonical model ids, e.g.
        ``['cbam', 'baseline']`` or ``['baseline', 'no_cbam']``.
        Defaults to ``['no_cbam']`` if nothing matched.

    Notes
    -----
    Separators such as commas, plus, slash, pipe, and the word ``and`` are
    normalized to whitespace before tokenization.
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

    return out or ["no_cbam"]

def train_model(model_script: Path) -> None:
    """
    Launch model training by invoking the selected model script.

    Parameters
    ----------
    model_script : pathlib.Path
        Absolute path to the Python script that implements the model training.

    Notes
    -----
    This function delegates execution to `_run_script`. Any non-zero exit
    code is caught and summarized to stderr/stdout for easier diagnosis.
    """
    print(f"Training model via: {model_script.relative_to(REPO_ROOT)}")
    try:
        _run_script(str(model_script))
    except subprocess.CalledProcessError as e:
        print(
            f"Training failed for {model_script.relative_to(REPO_ROOT)} "
            f"(exit code {e.returncode})"
        )
        if e.stdout:
            print("--- stdout ---")
            print(e.stdout)
        if e.stderr:
            print("--- stderr ---")
            print(e.stderr)


# ─────────────────────────────────────────────────────────────
### MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate the end-to-end workflow: data, features, training, and reports.

    Steps
    -----
    1. Check for presence of audio data. Optionally download either the
       full dataset or a small test subset, and record the choice in a
       scope marker file.
    2. Extract archives (if any), generate MFCC features, and ensure labels.
    3. Optionally visualize class imbalance.
    4. Optionally train one or more model variants selected by the user.
    5. Optionally visualize training logs (loss, accuracy, macro-F1).

    Notes
    -----
    This function is interactive and prompts the user for choices at
    multiple steps. It writes artifacts under the repository’s `data/`,
    `reports/`, and model-specific output folders.
    """

    # ─────────────────────────────────────────────────────────────
    # 1) Data downloading (tracking the data scope) 
    # ─────────────────────────────────────────────────────────────
    data_scope = None  # 'full' | 'test' | 'existing'

    if not DATA_AUDIO_DIR.exists() or not any(DATA_AUDIO_DIR.iterdir()):
        print("Wait! Data is missing.")
        download_question = (
            input("Do you wish to download the data? [y/n]: ").strip().lower()
        )
        if download_question == "y":
            test_data_question = (
                input("Do you wish to download the full dataset or the test? [full/test]: ")
                .strip()
                .lower()
            )
            if test_data_question == "test":
                print("Creating 50 file batches. This may take a while...")
                download_data(FOLDER_URL_TEST, str(UNPROCESSED_ROOT))
                extract_archives(UNPROCESSED_ROOT)
                _write_scope_marker("test")
                data_scope = "test"
            elif test_data_question == "full":
                print("Creating 50 file batches. This may take a while...")
                download_data(FOLDER_URL_DATA, UNPROCESSED_ROOT)
                extract_archives(UNPROCESSED_ROOT)
                _write_scope_marker("full")
                data_scope = "full"
            else:
                print("Unknown choice. Exiting.")
                return
        else:
            print("You may now only run the inference script, as the data is missing.")
            return
    else:
        # Data exists; try to identify scope
        scope = _read_scope_marker()
        if scope in {"full", "test"}:
            data_scope = scope
            print(f"Data already exists. Skipping download. (scope: {data_scope})")
        else:
            n_wavs = _count_audio_wavs()
            print(f"Data already exists. Skipping download. Detected {n_wavs} .wav files.")
            ask = input("Is this the FULL dataset? [y/n]: ").strip().lower()
            data_scope = "full" if ask == "y" else "test"
            _write_scope_marker(data_scope)

    if not DATA_AUDIO_DIR.exists():
        print("ERROR: expected audio directory", DATA_AUDIO_DIR)
        return

    # ─────────────────────────────────────────────────────────────
    # 2) Extracting MFCC features and labels 
    # ─────────────────────────────────────────────────────────────
    ensure_features(DATA_AUDIO_DIR)
    ensure_labels(PROCESSED_ROOT)

    # ─────────────────────────────────────────────────────────────
    # 3) Class-imbalance visualization 
    # ─────────────────────────────────────────────────────────────
    class_balance_question = input("Do you want to visualize class imbalance now? [y/n]: ").strip().lower()
    if class_balance_question == "y":
        outdir = REPO_ROOT / "reports" / "data_overview"
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            arts = plot_class_distribution(
                LABEL_FILE,
                outdir,
                label="emotion",        # change if your CSV uses another column name
                split="split",          # change if your CSV uses another split column
                title_prefix=f"Class distribution of emotions",
            )
            saved = [str(p) for p in arts.values()]
            print("Saved class distribution plots:")
            for p in saved:
                print(" -", p)
        except Exception as e:
            print(f"[visualization] Skipped: {e}")


    # ─────────────────────────────────────────────────────────────
    # 4) Training and logs visualization
    # ─────────────────────────────────────────────────────────────
    
    training_question = input("Do you wish to train the model? [y/n]: ").strip().lower()
    if training_question == "y":
        if data_scope == "test":
            print(
                "Training is disabled when only the TEST subset is present.\n"
                "Please download the FULL dataset to enable training (re-run and choose 'full')."
            )
            return

        model_choice = input("Which models do you wish to train? [CNN+CBAM, CNN, baseline]: ").strip().lower()
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
        
        results_question = input('Do you want to visualize training logs? [y/n]: ').strip().lower()
        if results_question == 'y':
            for m in models:
                OUT_DIR = REPO_ROOT / 'reports' / 'training_logs' / m 
                HISTORY_CSV = REPO_ROOT / 'scripts' / 'models' / m / 'logs' / 'history.csv'
                PREDICTIONS_CSV = REPO_ROOT / 'scripts' / 'models' / m / 'logs' / 'predictions.csv'

                # Ensure output dir exists even if history plotting fails
                OUT_DIR.mkdir(parents=True, exist_ok=True)

                try:
                    plot_training_history(history_csv=HISTORY_CSV, outdir=OUT_DIR, title_prefix=m)
                except Exception as exc:
                    print(f"[visualization] Skipped history for '{m}' – {exc}")
                try:
                    # Evaluate this model's predictions and plot confusion matrix
                    _, _, predictions, labels = evaluate_predictions(PREDICTIONS_CSV)
                    plot_confusion_matrix(
                        OUT_DIR,
                        predictions=predictions,
                        labels=labels,
                        model_name=m,
                    )
                except Exception as exc:
                    print(f"[visualization] Skipped confusion matrix for '{m}' – {exc}")
            print(
                'Displaying the training logs for training and validation sets.\n'
                'The following quantities are plotted: Loss, Accuracy, Macro F1 score.'
            )
        else:
            print('Training logs are not visualized.')
    else:
        print("Training model terminated.")
        view_existing = input("Do you want to view existing training log visualizations? [y/n]: ").strip().lower()
        if view_existing == "y":
            model_choice = input("Which model reports do you want to view? [CNN+CBAM, CNN, baseline]: ").strip().lower()
            models_to_view = _parse_model_list(model_choice)

            for m in models_to_view:
                png_paths: list[Path] = []
                # Check both possible locations
                for d in [
                    REPO_ROOT / "reports" / m,
                    REPO_ROOT / "reports" / "training_logs" / m,
                ]:
                    if d.exists():
                        png_paths.extend(sorted(d.glob("*.png")))

                if png_paths:
                    print(f"Found {len(png_paths)} PNG(s) for '{m}':")
                    for p in png_paths:
                        print(" -", p)
                        try:
                            os.startfile(p)  # Windows: open with default viewer
                        except Exception as exc:
                            print(f"   Could not open {p}: {exc}")
                else:
                    print(f"No report images found for '{m}'. Please train the model first.")
#


if __name__ == "__main__":
    main()
