import os
import sys
import json
import subprocess
import gdown
from typing import List, Dict


FOLDER_LINK = "https://drive.google.com/drive/folders/1rWylF6dUeP2D8k39gMPGEUY4yBhyXGee?usp=sharing"
OUTPUT_BASE = "data/unprocessed"



def batch_maker(folder_url: str) -> List[Dict]:
    """
    Runs 'python -m gdown.cli --dump-json <folder_url>' to get a complete
    listing (no 50-file limit) of every item in that Drive folder.
    """
    cmd = [
        sys.executable,
        "-m", "gdown.cli",
        "--dump-json",
        folder_url
    ]

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print("Error while running gdown command:\n", e.stderr, file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        print("ERROR: invalid JSON from gdown:", e, file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print("ERROR: expected a JSON list from gdown, got:", type(data), file=sys.stderr)
        sys.exit(1)

    return data


def download_all_files(file_entries: List[Dict], output_dir: str):
    """
    Downloads each non-folder entry from file_entries one by one using gdown.download().
    """
    for entry in file_entries:
        if entry.get("mimeType") == "application/vnd.google-apps.folder":
            continue

        rel_path = entry.get("path", entry["name"])
        local_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        file_id = entry["id"]
        file_url = f"https://drive.google.com/uc?id={file_id}"

        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            print(f"[SKIP] {local_path} already exists")
            continue

        print(f"Downloading '{rel_path}' …")
        try:
            gdown.download(url=file_url, output=local_path, quiet=False)
        except Exception as ex:
            print(f"ERROR downloading {rel_path}: {ex}", file=sys.stderr)

def download_data(folder_url, output_dir):

    print(f"Downloading data from {folder_url} to {output_dir}...")

    gdown.download_folder(url=folder_url, output=output_dir, quiet=False)

    print("Download complete.")


def main():
    # 1) Ask gdown CLI to list out *all* items (files + folders) in that Drive folder.
    print(f"Retrieving file list from {FOLDER_LINK!r} …")
    entries = batch_maker(FOLDER_LINK)

    # 2) Filter / confirm we got something
    if not entries:
        print("ERROR: gdown returned an empty list. Check your folder URL/ID.", file=sys.stderr)
        sys.exit(1)

    # 3) Download every non‐folder entry
    print(f"Found {len(entries)} items. Starting downloads…\n")
    download_all_files(entries, OUTPUT_BASE)

    print("\n✅ All done. Check", os.path.abspath(OUTPUT_BASE))


if __name__ == "__main__":
    main()


