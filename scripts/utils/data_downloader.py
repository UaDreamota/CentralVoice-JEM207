import gdown

def download_data(folder_url, output_dir):

    print(f"Downloading data from {folder_url} to {output_dir}...")

    gdown.download_folder(url=folder_url, output=output_dir, quiet=False)

    print("Download complete.")


