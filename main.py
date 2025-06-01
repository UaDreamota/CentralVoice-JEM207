
import os
from scripts.utils.data_downloader import download_data


folder_url_data = 'https://drive.google.com/drive/folders/1rWylF6dUeP2D8k39gMPGEUY4yBhyXGee?usp=sharing' # Dataset that contains the actual data (70+gb)
folder_url_test = 'https://drive.google.com/drive/folders/1-M4YQKUbNfAz-IZSGcSUguu7aqAa3AYf?usp=sharing' # Dataset that only contains a few fies for testing
output_dir = 'data/unprocessed'

def main():

    if not os.path.exists('data/unprocessed') or len(os.listdir('data/unprocessed')) == 1:
        print('Wait! Data is missing.')
        download_question = input('Do you wish to download the data? [y/n]: ').strip().lower()
        if download_question == 'y':
            test_data_question = input('Do you wish to download the full dataset or the test? [full/test]: ').strip().lower()
            if test_data_question == 'test':
                download_data(folder_url=folder_url_test, output_dir=output_dir)
            elif test_data_question != 'full':
                download_data(folder_url=folder_url_data, output_dir=output_dir)
        else:
            print('You may now only run the inference script, as the data is missing.')
            return


    else:
        print('Data already exists. Skipping download.')




if __name__ == "__main__":
    main()
