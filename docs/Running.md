# How to run this project?


This guide walks through downloading data, extracting features, training models,
and visualizing results.

When you ran main.py script, most of the choices are already written to you 
what you mostly need to do is to select the option from the terminal output. When
prompted with a question, for exmaple: 

```Do you wish to download the data? [y/n]:```

Write one of the responses in the terminal, for example "y" and press enter. \
This will bring you to the next question untill all the process is completed.


## Prerequisites 

Firstly, complete the environment setup and download all the required packages 
as mentioned in README.\
Ensure `gdown` is installed to download datasets

## 1. Download the data

Two dataset scopes are available:

### Full dataset (~600 mb)

```bash
python main.py
# When prompted:
# Do you wish to download the data? [y/n]: y
# Do you wish to download the FULL dataset or the CHECK data? [full/check]: full
```

### Small check subset (~250 mb)

```bash
python main.py
# When prompted:
# Do you wish to download the data? [y/n]: y
# Do you wish to download the FULL dataset or the CHECK data? [full/check]: check
```

oth options place audio under `data/unprocessed/` and extract any archives.

## 2. Extract features and generate labels

Run `main.py` again. If the data is present, it will automatically:

1. Extract MFCC features into `data/processed/`.
2. Create `data/labels.csv` if missing.

Manual commands are also available:

```bash
python scripts/utils/audio_features.py data/unprocessed/crema-d/AudioWAV --out data/processed
python scripts/utils/create_labels.py data/processed
```

## 3. Train a model

Start training via `main.py` and select one or more model variants:

```bash
python main.py
# Do you wish to train the model? [y/n]: y
# Which models do you wish to train? [CBAM, NO_CBAM, baseline]: cbam
```

You can also launch individual scripts directly:

```bash
python scripts/models/baseline/baseline.py
python scripts/models/cbam/cbam.py
python scripts/models/no_cbam/no_cbam.py
```

Do so with caution! 

## 4. Visualize logs and evaluate

After training completes, `main.py` can plot metrics and confusion matrices:

```bash
python main.py
# Do you want to visualize training logs? [y/n]: y
```

Saved figures appear under `reports/training_logs/<model>/`.

