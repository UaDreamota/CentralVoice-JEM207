# How to run this project?


This guide walks through downloading data, extracting features, training models,
and visualizing results.

When you run main.py script, most of the choices are already written to you as prompts in the terminal.
When prompted with a question, for exmaple: 

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

As of right now full dataset only consists of 7000 thousand file of CREMA-D
dataset.

```bash
python main.py
# When prompted:
# Do you wish to download the data? [y/n]: y
# Do you wish to download the FULL dataset or the CHECK data? [full/check]: full
```

### Small check subset (~30 mb)

It is made only for testing the whole script. It may be quite exauhsting for some 
computers to train three models with relatively big dataset. 

We recommend using check subset to test the script. 

```bash
python main.py
# When prompted:
# Do you wish to download the data? [y/n]: y
# Do you wish to download the FULL dataset or the CHECK data? [full/check]: check
```

Both options place audio under `data/unprocessed/` and extract any archives.

## 2. Extract features and generate labels

Run `main.py` again. If the data is present, it will automatically:

1. Extract MFCC features into `data/processed/`.
2. Create `data/labels.csv` if missing.

Manual commands are also available:

```bash
python scripts/utils/audio_features.py data/unprocessed/crema-d/AudioWAV --out data/processed
python scripts/utils/create_labels.py data/processed
```

After feature extraction, `main.py` prompts to visualize class imbalance:

```bash
python main.py
# When prompted:
# Do you want to visualize class imbalance now? [y/n]: y
# Saved class distribution plots:
#  - D:\Projects\GitRepositories\CentralVoice-JEM207\reports\data_overview\00_class_distribution_overall.png
#  - D:\Projects\GitRepositories\CentralVoice-JEM207\reports\data_overview\01_class_distribution_by_split.png

```

You will be prompted to vizualize the class imabalance. If you choose "y", 
new figure will be saved under `reports/data_overview`.

It will include the distribution of classes in the dataset.

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

## 5. Custom inference 

There is a functionaluty to classify your own audio clip. It should be no longer
then 10 seconds and either wav or mp3 format. 

After training the model, rerun the main.py script. Now, if you decline 
training the model, you can choose to classify your own audio clip. 

```bash
python main.py
# Do you wish to train the model? [y/n]: n
# Training model terminated.
# Do you want to view existing training log visualizations? [y/n]: n
# Do you want to classify your own audio clip (<=10s)? [y/n]: 
```

Type "y" to the terminal. You will be prompted to choose models, 
so you could compare different.

It is very important when prompted to type the path without the quotation marks.