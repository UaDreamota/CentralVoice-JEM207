# Central Voice

![Banner](/docs/banner.png)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/dataset-CREMA--D-orange.svg)](https://github.com/CheyneyComputerScience/CREMA-D)
[![Author 1](https://img.shields.io/badge/Arsenii%20Rybchenko-LinkedIn-pink?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arsenii-rybchenko-841b38278/)
[![Author 2](https://img.shields.io/badge/Mykyta%20Huskov-LinkedIn-green?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mykytahuskov)
---
> End-to-end Speech Emotion Recognition \
> From raw audio to trained models, developed for the course at IES in Charles University JEM207.  
> Current version classifies six emotions using the CREMA-D dataset.
---

##  Features
- Automatic dataset download & preprocessing  
-  Feature extraction (MFCCs, spectrograms)  
-  Train/test/dev splits with reproducibility  
-  Three CNN-based models (baseline & experimental)  
-  Visualization of results (reports & confusion matrices) 
---
 ## Installation 
1. **Clone the repository**
   ```bash
   git clone https://github.com/UaDreamota/CentralVoice-JEM207.git
   cd CentralVoice-JEM207
   ```
2. **Create and activate a Python**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Linux/Mac
   .venv\Scripts\activate      # on Windows
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

main.py is the file that manages all parts of the script. It handles all the steps of the project.
In order to start run main.py:

   ```bash
   python main.py
   ```

Common workflows are documented in
[docs/Running.md](docs/Running.md).

You can also find our full report in
[reports/data_and_modelling_report.pdf](reports/data_and_modelling_report.pdf)

## Project Structure
```
CentralVoice-JEM207/
├── main.py              # Entry script (start here)
├── docs                 # Documentation and guides
├── scripts/
│   ├── models/          # Baseline and experimental CNN models
│   │   ├── baseline     # Baseline CNN model
│   │   ├── no_cbam      # CNN model without attention
│   │   └── cbam         # CNN model with CBAM attention
│   └── utils/           # Data download, logging, evaluation utilities
├── docs/                # Usage and setup guides
├── reports/             # Generated reports and figures
└── requirements.txt
```