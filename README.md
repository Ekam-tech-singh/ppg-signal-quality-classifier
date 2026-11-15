# Smartphone PPG Signal Quality Classifier

## Project Overview

This project explores the use of machine learning to classify smartphone photoplethysmography (PPG) recordings as either good-quality or poor-quality signals.

The goal is to identify signal characteristics that can help distinguish usable PPG recordings from recordings with poor signal quality.

## Dataset

**Source:** [BUT PPG Database — PhysioNet](https://physionet.org/content/butppg/2.0.0/)

The project uses the RGB recordings from the BUT PPG Database.

- Recordings: 3,840
- Subjects: 38
- Sampling frequency: 30 Hz
- Recording duration: 10 seconds
- Good-quality label: 1
- Poor-quality label: 0

The dataset itself is not included in this repository. Users must obtain it from PhysioNet.

## Methodology

### 1. Data Loading
PPG recordings and their corresponding quality labels are loaded and organized for analysis.

### 2. Feature Extraction
Seventeen statistical and signal-based features are extracted from each recording, including:

- Mean, standard deviation, minimum, and maximum
- Signal range, median, and RMS
- Skewness and kurtosis
- Mean absolute difference and difference standard deviation
- Peak and trough counts
- Peak prominence statistics
- Peak interval statistics

### 3. Model Training
Three machine learning approaches are compared:

- Random Forest
- Gradient Boosting
- Logistic Regression

### 4. Evaluation
A subject-wise train/test split is used to reduce data leakage between recordings from the same subject.

Subject-aware cross-validation is used to select classification thresholds.

Balanced accuracy is the primary evaluation metric because the dataset contains an unequal number of good-quality and poor-quality recordings.

## Results

The best observed held-out result was obtained using Gradient Boosting.

| Metric | Result |
|---|---:|
| Balanced accuracy | 62.81% |
| Good-quality recall | 61.2% |
| Poor-quality recall | 64.5% |
| Test recordings | 744 |
| Held-out subjects | 8 |

These results represent one held-out subject-wise split. Performance may vary with different subjects and data splits.

## Repository Structure

```text
My-AI-Project/
├── .agents/
├── .github/
├── _bmad/
├── bmad-output/
├── plots/
├── load_dataset.py
├── extract_features.py
├── inspect_features.py
├── train_model.py
├── .gitignore
└── README.md
```

## Requirements

Python 3.12 or compatible version.

Install the required packages:

```bash
pip install pandas numpy scipy scikit-learn matplotlib wfdb
```

## Running the Project

1. Download the BUT PPG Database from PhysioNet.
2. Extract the database into the expected `data/raw/` directory.
3. Run the scripts in order:

```bash
python load_dataset.py
python extract_features.py
python inspect_features.py
python train_model.py
```

## Limitations

- The dataset has an imbalanced class distribution.
- Evaluation uses a single held-out subject split.
- Results may vary across subjects and recording conditions.
- Peak detection and hand-engineered features may not capture every characteristic of PPG signal quality.
- This project is a research prototype and is not intended for clinical diagnosis or medical decision-making.

## Tools Used

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib
- WFDB
- BMAD Method

## Acknowledgments

The BUT PPG Database is provided by Brno University of Technology and hosted by PhysioNet.
