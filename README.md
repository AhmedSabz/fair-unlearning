# Fair Machine Unlearning

Implementation and experimental reproduction of **Fair Machine Unlearning: Data Removal while Mitigating Disparities**.

Our project evaluates whether the proposed Fair Unlearning method can approximate retraining while preserving model accuracy and fairness.

## Team

- Ahmed — Fairness objective, gradients, Hessian, and Fair Unlearning
- Omar — Dataset downloading and preprocessing
- Aaron — Evaluation metrics, experiment logging, tables, and plots

## Clone the Repository

Open PowerShell and run:

```powershell
git clone https://github.com/AhmedSabz/fair-unlearning.git
cd fair-unlearning
```

## Create the Python Environment

Make sure Python 3.10 or newer is installed.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again:

```powershell
.venv\Scripts\Activate.ps1
```

The terminal should now begin with:

```text
(.venv)
```

## Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `requirements.txt` is not available yet, install the current packages manually:

```powershell
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn torch pyyaml
```

## Dataset Setup

Raw datasets are not stored in this repository.

Create the following folders:

```text
data/
├── raw/
│   ├── Adult/
│   ├── COMPAS/
│   └── HSLS/
└── processed/
    ├── Adult/
    ├── COMPAS/
    └── HSLS/
```

Each team member must download the datasets separately and place them in the appropriate folder.

### Adult

Place the following files in:

```text
data/raw/Adult/
```

```text
adult.data
adult.test
adult.names
```

### COMPAS

Use:

```text
compas-scores-two-years.csv
```

Place it in:

```text
data/raw/COMPAS/
```

The target variable is the two-year recidivism label, and race is used as the protected attribute.

### HSLS

Download the public-use HSLS data through the NCES Online Codebook. Place the downloaded file and documentation in:

```text
data/raw/HSLS/
```

The preprocessing script will create:

```text
data/processed/HSLS/hsls_cleaned.csv
```

Do not commit raw or processed datasets to GitHub.

## Run the Project

Run preprocessing first:

```powershell
python -m src.data.preprocessing
```

Run the Adult baseline:

```powershell
python experiments/run_adult_baselines.py
```

Run the core unlearning experiment:

```powershell
python experiments/run_core_unlearning.py
```

Run the complete experiment suite:

```powershell
python experiments/run_full_experiments.py
```

The exact commands may change as the project develops. Check the `experiments/` folder for the current experiment scripts.

## Methods

The project compares:

- Full training with BCE loss
- Full training with the fair loss
- BCE retraining after deletion
- Fair retraining after deletion
- Conventional Newton unlearning
- Fair Unlearning

## Evaluation Metrics

Each experiment records:

- Test accuracy
- Absolute Equalized Odds Difference (AEOD)
- Runtime
- Deletion percentage
- Deletion type
- Dataset
- Random seed
- Method name

The main goal is to determine whether:

```text
Fair Unlearning ≈ Fair Retraining
```

while requiring less computation than retraining.

## Project Structure

```text
fair-unlearning/
├── data/
├── src/
│   ├── data/
│   ├── models/
│   ├── objectives/
│   ├── unlearning/
│   ├── evaluation/
│   └── utils/
├── experiments/
├── tests/
├── configs/
├── results/
├── plots/
├── requirements.txt
└── README.md
```

## Working With Git

Before starting work:

```powershell
git pull origin main
```

Create a branch for your task:

```powershell
git checkout -b feature/your-task-name
```

After making changes:

```powershell
git add .
git commit -m "Describe your changes"
git push -u origin feature/your-task-name
```

Then create a pull request on GitHub.

Do not commit:

```text
.venv/
data/raw/
data/processed/
results/
plots/
__pycache__/
```

## Reproducibility

When running experiments:

- Use a fixed random seed.
- Record the dataset and method.
- Record the deletion percentage and deletion type.
- Save accuracy, AEOD, and runtime.
- Run important experiments five times.
- Report the mean and standard deviation.
