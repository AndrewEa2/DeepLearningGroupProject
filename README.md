# Deep Learning Group Project

## Dataset

Use [Marcel Wiechmann's cleaned Enron-Spam CSV release](https://github.com/MWiechmann/enron_spam_data).
Download [enron_spam_data.zip](https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip).

Place the extracted CSV in `data/enron_spam_data.csv`. Excluded from git.

## Setup

### With a virtual environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name deep-learning-group-project --display-name "Python 3.13 (Deep Learning Group Project)"
```

### Without a virtual environment

```powershell
py -3.13 -m pip install --user -r requirements.txt
py -3.13 -m ipykernel install --user --name deep-learning-group-project-py313 --display-name "Python 3.13 (Deep Learning Group Project)"
```

## Prepare data

Inspect the data:

```powershell
python scripts/data_setup.py
```

Create the train/validation/test split with:

```powershell
python scripts/make_splits.py
```

This writes `outputs/split_manifest.csv` 
