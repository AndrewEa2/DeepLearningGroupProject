# Deep Learning Group Project

Email spam classification with ModernBERT and a word co-occurrence GCN. See [Project_Proposal_DeepLearning.pdf](Project_Proposal_DeepLearning.pdf).

## Dataset

Use [Marcel Wiechmann's cleaned Enron-Spam CSV release](https://github.com/MWiechmann/enron_spam_data).
Download [enron_spam_data.zip](https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip).

Place the extracted CSV in `data/enron_spam_data.csv`. A local copy is already present but excluded from Git.

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

Inspect the dataset with `python scripts/inspect_dataset.py`.
