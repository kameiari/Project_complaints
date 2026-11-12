# Complaint Topics — Consumer Complaints NLP Analysis

Portfolio project for *Project: Data Analysis* (CSEBDSEDA02).

## What this project does
This project implements an end-to-end Natural Language Processing (NLP) pipeline to analyze unstructured citizen and customer complaint narratives from the Consumer Financial Protection Bureau (CFPB) dataset ($N=25,000$ sampled narratives). Using modular Python scripts, it applies regex and spaCy preprocessing to strip redaction tokens (`XXXX`) and noise, followed by a comparative topic extraction approach using sparse features (TF-IDF + NMF) versus dense semantic features (Sentence-BERT + BERTopic) to provide institutional decision-makers with actionable topic prevalence insights.

## Installation
```bash
git clone https://github.com/kameiari/Project_complaints.git
cd Project_complaints
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=topics
```
## Data

See `data/README.md` — raw data is not committed per `.gitignore` rules.
The project uses the *US Consumer Finance Complaints* dataset (CFPB / Kaggle, CC0 Public Domain).
`data/sample_complaints.csv` is included for lightweight testing.

## Usage

```bash
jupyter lab                       # open notebooks/01_setup_and_eda.ipynb
python src/preprocess.py          # runs data cleaning and saves to results/

```

Later: `python run.py` runs the whole pipeline and writes to `results/`.

## Project structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── data/            raw data (ignored by git) + data/README.md
├── notebooks/       01_setup_and_eda.ipynb
├── src/             reusable modules (preprocess.py, vectorize.py, topics.py)
└── results/         outputs: complaints_clean.csv, tables, figures

```

## Methods & results

*Filled in during Phase 2.*

## Reflection / known issues

*Log technical decisions, coherence score observations, and preprocessing choices here to feed the Phase-3 final abstract.*