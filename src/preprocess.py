import re
from pathlib import Path
import pandas as pd
import spacy

# Set project paths (checking both 'Data' and 'data')
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data" if (ROOT / "Data").exists() else ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load spaCy English model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


def clean_text_regex(text: str) -> str:
    """Remove URLs, emails, CFPB redaction tokens (XXXX), and extra spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\bX{2,}-?\d*\b", " ", text)
    text = re.sub(r"X{2,}/X{2,}/X{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def spacy_preprocess(text: str, custom_stopwords: set) -> tuple[str, str]:
    """
    Returns a tuple of:
    - tokens: Lemmatized words, removing stopwords (except negations) for TF-IDF / NMF.
    - clean_text: Lightly cleaned string preserving sentence structure for Embeddings / BERTopic.
    """
    cleaned_regex = clean_text_regex(text)
    if not cleaned_regex:
        return "", ""

    doc = nlp(cleaned_regex)

    tokens = []
    for token in doc:
        if token.lower_ in {"not", "no", "never"}:
            tokens.append(token.lower_)
        elif (
            token.is_alpha
            and not token.is_stop
            and token.lower_ not in custom_stopwords
            and len(token.lemma_) > 2
        ):
            tokens.append(token.lemma_)

    return " ".join(tokens), cleaned_regex


def run_preprocessing(input_csv: str = "us_consumer_finance_complaints.csv"):
    file_path = DATA_DIR / input_csv
    if not file_path.exists():
        print(f"File not found at {file_path}. Looking for sample_complaints.csv...")
        file_path = DATA_DIR / "sample_complaints.csv"

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    text_col = (
        "consumer_complaint_narrative"
        if "consumer_complaint_narrative" in df.columns
        else "text"
    )
    df[text_col] = df[text_col].fillna("").astype(str)

    initial_count = len(df)

    # 1. Deduplicate
    df = df.drop_duplicates(subset=[text_col], keep="first")
    dedup_count = len(df)
    print(f"Dropped {initial_count - dedup_count} exact duplicates.")

    # 2. Filter short texts (< 3 words)
    df["n_words_raw"] = df[text_col].str.split().str.len()
    df = df[df["n_words_raw"] >= 3]
    print(f"Retained {len(df)} documents after dropping short texts.")

    # 3. Custom domain stop-words
    domain_stopwords = {
        "bank",
        "company",
        "credit",
        "account",
        "please",
        "consumer",
        "card",
    }

    print("Running regex & spaCy preprocessing pipeline...")
    processed = [spacy_preprocess(t, domain_stopwords) for t in df[text_col]]

    df["tokens"] = [p[0] for p in processed]
    df["clean_text"] = [p[1] for p in processed]

    # Save processed dataset
    output_path = RESULTS_DIR / "complaints_clean.csv"
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete! Cleaned dataset saved to: {output_path}")


if __name__ == "__main__":
    run_preprocessing()