"""Object-oriented text preprocessing module for CFPB consumer complaints.

Provides the ``CorpusPreprocessor`` class to clean raw complaint narratives,
strip redaction artifacts, preserve negation cues, and apply spaCy lemmatization.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class CorpusPreprocessor:
    """Reusable NLP preprocessor for financial complaint narratives.

    Attributes:
        spacy_model: Name of the spaCy language pipeline used for lemmatization.
        min_tokens: Minimum number of tokens required to retain a document.
        stopwords: Set of stopwords stripped during lemmatization (negations preserved).
    """

    DEFAULT_NEGATIONS: Set[str] = {"not", "no", "never", "neither", "nor", "nothing", "nowhere"}
    EXTRA_NOISE_WORDS: Set[str] = {"xxxx", "xx", "xxx", "consumer", "complaint", "company"}

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        min_tokens: int = 5,
        extra_stopwords: Optional[Iterable[str]] = None,
    ) -> None:
        """Initializes the preprocessor with configurable filtering rules."""
        self.spacy_model = spacy_model
        self.min_tokens = min_tokens
        self.stopwords: Set[str] = (set(ENGLISH_STOP_WORDS) - self.DEFAULT_NEGATIONS) | self.EXTRA_NOISE_WORDS
        if extra_stopwords:
            self.stopwords |= set(extra_stopwords)
        self._nlp = None

    @property
    def nlp(self):
        """Lazily loads the spaCy pipeline only when lemmatization is executed."""
        if self._nlp is None:
            self._nlp = spacy.load(self.spacy_model, disable=["parser", "ner"])
        return self._nlp

    @staticmethod
    def clean_raw_text(text: str) -> str:
        """Normalizes casing, redaction placeholders, URLs, and punctuation."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\b[xX]{2,}\b", " ", text)  # Strip CFPB XXXX redactions
        text = re.sub(r"[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def lemmatize_texts(self, texts: Iterable[str], batch_size: int = 512) -> List[str]:
        """Lemmatizes cleaned narratives in batches while filtering short tokens and stopwords."""
        lemmatized_docs: List[str] = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            tokens = [
                token.lemma_
                for token in doc
                if token.is_alpha
                and len(token.lemma_) > 2
                and token.lemma_ not in self.stopwords
            ]
            lemmatized_docs.append(" ".join(tokens))
        return lemmatized_docs

    def transform(self, df: pd.DataFrame, text_col: str = "consumer_complaint_narrative") -> pd.DataFrame:
        """Applies end-to-end cleaning and lemmatization to a complaints DataFrame."""
        work_df = df.dropna(subset=[text_col]).copy()
        work_df["clean_text"] = work_df[text_col].map(self.clean_raw_text)
        work_df = work_df[work_df["clean_text"].str.split().str.len() >= self.min_tokens].copy()
        work_df["tokens"] = self.lemmatize_texts(work_df["clean_text"].tolist())
        work_df = work_df[work_df["tokens"].str.split().str.len() >= self.min_tokens].reset_index(drop=True)
        return work_df

    def process_file(
        self,
        input_csv: Path,
        output_csv: Path,
        text_col: str = "consumer_complaint_narrative",
    ) -> pd.DataFrame:
        """Loads a raw CSV, runs ``transform``, and persists the cleaned corpus."""
        raw_df = pd.read_csv(input_csv, low_memory=False)
        clean_df = self.transform(raw_df, text_col=text_col)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(output_csv, index=False)
        return clean_df
