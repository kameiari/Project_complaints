"""Object-oriented vectorization and topic modeling benchmark suite.

Supports:
1. Sparse lexical comparison: Bag-of-Words (CountVectorizer) vs. TF-IDF (TfidfVectorizer)
   paired with NMF and LDA (addresses Phase 2 report sections 2.3-2.6).
2. Neural semantic clustering: Sentence-BERT + BERTopic vs. TF-IDF + NMF
   (addresses Phase 3 report sections 3.4-3.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score


@dataclass
class TopicModelResult:
    """Structured container storing outputs and metrics for a fitted topic model."""

    model_name: str
    vectorizer_name: str
    n_topics: int
    npmi_coherence: float
    product_nmi: Optional[float]
    outlier_pct: float
    topics_table: pd.DataFrame
    doc_topics: np.ndarray


class TopicCoherenceEvaluator:
    """Computes Normalized Pointwise Mutual Information (NPMI) coherence from binary doc-term matrices."""

    def __init__(self, texts: List[str], max_vocab: int = 10000) -> None:
        self._vectorizer = CountVectorizer(binary=True, max_features=max_vocab)
        self._binary_dtm = self._vectorizer.fit_transform(texts).tocsc()
        self._vocab: Dict[str, int] = self._vectorizer.vocabulary_
        self._n_docs: int = self._binary_dtm.shape[0]

    def score_topics(self, topic_word_lists: List[List[str]], top_n: int = 10) -> float:
        """Returns mean NPMI across all topics using top-N words per topic."""
        scores: List[float] = []
        eps = 1e-12
        for words in topic_word_lists:
            valid_ids = [self._vocab[w] for w in words[:top_n] if w in self._vocab]
            if len(valid_ids) < 2:
                continue
            pair_scores: List[float] = []
            for i in range(len(valid_ids)):
                col_i = self._binary_dtm[:, valid_ids[i]]
                p_i = col_i.nnz / self._n_docs
                for j in range(i + 1, len(valid_ids)):
                    col_j = self._binary_dtm[:, valid_ids[j]]
                    p_j = col_j.nnz / self._n_docs
                    p_ij = col_i.multiply(col_j).nnz / self._n_docs
                    if p_ij > 0:
                        pmi = np.log((p_ij + eps) / (p_i * p_j + eps))
                        npmi = pmi / (-np.log(p_ij + eps))
                        pair_scores.append(float(npmi))
            if pair_scores:
                scores.append(float(np.mean(pair_scores)))
        return float(np.mean(scores)) if scores else 0.0


class SparseTopicModel:
    """Configurable sparse topic model supporting BoW/TF-IDF with NMF or LDA."""

    BOILERPLATE_STOPWORDS: List[str] = [
        "not", "no", "never", "tell", "say", "receive", "send",
        "time", "information", "know", "want", "ask", "go", "get",
    ]

    def __init__(
        self,
        vectorizer_type: str = "tfidf",
        algorithm: str = "nmf",
        n_topics: int = 10,
        max_features: int = 8000,
        ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42,
    ) -> None:
        self.vectorizer_type = vectorizer_type.lower()
        self.algorithm = algorithm.lower()
        self.n_topics = n_topics
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state

        if self.vectorizer_type == "bow":
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                min_df=10,
                max_df=0.85,
                ngram_range=ngram_range,
                stop_words=self.BOILERPLATE_STOPWORDS,
            )
        elif self.vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=10,
                max_df=0.85,
                ngram_range=ngram_range,
                stop_words=self.BOILERPLATE_STOPWORDS,
            )
        else:
            raise ValueError(f"Unsupported vectorizer_type: {vectorizer_type}")

        if self.algorithm == "nmf":
            self.model = NMF(
                n_components=n_topics,
                random_state=random_state,
                init="nndsvda",
                max_iter=400,
            )
        elif self.algorithm == "lda":
            self.model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=random_state,
                learning_method="batch",
                max_iter=25,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def fit_evaluate(
        self,
        texts: List[str],
        evaluator: TopicCoherenceEvaluator,
        labels: Optional[pd.Series] = None,
        top_n_words: int = 10,
    ) -> TopicModelResult:
        """Fits vectorizer + topic model and computes NPMI, NMI, and topic distributions."""
        dtm = self.vectorizer.fit_transform(texts)
        doc_topic_matrix = self.model.fit_transform(dtm)
        assignments = doc_topic_matrix.argmax(axis=1)

        feature_names = self.vectorizer.get_feature_names_out()
        topic_words: List[List[str]] = []
        rows = []
        for idx, comp in enumerate(self.model.components_):
            top_indices = comp.argsort()[::-1][:top_n_words]
            words = [str(feature_names[i]) for i in top_indices]
            topic_words.append(words)
            share = float((assignments == idx).mean() * 100.0)
            rows.append(
                {
                    "model": f"{self.vectorizer_type.upper()} + {self.algorithm.upper()}",
                    "topic_id": idx,
                    "share_pct": round(share, 2),
                    "top_words": ", ".join(words),
                }
            )

        npmi = evaluator.score_topics(topic_words, top_n=top_n_words)
        nmi = None
        if labels is not None:
            top_cats = labels.value_counts().head(6).index
            mask = labels.isin(top_cats)
            nmi = float(normalized_mutual_info_score(labels[mask], assignments[mask]))

        return TopicModelResult(
            model_name=self.algorithm.upper(),
            vectorizer_name=self.vectorizer_type.upper(),
            n_topics=self.n_topics,
            npmi_coherence=round(npmi, 4),
            product_nmi=round(nmi, 4) if nmi is not None else None,
            outlier_pct=0.0,
            topics_table=pd.DataFrame(rows),
            doc_topics=assignments,
        )


class NeuralBERTopicModel:
    """Wrapper around Sentence-BERT + UMAP + HDBSCAN + c-TF-IDF for semantic topic discovery."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_topics: int = 10,
        min_topic_size: int = 50,
        random_state: int = 42,
    ) -> None:
        self.embedding_model = embedding_model
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.random_state = random_state

    def fit_evaluate(
        self,
        raw_texts: List[str],
        evaluator: TopicCoherenceEvaluator,
        top_n_words: int = 10,
    ) -> TopicModelResult:
        """Encodes prose narratives with Sentence-BERT and extracts reduced BERTopic clusters."""
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from umap import UMAP

        embedder = SentenceTransformer(self.embedding_model)
        embeddings = embedder.encode(raw_texts, show_progress_bar=False)

        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=self.random_state,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=self.n_topics + 1,
            calculate_probabilities=False,
            verbose=False,
        )
        topics, _ = topic_model.fit_transform(raw_texts, embeddings)
        topics_arr = np.array(topics)
        outlier_pct = float((topics_arr == -1).mean() * 100.0)

        info_df = topic_model.get_topic_info()
        valid_topics = info_df[info_df["Topic"] != -1].head(self.n_topics)

        topic_words: List[List[str]] = []
        rows = []
        for _, row in valid_topics.iterrows():
            tid = int(row["Topic"])
            words = [w for w, _ in topic_model.get_topic(tid)[:top_n_words]]
            topic_words.append(words)
            share = float((topics_arr == tid).mean() * 100.0)
            rows.append(
                {
                    "model": "BERTopic",
                    "topic_id": tid,
                    "share_pct": round(share, 2),
                    "top_words": ", ".join(words),
                }
            )

        npmi = evaluator.score_topics(topic_words, top_n=top_n_words)
        return TopicModelResult(
            model_name="BERTopic",
            vectorizer_name=self.embedding_model,
            n_topics=self.n_topics,
            npmi_coherence=round(npmi, 4),
            product_nmi=None,
            outlier_pct=round(outlier_pct, 2),
            topics_table=pd.DataFrame(rows),
            doc_topics=topics_arr,
        )
