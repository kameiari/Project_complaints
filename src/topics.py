from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def compute_npmi_coherence(
    topics_words: list[list[str]], tokenized_docs: list[str], eps: float = 1e-12
) -> float:
    """Compute mean NPMI topic coherence across topics using binary doc co-occurrence."""
    vocab = sorted({w for topic in topics_words for w in topic})
    if not vocab:
        return 0.0

    vec = CountVectorizer(vocabulary=vocab, binary=True)
    dtm = vec.fit_transform(tokenized_docs)  # shape: (n_docs, n_vocab)
    n_docs = dtm.shape[0]
    word_to_idx = vec.vocabulary_

    # Document frequencies & co-occurrence counts
    doc_freq = np.asarray(dtm.sum(axis=0)).ravel()
    co_doc_freq = (dtm.T @ dtm).toarray()

    topic_scores = []
    for words in topics_words:
        valid_idx = [word_to_idx[w] for w in words if w in word_to_idx]
        pair_scores = []
        for i in range(1, len(valid_idx)):
            for j in range(i):
                idx_i, idx_j = valid_idx[i], valid_idx[j]
                p_i = doc_freq[idx_i] / n_docs
                p_j = doc_freq[idx_j] / n_docs
                p_ij = co_doc_freq[idx_i, idx_j] / n_docs
                if p_ij > 0 and p_i > 0 and p_j > 0:
                    pmi = np.log((p_ij + eps) / (p_i * p_j))
                    npmi = pmi / (-np.log(p_ij + eps))
                    pair_scores.append(npmi)
        if pair_scores:
            topic_scores.append(float(np.mean(pair_scores)))

    return float(np.mean(topic_scores)) if topic_scores else 0.0


def run_nmf_pipeline(df: pd.DataFrame, n_topics: int = 10, top_n_words: int = 10):
    """Fit TF-IDF + NMF and return topic summary DataFrame + NPMI coherence."""
    extra_stopwords = [
        "not", "no", "never", "tell", "say", "receive", "send",
        "time", "information", "know", "want", "ask", "go", "get",
    ]
    vectorizer = TfidfVectorizer(
        max_features=8000,
        min_df=10,
        max_df=0.85,
        ngram_range=(1, 2),
        stop_words=extra_stopwords,
    )
    tfidf_matrix = vectorizer.fit_transform(df["tokens"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    nmf = NMF(
        n_components=n_topics,
        random_state=42,
        init="nndsvda",
        max_iter=400,
    )
    doc_topic = nmf.fit_transform(tfidf_matrix)
    df["nmf_topic"] = doc_topic.argmax(axis=1)

    topics_words = []
    rows = []
    for idx, comp in enumerate(nmf.components_):
        top_idx = comp.argsort()[::-1][:top_n_words]
        # Use single-token terms for NPMI lookup alongside full display words
        words = feature_names[top_idx].tolist()
        unigram_words = [w.split()[0] for w in words]
        topics_words.append(unigram_words)

        count = int((df["nmf_topic"] == idx).sum())
        rows.append({
            "model": "TF-IDF + NMF",
            "topic_id": idx,
            "count": count,
            "share_pct": round(100 * count / len(df), 2),
            "top_words": ", ".join(words),
        })

    npmi_score = compute_npmi_coherence(topics_words, df["tokens"].tolist())
    return pd.DataFrame(rows), npmi_score


def run_bertopic_pipeline(df: pd.DataFrame, n_topics: int = 10, top_n_words: int = 10):
    """Fit Sentence-BERT + BERTopic on clean_text and return summary + NPMI coherence."""
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = df["clean_text"].tolist()
    embeddings = embed_model.encode(docs, show_progress_bar=True, batch_size=64)

    # Strip English stopwords from c-TF-IDF topic representations
    vectorizer_model = CountVectorizer(stop_words="english", min_df=5)
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        nr_topics=n_topics + 1,  # +1 accounts for outlier topic -1
        min_topic_size=50,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)
    df["bertopic_topic"] = topics

    info = topic_model.get_topic_info()
    info = info[info["Topic"] != -1].head(n_topics)

    topics_words = []
    rows = []
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        words = [w for w, _ in topic_model.get_topic(tid)[:top_n_words]]
        topics_words.append(words)
        count = int(row["Count"])
        rows.append({
            "model": "BERTopic",
            "topic_id": tid,
            "count": count,
            "share_pct": round(100 * count / len(df), 2),
            "top_words": ", ".join(words),
        })

    npmi_score = compute_npmi_coherence(topics_words, df["clean_text"].tolist())
    return pd.DataFrame(rows), npmi_score


def main(sample_size: int = 25000, n_topics: int = 10):
    df = pd.read_csv(RESULTS_DIR / "complaints_clean.csv")
    df = df.dropna(subset=["tokens", "clean_text"]).reset_index(drop=True)

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"Running comparative topic modeling on N={len(df):,} narratives...")

    nmf_summary, nmf_npmi = run_nmf_pipeline(df, n_topics=n_topics)
    print(f"\n[TF-IDF + NMF] Mean NPMI Coherence: {nmf_npmi:.4f}")
    print(nmf_summary[["topic_id", "share_pct", "top_words"]].to_string(index=False))

    bert_summary, bert_npmi = run_bertopic_pipeline(df, n_topics=n_topics)
    print(f"\n[BERTopic] Mean NPMI Coherence: {bert_npmi:.4f}")
    print(bert_summary[["topic_id", "share_pct", "top_words"]].to_string(index=False))

    summary_all = pd.concat([nmf_summary, bert_summary], ignore_index=True)
    summary_all.to_csv(RESULTS_DIR / "topic_comparison_summary.csv", index=False)

    metrics = pd.DataFrame([
        {"model": "TF-IDF + NMF", "n_docs": len(df), "n_topics": n_topics, "npmi_coherence": round(nmf_npmi, 4)},
        {"model": "BERTopic", "n_docs": len(df), "n_topics": n_topics, "npmi_coherence": round(bert_npmi, 4)},
    ])
    metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    print(f"\nSaved topic tables to {RESULTS_DIR}.")


if __name__ == "__main__":
    main()