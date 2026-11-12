"""End-to-end CLI pipeline orchestrating OOP preprocessing and model benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.preprocessor import CorpusPreprocessor
from src.topic_models import (
    NeuralBERTopicModel,
    SparseTopicModel,
    TopicCoherenceEvaluator,
    TopicModelResult,
)


class ComplaintAnalysisPipeline:
    """Coordinates data loading, preprocessing, vectorization comparison, and artifact export."""

    def __init__(self, root_dir: Path, n_samples: int = 25000, n_topics: int = 10, seed: int = 42) -> None:
        self.root_dir = root_dir
        self.results_dir = root_dir / "results"
        self.n_samples = n_samples
        self.n_topics = n_topics
        self.seed = seed

    def load_or_preprocess(self, force_preprocess: bool = False) -> pd.DataFrame:
        """Loads cached cleaned complaints or runs ``CorpusPreprocessor`` if unavailable."""
        clean_path = self.results_dir / "complaints_clean.csv"
        if clean_path.exists() and not force_preprocess:
            df = pd.read_csv(clean_path)
        else:
            raw_candidates = list((self.root_dir / "data").glob("*.csv"))
            if not raw_candidates:
                raise FileNotFoundError("No raw CSV dataset found in data/.")
            preprocessor = CorpusPreprocessor()
            df = preprocessor.process_file(raw_candidates[0], clean_path)
        return df.dropna(subset=["clean_text", "tokens"]).reset_index(drop=True)

    def run_benchmarks(self, include_bertopic: bool = True) -> None:
        """Executes BoW vs. TF-IDF (NMF & LDA) and optional BERTopic comparison."""
        df = self.load_or_preprocess()
        sample_df = df.sample(n=min(self.n_samples, len(df)), random_state=self.seed).reset_index(drop=True)

        evaluator = TopicCoherenceEvaluator(sample_df["tokens"].tolist())
        results: List[TopicModelResult] = []

        # Phase 2 & 3 Sparse Configurations: BoW vs TF-IDF across NMF and LDA
        for vec_type in ("bow", "tfidf"):
            for algo in ("nmf", "lda"):
                model = SparseTopicModel(
                    vectorizer_type=vec_type,
                    algorithm=algo,
                    n_topics=self.n_topics,
                    random_state=self.seed,
                )
                res = model.fit_evaluate(
                    texts=sample_df["tokens"].tolist(),
                    evaluator=evaluator,
                    labels=sample_df["product"],
                )
                results.append(res)

        if include_bertopic:
            bert_model = NeuralBERTopicModel(n_topics=self.n_topics, random_state=self.seed)
            results.append(
                bert_model.fit_evaluate(
                    raw_texts=sample_df["clean_text"].tolist(),
                    evaluator=evaluator,
                )
            )

        metrics_df = pd.DataFrame(
            [
                {
                    "model": f"{r.vectorizer_name} + {r.model_name}",
                    "n_topics": r.n_topics,
                    "npmi_coherence": r.npmi_coherence,
                    "product_nmi": r.product_nmi,
                    "outlier_pct": r.outlier_pct,
                }
                for r in results
            ]
        )
        topics_df = pd.concat([r.topics_table for r in results], ignore_index=True)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(self.results_dir / "model_metrics.csv", index=False)
        topics_df.to_csv(self.results_dir / "topic_comparison_summary.csv", index=False)
        print("\n=== Benchmark Summary ===")
        print(metrics_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OOP CFPB Topic Modeling Pipeline")
    parser.add_argument("--samples", type=int, default=25000, help="Stratified evaluation sample size")
    parser.add_argument("--topics", type=int, default=10, help="Number of latent topics K")
    parser.add_argument("--skip-bertopic", action="store_true", help="Run only sparse BoW/TF-IDF models")
    args = parser.parse_args()

    pipeline = ComplaintAnalysisPipeline(
        root_dir=Path(__file__).resolve().parent.parent,
        n_samples=args.samples,
        n_topics=args.topics,
    )
    pipeline.run_benchmarks(include_bertopic=not args.skip_bertopic)


if __name__ == "__main__":
    main()
