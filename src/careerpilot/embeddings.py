"""Semantic embedding index using Sentence Transformers and Scikit-learn cosine similarity."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from careerpilot.models import JobPosting

logger = logging.getLogger(__name__)


class SemanticJobIndex:
    """Encodes job postings and computes semantic cosine similarity scores."""

    def __init__(
        self,
        jobs: Sequence[JobPosting],
        model_name: str = "all-MiniLM-L6-v2",
        prefer_transformer: bool = True,
    ) -> None:
        self.jobs = list(jobs)
        self.model_name = model_name
        self.backend: str = "tfidf"
        self._st_model = None
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._job_matrix: np.ndarray

        corpus = [job.to_embedding_text() for job in self.jobs]
        if not corpus:
            self._job_matrix = np.zeros((0, 1), dtype=np.float32)
            return

        if prefer_transformer:
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(model_name)
                self._job_matrix = np.asarray(
                    self._st_model.encode(corpus, normalize_embeddings=True, show_progress_bar=False),
                    dtype=np.float32,
                )
                self.backend = f"sentence-transformers ({model_name})"
                return
            except Exception as exc:
                logger.warning(
                    "SentenceTransformer '%s' unavailable (%s); using Scikit-learn TF-IDF fallback.",
                    model_name,
                    exc,
                )
                self._st_model = None

        self._tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._job_matrix = self._tfidf_vectorizer.fit_transform(corpus)
        self.backend = "tfidf-fallback"

    def score_query(self, query_text: str) -> list[float]:
        """Return cosine similarity scores in [0.0, 1.0] aligned with `self.jobs`."""
        if not self.jobs or not query_text.strip():
            return [0.0] * len(self.jobs)

        if self._st_model is not None:
            query_vec = np.asarray(
                self._st_model.encode([query_text], normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )
            sims = cosine_similarity(query_vec, self._job_matrix)[0]
        elif self._tfidf_vectorizer is not None:
            query_vec = self._tfidf_vectorizer.transform([query_text])
            sims = cosine_similarity(query_vec, self._job_matrix)[0]
        else:
            return [0.0] * len(self.jobs)

        # Clip to [0, 1] for clean percentage scaling
        return [float(max(0.0, min(1.0, s))) for s in sims]
