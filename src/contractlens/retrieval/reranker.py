"""Cross-encoder reranker for retrieval results."""

from typing import Optional

from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-encoder reranker for retrieval results."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        # Default to a lightweight cross-encoder
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return []

        # Score query-document pairs
        pairs = [(query, doc) for doc, _ in candidates]
        scores = self.model.predict(pairs)

        # Sort by score and return top-k
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [(doc, float(score)) for (doc, _), score in scored[:top_k]]

    def score(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        scores = self.model.predict([(query, document)])
        return float(scores[0])