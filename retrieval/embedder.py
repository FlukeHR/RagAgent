from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from config.settings import resolve_model_path


class Embedder:
    """Embedding backend with an explicit, serializable compatibility signature."""

    def __init__(
        self,
        model_name: str,
        use_sentence_transformers: bool = True,
        fallback_dimension: int = 384,
    ) -> None:
        self.model_name = model_name
        self.use_sentence_transformers = use_sentence_transformers
        self.fallback_dimension = fallback_dimension
        self._st_model = None
        self.load_error: str | None = None
        self._vectorizer = HashingVectorizer(
            n_features=fallback_dimension,
            alternate_sign=False,
            norm=None,
        )
        if use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(resolve_model_path(model_name))
            except Exception as exc:  # local deterministic fallback, recorded in manifest
                self.load_error = str(exc)

    @property
    def backend(self) -> str:
        return "sentence_transformers" if self._st_model is not None else "hashing"

    @property
    def dimension(self) -> int:
        if self._st_model is not None:
            return int(self._st_model.get_sentence_embedding_dimension())
        return self.fallback_dimension

    @property
    def signature(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "backend": self.backend,
            "model_name": self.model_name if self._st_model is not None else "hashing",
            "dimension": self.dimension,
            "normalized": True,
            "fallback_dimension": self.fallback_dimension,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        payload["fingerprint"] = hashlib.sha256(encoded).hexdigest()
        return payload

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if self._st_model is not None:
            values = self._st_model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return values.astype(np.float32)
        sparse = self._vectorizer.transform(list(texts))
        dense = sparse.toarray().astype(np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-8
        return dense / norms
