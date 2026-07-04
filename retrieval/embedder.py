from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class Embedder:
    def __init__(self, model_name: str, use_sentence_transformers: bool = True) -> None:
        self.model_name = model_name
        self.use_sentence_transformers = use_sentence_transformers
        self._st_model = None
        # 回退维度与默认嵌入模型 all-MiniLM-L6-v2 对齐（384），避免建索引与
        # 查询分别走 ST / 哈希两条路径时维度不一致导致 faiss 维度断言失败
        self._vectorizer = HashingVectorizer(n_features=384, alternate_sign=False, norm=None)

        if use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer

                from config.settings import resolve_model_path

                self._st_model = SentenceTransformer(resolve_model_path(model_name))
            except Exception:
                self._st_model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if self._st_model is not None:
            arr = self._st_model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
            return arr.astype(np.float32)

        sparse = self._vectorizer.transform(list(texts))
        dense = sparse.toarray().astype(np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-8
        return dense / norms
