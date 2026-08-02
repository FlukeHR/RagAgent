from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import uuid
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from config.settings import resolve_model_path
from retrieval.chunker import Chunk


class Embedder:
    """Embedding backend with an explicit compatibility signature."""

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
        self._load_attempted = False
        self._load_lock = threading.Lock()

    def _ensure_model(self) -> None:
        if not self.use_sentence_transformers:
            return
        with self._load_lock:
            if self._load_attempted or not self.use_sentence_transformers:
                return
            self._load_attempted = True
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(resolve_model_path(self.model_name))
            except Exception as exc:  # deterministic local fallback
                self.load_error = str(exc)

    @property
    def backend(self) -> str:
        self._ensure_model()
        return "sentence_transformers" if self._st_model is not None else "hashing"

    @property
    def dimension(self) -> int:
        self._ensure_model()
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
        self._ensure_model()
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if self._st_model is not None:
            values = self._st_model.encode(
                list(texts), convert_to_numpy=True, normalize_embeddings=True
            )
            return values.astype(np.float32)
        sparse = self._vectorizer.transform(list(texts))
        dense = sparse.toarray().astype(np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-8
        return dense / norms


class IndexCompatibilityError(RuntimeError):
    """Raised when query and index embeddings use different vector spaces."""


class VectorStore:
    """Versioned local vector snapshot with manifest-last atomic publication."""

    def __init__(self, index_dir: str) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.index_dir / "manifest.json"
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._faiss_index = None
        self._generation: str | None = None
        self._embedding_signature: dict = {}

    @property
    def generation(self) -> str | None:
        return self._generation

    @property
    def embedding_signature(self) -> dict:
        return self._embedding_signature

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def build(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        files: dict[str, str] | None = None,
        params: dict | None = None,
        embedding_signature: dict | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if vectors.ndim != 2 or (chunks and vectors.shape[1] <= 0):
            raise ValueError("vectors must be a non-empty two-dimensional matrix")

        generation = uuid.uuid4().hex
        artifacts = {
            "vectors": f"vectors-{generation}.npy",
            "metadata": f"metadata-{generation}.pkl",
            "faiss": f"faiss-{generation}.index",
        }
        vector_path = self.index_dir / artifacts["vectors"]
        metadata_path = self.index_dir / artifacts["metadata"]
        faiss_path = self.index_dir / artifacts["faiss"]
        manifest_tmp = self.index_dir / f"manifest-{generation}.tmp"

        values = vectors.astype(np.float32)
        try:
            with vector_path.open("wb") as handle:
                np.save(handle, values)
                handle.flush()
                os.fsync(handle.fileno())
            with metadata_path.open("wb") as handle:
                pickle.dump(chunks, handle)
                handle.flush()
                os.fsync(handle.fileno())

            has_faiss = False
            try:
                import faiss

                faiss_index = faiss.IndexFlatIP(values.shape[1])
                faiss_index.add(values)
                faiss.write_index(faiss_index, str(faiss_path))
                has_faiss = True
            except (ImportError, RuntimeError):
                faiss_path.unlink(missing_ok=True)

            manifest = {
                "generation": generation,
                "num_chunks": len(chunks),
                "dim": int(values.shape[1]),
                "files": files or {},
                "params": params or {},
                "embedding": embedding_signature or {},
                "artifacts": {
                    "vectors": artifacts["vectors"],
                    "metadata": artifacts["metadata"],
                    "faiss": artifacts["faiss"] if has_faiss else None,
                },
            }
            manifest_tmp.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(manifest_tmp, self.manifest_path)
        except Exception:
            for path in (vector_path, metadata_path, faiss_path, manifest_tmp):
                path.unlink(missing_ok=True)
            raise

        self._chunks = chunks
        self._vectors = values
        self._generation = generation
        self._embedding_signature = embedding_signature or {}
        self._load_faiss(faiss_path if faiss_path.exists() else None)
        self._remove_old_snapshots(generation)

    def _remove_old_snapshots(self, current: str) -> None:
        for pattern in ("vectors-*.npy", "metadata-*.pkl", "faiss-*.index"):
            for path in self.index_dir.glob(pattern):
                if current not in path.name:
                    path.unlink(missing_ok=True)

    def read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def load(self, expected_embedding: dict | None = None) -> None:
        manifest = self.read_manifest()
        artifacts = manifest.get("artifacts", {})
        if not manifest or not artifacts:
            raise FileNotFoundError(
                "Index snapshot is missing or legacy; run indexing/build_index.py"
            )
        stored_embedding = manifest.get("embedding", {})
        if expected_embedding and (
            stored_embedding.get("fingerprint") != expected_embedding.get("fingerprint")
        ):
            raise IndexCompatibilityError(
                "Embedding backend differs from the index snapshot; rebuild the index"
            )
        vector_path = self._artifact_path(artifacts.get("vectors"))
        metadata_path = self._artifact_path(artifacts.get("metadata"))
        if vector_path is None or metadata_path is None:
            raise FileNotFoundError("Index artifact path is invalid")
        self._vectors = np.load(vector_path)
        with metadata_path.open("rb") as handle:
            self._chunks = pickle.load(handle)
        self._generation = str(manifest["generation"])
        self._embedding_signature = stored_embedding
        self._load_faiss(self._artifact_path(artifacts.get("faiss")))

    def reload_if_changed(self, expected_embedding: dict | None = None) -> bool:
        generation = self.read_manifest().get("generation")
        if generation and generation != self._generation:
            self.load(expected_embedding)
            return True
        return False

    def _artifact_path(self, name: str | None) -> Path | None:
        if not name:
            return None
        candidate = (self.index_dir / name).resolve()
        if candidate.parent != self.index_dir.resolve() or not candidate.exists():
            return None
        return candidate

    def _load_faiss(self, path: Path | None) -> None:
        self._faiss_index = None
        if path is None:
            return
        try:
            import faiss

            self._faiss_index = faiss.read_index(str(path))
        except (ImportError, RuntimeError):
            self._faiss_index = None

    @property
    def vectors(self) -> np.ndarray | None:
        return self._vectors

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        allowed_indices: Iterable[int] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if self._vectors is None or not self._chunks:
            self.load()
        assert self._vectors is not None
        if query_vector.ndim == 1:
            query_vector = query_vector[None, :]
        if query_vector.shape[1] != self._vectors.shape[1]:
            raise IndexCompatibilityError("Query/index vector dimensions differ; rebuild")

        if allowed_indices is not None:
            indices = np.asarray(list(allowed_indices), dtype=np.int64)
            if not len(indices):
                return []
            scores = self._vectors[indices] @ query_vector[0]
            order = np.argsort(-scores)[:top_k]
            return [
                (self._chunks[int(indices[position])], float(scores[position]))
                for position in order
            ]

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(
                query_vector.astype(np.float32), min(top_k, len(self._chunks))
            )
            return [
                (self._chunks[int(index)], float(score))
                for index, score in zip(indices[0], scores[0])
                if index >= 0
            ]
        scores = self._vectors @ query_vector[0]
        indices = np.argsort(-scores)[:top_k]
        return [(self._chunks[int(index)], float(scores[index])) for index in indices]
