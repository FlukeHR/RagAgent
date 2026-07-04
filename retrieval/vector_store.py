from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from retrieval.chunker import Chunk


class VectorStore:
    def __init__(self, index_dir: str) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_path = self.index_dir / "vectors.npy"
        self.meta_path = self.index_dir / "metadata.pkl"
        self.faiss_path = self.index_dir / "faiss.index"
        self.manifest_path = self.index_dir / "manifest.json"

        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._faiss_index = None

    def build(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        files: dict[str, str] | None = None,
        params: dict | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")

        self._chunks = chunks
        self._vectors = vectors.astype(np.float32)

        np.save(self.vectors_path, self._vectors)
        with self.meta_path.open("wb") as f:
            pickle.dump(self._chunks, f)

        # files / params 用于增量索引：记录每个源文件的内容 hash 与切块/嵌入签名，
        # 下次 build 时据此判断哪些文件可复用（见 indexing/build_index.py:plan_incremental）。
        manifest = {
            "num_chunks": len(chunks),
            "dim": int(vectors.shape[1]) if vectors.size else 0,
            "files": files or {},
            "params": params or {},
        }
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            import faiss

            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(self._vectors)
            faiss.write_index(index, str(self.faiss_path))
            self._faiss_index = index
        except Exception:  # noqa: BLE001 - faiss 不可用时退化为 numpy 点积检索
            self._faiss_index = None

    @property
    def vectors(self) -> np.ndarray | None:
        return self._vectors

    def read_manifest(self) -> dict:
        """读取 manifest（含 files/params）；不存在时返回空 dict。"""
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def load(self) -> None:
        if not self.vectors_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("Index files are missing, run indexing/build_index.py first")

        self._vectors = np.load(self.vectors_path)
        with self.meta_path.open("rb") as f:
            self._chunks = pickle.load(f)

        if self.faiss_path.exists():
            try:
                import faiss

                self._faiss_index = faiss.read_index(str(self.faiss_path))
            except Exception:
                self._faiss_index = None

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if self._vectors is None or not self._chunks:
            self.load()

        if query_vector.ndim == 1:
            query_vector = query_vector[None, :]

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_vector.astype(np.float32), top_k)
            results: list[tuple[Chunk, float]] = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0:
                    continue
                results.append((self._chunks[int(idx)], float(score)))
            return results

        vectors = self._vectors
        q = query_vector[0]
        scores = vectors @ q
        top_indices = np.argsort(-scores)[:top_k]
        return [(self._chunks[int(i)], float(scores[int(i)])) for i in top_indices]
