from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import BASE_DIR, Settings, load_settings
from retrieval.chunker import PaperChunker
from retrieval.embedder import Embedder
from retrieval.loader import PaperLoader
from retrieval.vector_store import VectorStore


def build_collection(settings: Settings, collection: str) -> int:
    """为指定论文集合构建索引，返回 chunk 数量。"""
    data_dir = BASE_DIR / settings.project.data_root / collection
    index_dir = BASE_DIR / settings.index.index_root / collection

    loader = PaperLoader(str(data_dir))
    docs = loader.load()

    chunker = PaperChunker(settings.index.chunk_size, settings.index.chunk_overlap)
    chunks = chunker.split(docs)
    if not chunks:
        raise ValueError(f"集合 '{collection}' 下未找到可索引的论文：{data_dir}")

    embedder = Embedder(
        settings.embedding.model_name, settings.embedding.use_sentence_transformers
    )
    vectors = embedder.encode([c.content for c in chunks])

    store = VectorStore(str(index_dir))
    store.build(chunks, vectors)
    return len(chunks)


def main() -> None:
    settings = load_settings()
    collection = (
        sys.argv[1] if len(sys.argv) > 1 else settings.project.default_collection
    )

    data_dir = BASE_DIR / settings.project.data_root / collection
    index_dir = BASE_DIR / settings.index.index_root / collection
    print(f"[Index] collection = {collection}")
    print(f"[Index] loading papers from: {data_dir}")

    n = build_collection(settings, collection)
    print(f"[Index] produced {n} chunks")
    print(f"[Index] saved index to: {index_dir}")


if __name__ == "__main__":
    main()
