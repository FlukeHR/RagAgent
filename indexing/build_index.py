from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import BASE_DIR, load_settings
from retrieval.chunker import CodeChunker
from retrieval.embedder import Embedder
from retrieval.loader import CodeLoader
from retrieval.vector_store import VectorStore


def main() -> None:
    settings = load_settings()

    repo_name = settings.project.default_repo
    repo_root = BASE_DIR / settings.project.data_root / repo_name
    index_dir = BASE_DIR / settings.index.index_root / repo_name

    print(f"[Index] Loading files from: {repo_root}")
    loader = CodeLoader(str(repo_root))
    docs = loader.load()
    print(f"[Index] Loaded {len(docs)} files")

    chunker = CodeChunker(settings.index.chunk_size, settings.index.chunk_overlap)
    chunks = chunker.split(docs)
    print(f"[Index] Produced {len(chunks)} chunks")

    embedder = Embedder(settings.embedding.model_name, settings.embedding.use_sentence_transformers)
    vectors = embedder.encode([c.content for c in chunks])

    store = VectorStore(str(index_dir))
    store.build(chunks, vectors)
    print(f"[Index] Saved index to: {index_dir}")


if __name__ == "__main__":
    main()
