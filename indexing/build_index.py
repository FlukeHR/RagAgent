from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import BASE_DIR, Settings, load_settings
from retrieval.chunker import Chunk, PaperChunker
from retrieval.embedder import Embedder
from retrieval.loader import PaperLoader
from retrieval.image_index import PageImageIndex
from retrieval.pdf_parse import provider_from_config
from retrieval.vector_store import VectorStore


def _file_hash(path: Path) -> str:
    """源文件内容 hash（sha1）。

    PDF 的 OCR/VLM sidecar 也是索引输入，必须纳入 hash；否则 sidecar 改动后增量
    构建会错误复用旧向量。
    """
    h = hashlib.sha1()
    for item in _index_input_files(path):
        with item.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def _index_input_files(path: Path) -> list[Path]:
    files = [path]
    if path.suffix.lower() == ".pdf":
        files.extend(
            p
            for p in (
                path.with_suffix(".ocr.json"),
                path.with_suffix(".vlm.json"),
                path.with_suffix(".elements.json"),
                path.with_suffix(".layout.json"),
                path.with_suffix(".tables.json"),
                path.with_suffix(".figures.json"),
                path.with_suffix(".formulas.json"),
                path.with_suffix(".bboxes.json"),
            )
            if p.exists()
        )
    return files


def _params_signature(settings: Settings, embedding_signature: dict) -> dict:
    """切块 / 嵌入相关参数签名：任一变化都使旧向量不可复用，触发全量重建。"""
    return {
        "chunk_size": settings.index.chunk_size,
        "chunk_overlap": settings.index.chunk_overlap,
        "chunk_metadata_version": 6,
        "embedding": embedding_signature,
        "pdf_parse_provider": settings.pdf_parse.provider,
        "pdf_parse_auto_ocr": settings.pdf_parse.auto_ocr,
    }


def plan_incremental(
    prev_chunks: list[Chunk] | None,
    prev_vectors: np.ndarray | None,
    prev_files: dict[str, str],
    prev_params: dict,
    cur_files: dict[str, str],
    params: dict,
) -> tuple[list[Chunk], np.ndarray | None, list[str], list[str]]:
    """增量索引规划（纯函数，便于测试）。

    返回 (复用的 chunks, 复用的向量, 需重新嵌入的源文件, 已删除的源文件)。
    - 参数签名变化或无历史 → 全量重建（复用为空，所有文件待嵌入）。
    - 文件 hash 未变且历史里有其 chunk → 复用其 chunks + 向量。
    - 新增 / 改动文件 → 待重新嵌入；历史里有、当前没有的文件 → 删除。
    """
    if not prev_chunks or prev_params != params:
        return [], None, list(cur_files), []

    by_source: dict[str, list[int]] = {}
    for i, ch in enumerate(prev_chunks):
        by_source.setdefault(ch.source, []).append(i)

    kept_chunks: list[Chunk] = []
    kept_idx: list[int] = []
    for src, h in cur_files.items():
        if prev_files.get(src) == h and src in by_source:
            for i in by_source[src]:
                kept_chunks.append(prev_chunks[i])
                kept_idx.append(i)

    reused_sources = {c.source for c in kept_chunks}
    build_sources = [s for s in cur_files if s not in reused_sources]
    removed = [s for s in prev_files if s not in cur_files]
    kept_vectors = (
        prev_vectors[kept_idx] if (kept_idx and prev_vectors is not None) else None
    )
    return kept_chunks, kept_vectors, build_sources, removed


def build_index(
    settings: Settings, incremental: bool = True, verbose: bool = False
) -> int:
    """为统一论文库构建索引（默认增量），返回 chunk 数量。

    增量：只对新增/改动的文件重新嵌入，复用未变文件的向量；切块/嵌入参数变化时自动全量重建。
    """
    data_dir = BASE_DIR / settings.project.data_root
    index_dir = BASE_DIR / settings.index.index_root

    pdf_provider = provider_from_config(
        settings.pdf_parse.provider,
        auto_ocr=settings.pdf_parse.auto_ocr,
        timeout_seconds=settings.pdf_parse.timeout_seconds,
    )
    loader = PaperLoader(str(data_dir), pdf_provider=pdf_provider)
    files = list(loader.iter_files())
    cur_files = {str(f): _file_hash(f) for f in files}
    embedder = Embedder(
        settings.embedding.model_name,
        settings.embedding.use_sentence_transformers,
        settings.embedding.fallback_dimension,
    )
    params = _params_signature(settings, embedder.signature)

    store = VectorStore(str(index_dir))
    prev_chunks: list[Chunk] | None = None
    prev_vectors: np.ndarray | None = None
    prev_manifest: dict = {}
    if incremental:
        try:
            store.load()
            prev_chunks, prev_vectors = store.chunks, store.vectors
            prev_manifest = store.read_manifest()
        except FileNotFoundError:
            prev_chunks = None  # 首次构建，无历史

    kept_chunks, kept_vectors, build_sources, removed = plan_incremental(
        prev_chunks,
        prev_vectors,
        prev_manifest.get("files", {}),
        prev_manifest.get("params", {}),
        cur_files,
        params,
    )

    # 仅对需要的文件解析 + 切块 + 嵌入。索引是混合的：
    # 1) 细粒度语义 chunk，用于精确证据；
    # 2) 页级 page chunk，用于页码定位、跨页近似搜索与后续图像/OCR 工具路由。
    chunker = PaperChunker(settings.index.chunk_size, settings.index.chunk_overlap)
    by_path = {str(f): f for f in files}
    new_chunks: list[Chunk] = []
    for src in build_sources:
        doc = loader.load_file(by_path[src])
        if doc is not None:
            new_chunks.extend(chunker.build([doc]))

    all_chunks = kept_chunks + new_chunks
    if not all_chunks:
        raise ValueError(f"未找到可索引的论文：{data_dir}")

    if new_chunks:
        new_vectors = embedder.encode([c.content for c in new_chunks])
    else:
        new_vectors = None

    parts = [v for v in (kept_vectors, new_vectors) if v is not None and len(v)]
    all_vectors = np.vstack(parts) if parts else np.empty((0, 0), dtype=np.float32)

    # OCR runtime may have generated a sidecar while parsing; persist its final hash.
    cur_files = {str(f): _file_hash(f) for f in files}
    store.build(
        all_chunks,
        all_vectors,
        files=cur_files,
        params=params,
        embedding_signature=embedder.signature,
    )
    if settings.image_search.enabled:
        PageImageIndex(index_dir).build(
            [path for path in files if path.suffix.lower() == ".pdf"],
            max_pages=settings.image_search.max_pages,
            max_side=settings.image_search.max_side,
        )

    if verbose:
        n_changed = len([s for s in build_sources if s in prev_manifest.get("files", {})])
        n_new = len(build_sources) - n_changed
        print(
            f"[Index] 复用 {len(kept_chunks)} chunks / "
            f"新增 {n_new} 文件、改动 {n_changed} 文件、删除 {len(removed)} 文件 "
            f"→ 重新嵌入 {len(new_chunks)} chunks"
        )
    return len(all_chunks)


def main() -> None:
    settings = load_settings()
    full = "--full" in sys.argv[1:]

    data_dir = BASE_DIR / settings.project.data_root
    index_dir = BASE_DIR / settings.index.index_root
    print(f"[Index] mode = {'全量' if full else '增量'}")
    print(f"[Index] loading papers from: {data_dir}")

    n = build_index(settings, incremental=not full, verbose=True)
    print(f"[Index] 索引共 {n} chunks")
    print(f"[Index] saved index to: {index_dir}")


if __name__ == "__main__":
    main()
