"""增量索引规划的离线单测：复用 / 改动 / 新增 / 删除 / 参数失配全量。"""
from __future__ import annotations

import numpy as np

from indexing.build_index import plan_incremental
from retrieval.chunker import Chunk


def _chunk(src: str, i: int) -> Chunk:
    return Chunk(
        chunk_id=f"{src}::{i}", paper_id=src, paper_title=src,
        section="Body", content=f"{src}-{i}", source=src,
    )


PARAMS = {"chunk_size": 900, "chunk_overlap": 150, "embedding_model": "m", "use_sentence_transformers": True}


def test_reuse_unchanged_and_embed_new_and_changed():
    # 历史：A(2 chunks) + B(1 chunk)
    prev_chunks = [_chunk("A", 0), _chunk("A", 1), _chunk("B", 0)]
    prev_vectors = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    prev_files = {"A": "ha", "B": "hb"}
    # 当前：A 未变、B 改动、C 新增
    cur_files = {"A": "ha", "B": "hb2", "C": "hc"}

    kept, kept_vecs, build_sources, removed = plan_incremental(
        prev_chunks, prev_vectors, prev_files, PARAMS, cur_files, PARAMS
    )
    assert [c.source for c in kept] == ["A", "A"]          # 仅 A 复用
    assert kept_vecs.tolist() == [[1.0], [2.0]]            # A 的向量按原顺序复用
    assert build_sources == ["B", "C"]                     # B 改动 + C 新增需重嵌
    assert removed == []


def test_removed_file_dropped():
    prev_chunks = [_chunk("A", 0), _chunk("B", 0)]
    prev_vectors = np.array([[1.0], [2.0]], dtype=np.float32)
    prev_files = {"A": "ha", "B": "hb"}
    cur_files = {"A": "ha"}  # B 被删除

    kept, kept_vecs, build_sources, removed = plan_incremental(
        prev_chunks, prev_vectors, prev_files, PARAMS, cur_files, PARAMS
    )
    assert [c.source for c in kept] == ["A"]
    assert build_sources == []
    assert removed == ["B"]


def test_params_change_forces_full_rebuild():
    prev_chunks = [_chunk("A", 0)]
    prev_vectors = np.array([[1.0]], dtype=np.float32)
    new_params = {**PARAMS, "chunk_size": 1000}
    kept, kept_vecs, build_sources, removed = plan_incremental(
        prev_chunks, prev_vectors, {"A": "ha"}, PARAMS, {"A": "ha"}, new_params
    )
    assert kept == [] and kept_vecs is None
    assert build_sources == ["A"]  # 全量


def test_no_history_is_full_rebuild():
    kept, kept_vecs, build_sources, removed = plan_incremental(
        None, None, {}, {}, {"A": "ha", "B": "hb"}, PARAMS
    )
    assert kept == [] and build_sources == ["A", "B"]
