from __future__ import annotations

import threading
from pathlib import Path

from config.settings import BASE_DIR, Settings
from indexing.build_index import build_collection
from indexing.prune import prune_collection, touch_papers
from retrieval.retriever import Retriever
from tools.base import ToolResult

# 集合级锁：串行化「同一全文集合」的下载 + 增量重建，避免并发 /ask 写索引文件竞争（护栏 #2）。
_COLLECTION_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _collection_lock(name: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _COLLECTION_LOCKS.setdefault(name, threading.Lock())


class ArxivIngestTool:
    """把模型选定的 arXiv 论文下载入库（增量嵌入）并在其中检索，返回可引用的全文片段。

    与 search_arxiv（只回摘要、负责侦察）配合：模型先浏览摘要挑出确需精读的论文，
    再把它们的 arxiv_id 交给本工具拉全文。下载落到共享全文集合（默认 arxiv），按 hash
    去重 + 增量索引，只对新论文做嵌入；随后用 query 在该集合检索，输出 [S编号] 来源。
    """

    name = "ingest_arxiv_papers"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection = settings.arxiv.full_text_collection
        self.data_dir = BASE_DIR / settings.project.data_root / self.collection
        self.index_dir = BASE_DIR / settings.index.index_root / self.collection
        # 供 harness 按工具覆盖超时：这一步含网络下载 + 嵌入推理，需比默认更长的有界超时。
        self.timeout_seconds = settings.arxiv.ingest_timeout_seconds

    @staticmethod
    def schema() -> dict:
        return {
            "name": "ingest_arxiv_papers",
            "description": (
                "下载指定 arXiv 论文的全文 PDF 并增量入库，再在其中语义检索，返回可引用的全文片段。"
                "用法：先用 search_arxiv 浏览摘要，挑出确实需要精读 / 引用全文的论文，"
                "把它们的 arxiv_id 传入本工具；query 用于在新入库全文中检索最相关片段。"
                "每轮下载数量有上限（见配置 max_ingest_papers），已下载过的论文会自动复用、不重复嵌入。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "arxiv_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要入库精读的 arXiv id 列表（取自 search_arxiv 结果，如 2106.04026v2）",
                    },
                    "query": {
                        "type": "string",
                        "description": "在入库全文中检索用的查询（可用关键词或自然语言）",
                    },
                },
                "required": ["arxiv_ids", "query"],
            },
        }

    @staticmethod
    def _normalize_ids(arxiv_ids) -> list[str]:
        """去重 + 去空白，保持顺序。"""
        seen, out = set(), []
        for x in arxiv_ids or []:
            aid = str(x).strip()
            if aid and aid not in seen:
                seen.add(aid)
                out.append(aid)
        return out

    def run(self, arxiv_ids, query: str, _id_base: int = 0) -> ToolResult:
        ids = self._normalize_ids(arxiv_ids)[: self.settings.arxiv.max_ingest_papers]
        query = (query or "").strip()
        if not ids or not query:
            return ToolResult(text="未提供有效的 arxiv_ids 或 query。", sources=[])

        downloaded: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []  # 已在库，直接复用

        with _collection_lock(self.collection):
            self.data_dir.mkdir(parents=True, exist_ok=True)
            for aid in ids:
                target = self.data_dir / f"{aid}.pdf"
                if target.exists():
                    skipped.append(aid)
                    continue
                if self._ensure_downloaded(aid, target):
                    downloaded.append(aid)
                else:
                    failed.append(aid)

            # 有新论文，或索引尚不存在时，触发增量重建（已存在论文会被复用、不重嵌）。
            if downloaded or not (self.index_dir / "vectors.npy").exists():
                if any(self.data_dir.glob("*.pdf")):
                    build_collection(self.settings, self.collection, incremental=True)

            results = self._search(query)

            # registry：刷新本轮涉及论文的 last_used_at（入库 + 复用 + 检索命中）
            touched = set(downloaded) | set(skipped) | {r.chunk.paper_id for r in results}
            if touched:
                touch_papers(self.settings, self.collection, touched)
            # 入库后自动按容量 LRU 淘汰，保护本轮用到的论文（无超量时几乎零开销）
            prune_collection(self.settings, self.collection, protect=touched)

        return self._format(results, downloaded, skipped, failed, _id_base)

    # ---------- 可注入的网络 / 检索 seam（测试中被 mock） ----------
    def _ensure_downloaded(self, aid: str, target: Path) -> bool:
        """下载单篇 arXiv PDF 到 target，带体积上限；成功返回 True，失败返回 False。"""
        import requests

        url = f"https://arxiv.org/pdf/{aid}"
        max_bytes = int(self.settings.arxiv.max_pdf_mb * 1024 * 1024)
        try:
            with requests.get(
                url, timeout=60, stream=True, headers={"User-Agent": "paper-rag-agent"}
            ) as resp:
                resp.raise_for_status()
                clen = resp.headers.get("Content-Length")
                if clen and int(clen) > max_bytes:
                    return False  # 声明体积超限，跳过
                size, chunks = 0, []
                for block in resp.iter_content(chunk_size=1 << 16):
                    size += len(block)
                    if size > max_bytes:
                        return False  # 流式超限，放弃
                    chunks.append(block)
                target.write_bytes(b"".join(chunks))
                return True
        except Exception:  # noqa: BLE001 - 网络/IO 失败不应中断检索，记为失败由模型调整
            return False

    def _search(self, query: str):
        """在全文集合上做一次检索；索引缺失时返回空。"""
        try:
            retriever = Retriever(settings=self.settings, index_dir=str(self.index_dir))
            return retriever.search(query)
        except FileNotFoundError:
            return []

    # ---------- 结果拼装 ----------
    def _format(
        self, results, downloaded, skipped, failed, _id_base
    ) -> ToolResult:
        note_bits = []
        if downloaded:
            note_bits.append(f"新入库 {len(downloaded)} 篇")
        if skipped:
            note_bits.append(f"复用 {len(skipped)} 篇")
        if failed:
            note_bits.append(f"下载失败 {len(failed)} 篇（{', '.join(failed)}）")
        note = "；".join(note_bits) or "无可入库论文"

        if not results:
            return ToolResult(
                text=f"[入库情况] {note}。未在已入库全文中检索到相关片段。", sources=[]
            )

        blocks: list[str] = []
        sources: list[dict] = []
        for i, r in enumerate(results, start=1):
            c = r.chunk
            sid = f"S{_id_base + i}"
            blocks.append(
                f"[{sid}]《{c.paper_title}》｜章节 {c.section}｜论文ID {c.paper_id}\n{c.content}"
            )
            sources.append(
                {
                    "id": sid,
                    "chunk_id": c.chunk_id,
                    "paper_id": c.paper_id,
                    "paper_title": c.paper_title,
                    "section": c.section,
                    "source": c.source,
                    "score": round(float(r.score), 4),
                    "snippet": c.content[:600],
                    "collection": self.collection,  # 指向 arxiv 全文集合，供前端 PDF 预览定位高亮
                }
            )
        text = f"[入库情况] {note}。检索到以下全文片段：\n\n" + "\n\n".join(blocks)
        return ToolResult(text=text, sources=sources)
