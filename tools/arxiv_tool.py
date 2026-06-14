from __future__ import annotations

from config.settings import BASE_DIR, Settings
from tools.base import ToolResult


class ArxivTool:
    """arXiv 在线检索工具，可选下载 PDF 入库。"""

    name = "search_arxiv"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.download_dir = BASE_DIR / settings.arxiv.download_dir
        self.max_results = settings.arxiv.max_results

    @staticmethod
    def schema() -> dict:
        return {
            "name": "search_arxiv",
            "description": (
                "在线检索 arXiv 最新论文，返回标题、作者、发表日期、摘要与链接。"
                "当本地论文库不足以回答、或用户询问最新研究进展时使用。"
                "可设 download=true 把 PDF 下载到本地以便后续精读。"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "arXiv 检索关键词"},
                    "max_results": {
                        "type": "integer",
                        "description": "返回论文数量，省略则用默认配置",
                    },
                    "download": {
                        "type": "boolean",
                        "description": "是否下载 PDF 到本地，默认 false",
                    },
                },
                "required": ["query"],
            },
        }

    def run(
        self,
        query: str,
        max_results: int | None = None,
        download: bool = False,
        _id_base: int = 0,
    ) -> ToolResult:
        import arxiv

        n = max_results or self.max_results
        search = arxiv.Search(
            query=query, max_results=n, sort_by=arxiv.SortCriterion.Relevance
        )
        client = arxiv.Client()

        blocks: list[str] = []
        sources: list[dict] = []
        for i, result in enumerate(client.results(search), start=1):
            aid = result.get_short_id()
            sid = f"S{_id_base + i}"
            authors = ", ".join(a.name for a in result.authors[:5])
            summary = " ".join(result.summary.split())
            block = (
                f"[{sid}] {result.title}\n"
                f"作者: {authors}\n"
                f"发表: {result.published.date()}\n"
                f"arxiv_id: {aid}  链接: {result.entry_id}\n"
                f"摘要: {summary}"
            )
            if download:
                self.download_dir.mkdir(parents=True, exist_ok=True)
                target = self.download_dir / f"{aid}.pdf"
                try:
                    self._download_pdf(result, target)
                    block += f"\n（已下载到 {target}）"
                except Exception as exc:  # noqa: BLE001 - 网络/IO 失败不应中断检索
                    block += f"\n（下载失败: {exc}）"
            blocks.append(block)
            sources.append(
                {
                    "id": sid,
                    "chunk_id": aid,
                    "paper_id": aid,
                    "paper_title": result.title,
                    "section": "Abstract",
                    "source": result.entry_id,
                    "score": None,
                    "snippet": summary[:600],
                }
            )

        if not blocks:
            return ToolResult(text=f"arXiv 未检索到与 '{query}' 相关的论文。", sources=[])
        return ToolResult(text="\n\n".join(blocks), sources=sources)

    @staticmethod
    def _download_pdf(result, path) -> None:
        """arxiv 4.x 移除了内置下载，这里用 requests 直接拉取 PDF。"""
        import requests

        url = getattr(result, "pdf_url", None) or result.entry_id.replace(
            "/abs/", "/pdf/"
        )
        resp = requests.get(url, timeout=60, headers={"User-Agent": "paper-rag-agent"})
        resp.raise_for_status()
        path.write_bytes(resp.content)
