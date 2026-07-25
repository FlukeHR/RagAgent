from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlencode
from xml.etree import ElementTree

from config.settings import Settings
from retrieval.repository import normalize_arxiv_id


@dataclass(frozen=True)
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: tuple[str, ...]
    summary: str
    published: datetime | None
    entry_url: str


class ArxivSearchService:
    """Bounded arXiv Atom API client with an explicit network timeout."""

    endpoint = "https://export.arxiv.org/api/query"

    def __init__(self, settings: Settings) -> None:
        self.timeout = settings.arxiv.request_timeout_seconds
        self.default_max_results = settings.arxiv.max_results

    def search(self, query: str, max_results: int | None = None) -> list[ArxivPaper]:
        import requests

        count = min(max_results or self.default_max_results, self.default_max_results)
        params = urlencode(
            {
                "search_query": f"all:{query.strip()}",
                "start": 0,
                "max_results": count,
                "sortBy": "relevance",
            }
        )
        response = requests.get(
            f"{self.endpoint}?{params}",
            timeout=self.timeout,
            headers={"User-Agent": "paper-rag-agent/1.0"},
        )
        response.raise_for_status()
        return self._parse(response.content)[:count]

    @staticmethod
    def _parse(payload: bytes) -> list[ArxivPaper]:
        root = ElementTree.fromstring(payload)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers: list[ArxivPaper] = []
        for entry in root.findall("atom:entry", ns):
            entry_url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            raw_id = (
                entry_url.split("/abs/", 1)[1]
                if "/abs/" in entry_url
                else entry_url.rstrip("/").split("/")[-1]
            )
            try:
                arxiv_id = normalize_arxiv_id(raw_id)
            except ValueError:
                continue
            published_raw = (
                entry.findtext("atom:published", default="", namespaces=ns) or ""
            ).strip()
            try:
                published = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            except ValueError:
                published = None
            papers.append(
                ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=" ".join(
                        (entry.findtext("atom:title", default="", namespaces=ns) or "").split()
                    ),
                    authors=tuple(
                        (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
                        for author in entry.findall("atom:author", ns)
                    ),
                    summary=" ".join(
                        (entry.findtext("atom:summary", default="", namespaces=ns) or "").split()
                    ),
                    published=published,
                    entry_url=entry_url,
                )
            )
        return papers
