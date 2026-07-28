"""Shared offline metrics for PDF parsing, chunking and evidence retrieval."""

from __future__ import annotations

import ast
import csv
import difflib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from config.settings import Settings
from retrieval.analyzer import QueryAnalyzer
from retrieval.chunker import Chunk, PaperChunker
from retrieval.embedder import Embedder
from retrieval.models import PaperDocument
from retrieval.pipeline import RetrievalResult, rank_in_memory
from retrieval.reranker import Reranker


@dataclass
class EvidenceCase:
    """One question with document, page and optional verbatim evidence labels."""

    case_id: str
    question: str
    document_ids: list[str]
    gold_pages: list[int]
    gold_evidence: list[str]
    answer: str | None = None
    answerable: bool = True
    evidence_types: list[str] | None = None
    gold_page_texts: dict[int, str] | None = None


def normalize_text(text: str) -> str:
    """Normalize PDF artifacts while retaining content and token order."""

    value = unicodedata.normalize("NFKC", text or "")
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", value)
    value = value.replace("\u00ad", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def text_tokens(text: str) -> list[str]:
    """Tokenize normalized Latin words/numbers and individual CJK characters."""

    normalized = re.sub(r"(?<=\w)-(?=\w)", "", normalize_text(text))
    return re.findall(
        r"[a-z0-9]+(?:[._/%+][a-z0-9]+)*|[\u3400-\u9fff]",
        normalized,
    )


def token_recall(gold: str, prediction: str) -> float:
    """Multiset token recall, robust to PDF whitespace and line-wrap changes."""

    expected = Counter(text_tokens(gold))
    if not expected:
        return 0.0
    actual = Counter(text_tokens(prediction))
    matched = sum((expected & actual).values())
    return matched / sum(expected.values())


def token_precision(gold: str, prediction: str) -> float:
    """Multiset token precision relative to a gold text."""

    actual = Counter(text_tokens(prediction))
    if not actual:
        return 0.0
    expected = Counter(text_tokens(gold))
    matched = sum((expected & actual).values())
    return matched / sum(actual.values())


def token_f1(gold: str, prediction: str) -> float:
    precision = token_precision(gold, prediction)
    recall = token_recall(gold, prediction)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def ordered_similarity(gold: str, prediction: str) -> float:
    """Ordered token similarity using a bounded sequence alignment."""

    expected = text_tokens(gold)
    actual = text_tokens(prediction)
    if not expected:
        return 0.0
    # Downsample pathological pages deterministically while retaining order.
    limit = 4000
    if len(expected) > limit:
        step = math.ceil(len(expected) / limit)
        expected = expected[::step]
    if len(actual) > limit:
        step = math.ceil(len(actual) / limit)
        actual = actual[::step]
    return difflib.SequenceMatcher(
        None,
        expected,
        actual,
        autojunk=False,
    ).ratio()


def evidence_coverage(evidence: str, text: str) -> float:
    """Return 1 for normalized containment, otherwise token recall."""

    gold = normalize_text(evidence)
    candidate = normalize_text(text)
    if not gold:
        return 0.0
    if gold in candidate:
        return 1.0
    return token_recall(gold, candidate)


def parse_list(value: Any) -> list[Any]:
    """Accept native lists, JSON strings, Python-literal strings or CSV scalars."""

    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(stripped)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return [value]


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSON, JSONL or CSV records without requiring a dataset SDK."""

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("data", "rows", "questions", "train"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        raise ValueError(f"JSON 中没有记录列表：{path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"不支持的数据格式：{path}")


def find_structured_file(root: Path, required_keys: set[str]) -> Path:
    """Find the first structured data file whose first record has required keys."""

    candidates: list[Path] = []
    for pattern in ("*.jsonl", "*.json", "*.csv"):
        candidates.extend(sorted(root.rglob(pattern)))
    for candidate in candidates:
        try:
            records = load_records(candidate)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
        if records and required_keys.issubset(records[0]):
            return candidate
    expected = ", ".join(sorted(required_keys))
    raise FileNotFoundError(f"在 {root} 下找不到包含字段 [{expected}] 的数据文件")


def find_pdf(pdf_dir: Path, document_id: str) -> Path:
    """Resolve a PDF by filename or stem, case-insensitively and recursively."""

    requested = Path(document_id).name
    requested_stem = Path(requested).stem.casefold()
    direct_names = [requested, f"{requested}.pdf"] if not requested.lower().endswith(".pdf") else [requested]
    for name in direct_names:
        direct = pdf_dir / name
        if direct.is_file():
            return direct
    for candidate in pdf_dir.rglob("*.pdf"):
        if candidate.name.casefold() == requested.casefold() or candidate.stem.casefold() == requested_stem:
            return candidate
    raise FileNotFoundError(f"找不到 PDF：{document_id}（目录：{pdf_dir}）")


def _page_matches(chunk: Chunk, pages: set[int]) -> bool:
    if chunk.page_start is None or chunk.page_end is None:
        return False
    return any(chunk.page_start <= page <= chunk.page_end for page in pages)


def _strict_page_matches(chunk: Chunk, pages: set[int]) -> bool:
    """Avoid crediting a broad section range as an exact evidence-page hit."""

    return (
        chunk.page_start is not None
        and chunk.page_start == chunk.page_end
        and chunk.page_start in pages
    )


def _evidence_matches(chunk: Chunk, evidence: Sequence[str], threshold: float) -> bool:
    return any(evidence_coverage(item, chunk.content) >= threshold for item in evidence)


def _first_rank(
    results: Sequence[RetrievalResult],
    predicate,
    limit: int,
) -> int | None:
    return next(
        (rank for rank, result in enumerate(results[:limit], start=1) if predicate(result.chunk)),
        None,
    )


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def parsing_metrics(
    documents: Sequence[PaperDocument],
    cases: Sequence[EvidenceCase],
) -> dict[str, float | int]:
    """Measure whether labelled pages/evidence survived PDF parsing."""

    by_id = {document.paper_id: document for document in documents}
    evidence_scores: list[float] = []
    page_text_scores: list[float] = []
    page_nonempty: list[float] = []
    missing_documents = 0
    for case in cases:
        docs = [by_id[item] for item in case.document_ids if item in by_id]
        if not docs:
            missing_documents += 1
            continue
        pages = {
            page.page_number: page.primary_text
            for document in docs
            for page in document.pages
        }
        for page_number in case.gold_pages:
            page_nonempty.append(float(bool(pages.get(page_number, "").strip())))
        for evidence in case.gold_evidence:
            candidates = (
                [pages[page] for page in case.gold_pages if page in pages]
                or list(pages.values())
            )
            evidence_scores.append(
                max((evidence_coverage(evidence, text) for text in candidates), default=0.0)
            )
        for page_number, gold_text in (case.gold_page_texts or {}).items():
            page_text_scores.append(token_f1(gold_text, pages.get(page_number, "")))
    return {
        "cases": len(cases),
        "missing_documents": missing_documents,
        "gold_page_nonempty_rate": _mean(page_nonempty),
        "evidence_text_coverage": _mean(evidence_scores),
        "gold_page_token_f1": _mean(page_text_scores),
    }


def chunking_metrics(
    documents: Sequence[PaperDocument],
    chunks: Sequence[Chunk],
    cases: Sequence[EvidenceCase],
    threshold: float,
) -> dict[str, float | int]:
    """Measure evidence preservation before embedding or retrieval is involved."""

    by_document: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.paper_id, []).append(chunk)
    coverage_scores: list[float] = []
    evidence_hits: list[float] = []
    split_flags: list[float] = []
    lost_flags: list[float] = []
    page_metadata: list[float] = []
    for case in cases:
        candidates = [
            chunk
            for document_id in case.document_ids
            for chunk in by_document.get(document_id, [])
        ]
        for evidence in case.gold_evidence:
            scores = [evidence_coverage(evidence, chunk.content) for chunk in candidates]
            best = max(scores, default=0.0)
            union = evidence_coverage(evidence, "\n".join(chunk.content for chunk in candidates))
            coverage_scores.append(best)
            evidence_hits.append(float(best >= threshold))
            split_flags.append(float(best < threshold <= union))
            lost_flags.append(float(union < threshold))
            if scores and case.gold_pages:
                best_chunk = candidates[int(np.argmax(scores))]
                page_metadata.append(float(_page_matches(best_chunk, set(case.gold_pages))))

    parsed_chars = sum(
        len(normalize_text(page.primary_text))
        for document in documents
        for page in document.pages
    )
    indexed_chars = sum(len(normalize_text(chunk.content)) for chunk in chunks)
    return {
        "chunks": len(chunks),
        "mean_evidence_coverage": _mean(coverage_scores),
        "evidence_preservation_rate": _mean(evidence_hits),
        "evidence_split_rate": _mean(split_flags),
        "evidence_lost_rate": _mean(lost_flags),
        "evidence_page_metadata_accuracy": _mean(page_metadata),
        "indexed_to_parsed_char_ratio": indexed_chars / parsed_chars if parsed_chars else 0.0,
    }


def run_retrieval_benchmark(
    documents: Sequence[PaperDocument],
    cases: Sequence[EvidenceCase],
    settings: Settings,
    *,
    chunk_sizes: Sequence[int],
    overlap_ratios: Sequence[float],
    top_ks: Sequence[int],
    mode: str,
    use_reranker: bool,
    evidence_threshold: float,
    scope: str,
) -> dict[str, Any]:
    """Sweep production chunking and retrieval over already parsed PDFs."""

    if scope not in {"global", "document"}:
        raise ValueError("scope must be global or document")
    if not top_ks or min(top_ks) <= 0:
        raise ValueError("top_ks must contain positive values")
    analyzer = QueryAnalyzer(settings.retrieval.cjk_ngram_size)
    embedder = Embedder(
        settings.embedding.model_name,
        settings.embedding.use_sentence_transformers,
        settings.embedding.fallback_dimension,
    )
    reranker = Reranker(
        settings.rerank.model_name,
        settings.rerank.use_cross_encoder and use_reranker,
        analyzer=analyzer,
    )
    parse_report = parsing_metrics(documents, cases)
    max_k = max(top_ks)
    runs: list[dict[str, Any]] = []

    for chunk_size in chunk_sizes:
        for ratio in overlap_ratios:
            overlap = int(chunk_size * ratio)
            chunker = PaperChunker(chunk_size, overlap)
            chunks = chunker.build(list(documents))
            vectors = embedder.encode([chunk.content for chunk in chunks])
            chunk_report = chunking_metrics(
                documents,
                chunks,
                cases,
                evidence_threshold,
            )
            ranked_by_case: dict[str, list[RetrievalResult]] = {}
            for case in cases:
                allowed = (
                    [
                        index
                        for index, chunk in enumerate(chunks)
                        if chunk.paper_id in set(case.document_ids)
                    ]
                    if scope == "document"
                    else list(range(len(chunks)))
                )
                scoped_chunks = [chunks[index] for index in allowed]
                scoped_vectors = vectors[allowed] if allowed else np.empty((0, embedder.dimension))
                if not scoped_chunks:
                    ranked_by_case[case.case_id] = []
                    continue
                query_vector = embedder.encode([case.question])[0]
                ranked_by_case[case.case_id] = rank_in_memory(
                    case.question,
                    scoped_chunks,
                    scoped_vectors,
                    query_vector,
                    analyzer=analyzer,
                    reranker=reranker,
                    top_k=max_k,
                    top_n=max_k,
                    mode=mode,
                    use_reranker=use_reranker,
                    rrf_k=settings.index.rrf_k,
                    max_chunks_per_parent=settings.index.max_chunks_per_parent,
                )

            for top_k in top_ks:
                paper_hits: list[float] = []
                page_hits: list[float] = []
                page_range_hits: list[float] = []
                page_recalls: list[float] = []
                all_page_hits: list[float] = []
                evidence_hits: list[float] = []
                evidence_recalls: list[float] = []
                reciprocal_ranks: list[float] = []
                unanswerable_low_confidence: list[float] = []
                page_hits_by_type: dict[str, list[float]] = {}
                failed_case_ids: list[str] = []
                for case in cases:
                    results = ranked_by_case[case.case_id]
                    if not case.answerable:
                        confidence = results[0].confidence if results else 0.0
                        unanswerable_low_confidence.append(
                            float(
                                confidence
                                < settings.retrieval.answerability_min_confidence
                            )
                        )
                        continue
                    document_ids = set(case.document_ids)
                    pages = set(case.gold_pages)
                    paper_rank = _first_rank(
                        results,
                        lambda chunk: chunk.paper_id in document_ids,
                        top_k,
                    )
                    paper_hits.append(float(paper_rank is not None))
                    if pages:
                        page_range_hits.append(
                            float(
                                any(
                                    _page_matches(result.chunk, pages)
                                    for result in results[:top_k]
                                )
                            )
                        )
                        retrieved_pages = {
                            page
                            for result in results[:top_k]
                            for page in pages
                            if _strict_page_matches(result.chunk, {page})
                        }
                        page_hit = float(bool(retrieved_pages))
                        page_hits.append(page_hit)
                        page_recalls.append(len(retrieved_pages) / len(pages))
                        all_page_hits.append(float(retrieved_pages == pages))
                        for evidence_type in case.evidence_types or ["unspecified"]:
                            page_hits_by_type.setdefault(evidence_type, []).append(
                                page_hit
                            )
                    if case.gold_evidence:
                        matched_evidence = [
                            evidence
                            for evidence in case.gold_evidence
                            if any(
                                evidence_coverage(evidence, result.chunk.content)
                                >= evidence_threshold
                                for result in results[:top_k]
                            )
                        ]
                        evidence_rank = _first_rank(
                            results,
                            lambda chunk: _evidence_matches(
                                chunk,
                                case.gold_evidence,
                                evidence_threshold,
                            ),
                            top_k,
                        )
                        evidence_hits.append(float(evidence_rank is not None))
                        evidence_recalls.append(
                            len(matched_evidence) / len(case.gold_evidence)
                        )
                        reciprocal_ranks.append(
                            1.0 / evidence_rank if evidence_rank else 0.0
                        )
                    elif pages:
                        page_rank = _first_rank(
                            results,
                            lambda chunk: _strict_page_matches(chunk, pages),
                            top_k,
                        )
                        reciprocal_ranks.append(1.0 / page_rank if page_rank else 0.0)
                    elif paper_rank:
                        reciprocal_ranks.append(1.0 / paper_rank)
                    page_failed = bool(pages) and not any(
                        _strict_page_matches(result.chunk, pages)
                        for result in results[:top_k]
                    )
                    evidence_failed = bool(case.gold_evidence) and not any(
                        _evidence_matches(
                            result.chunk,
                            case.gold_evidence,
                            evidence_threshold,
                        )
                        for result in results[:top_k]
                    )
                    if page_failed or evidence_failed:
                        failed_case_ids.append(case.case_id)
                runs.append(
                    {
                        "chunk_size": chunk_size,
                        "chunk_overlap": overlap,
                        "overlap_ratio": ratio,
                        "top_k": top_k,
                        "mode": mode,
                        "scope": scope,
                        "answerable_cases": sum(case.answerable for case in cases),
                        "paper_hit_rate": _mean(paper_hits),
                        "page_hit_rate": _mean(page_hits),
                        "page_range_hit_rate": _mean(page_range_hits),
                        "page_recall": _mean(page_recalls),
                        "all_gold_pages_hit_rate": _mean(all_page_hits),
                        "evidence_hit_rate": _mean(evidence_hits),
                        "evidence_recall": _mean(evidence_recalls),
                        "mrr": _mean(reciprocal_ranks),
                        "unanswerable_low_confidence_rate": _mean(
                            unanswerable_low_confidence
                        ),
                        "page_hit_rate_by_evidence_type": {
                            key: _mean(values)
                            for key, values in sorted(page_hits_by_type.items())
                        },
                        "failed_case_ids": failed_case_ids,
                        "chunking": chunk_report,
                    }
                )
    return {
        "parsing": parse_report,
        "embedding": embedder.signature,
        "embedding_load_error": embedder.load_error,
        "reranker_backend": reranker.backend,
        "reranker_load_error": reranker.load_error,
        "evidence_threshold": evidence_threshold,
        "runs": runs,
    }


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Write an evaluation report as UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def cases_as_dicts(cases: Sequence[EvidenceCase]) -> list[dict[str, Any]]:
    """Serializable case snapshot for debugging dataset adapters."""

    return [asdict(case) for case in cases]
