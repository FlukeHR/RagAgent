from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import requests

from config.settings import BASE_DIR, MinerUConfig, load_settings
from retrieval.documents import (
    PaperElement,
    PaperPage,
    PaperSection,
    ParsedPDF,
    TextBlock,
)


SIDECAR_SCHEMA_VERSION = 1
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class MinerUError(RuntimeError):
    """Raised when bounded MinerU parsing or canonicalization fails."""


class PDFParseProvider(Protocol):
    """Interface implemented by the single supported PDF parser."""

    name: str

    def parse(self, file: Path) -> ParsedPDF: ...


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class MinerUClient:
    """Bounded localhost client for the MinerU asynchronous task API."""

    def __init__(self, config: MinerUConfig) -> None:
        self.config = config
        parsed = urlparse(config.api_base_url)
        hosts = {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme not in {"http", "https"} or (
            parsed.hostname or ""
        ).lower() not in hosts:
            raise ValueError("MinerU api_base_url must be a localhost HTTP endpoint")
        self.base_url = config.api_base_url.rstrip("/")
        self.session = requests.Session()
        self._retry_events: list[dict[str, Any]] = []

    def health(self) -> dict[str, Any]:
        """Fetch bounded MinerU service metadata."""

        response = self._get(f"{self.base_url}/health", "health")
        try:
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise MinerUError(f"MinerU health failed [service_unavailable]: {exc}") from exc
        return payload if isinstance(payload, dict) else {}

    def parse(self, pdf: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
        """Submit one PDF, poll to completion, and persist its bounded raw result."""

        size = pdf.stat().st_size
        if size > int(self.config.max_pdf_mb * 1024 * 1024):
            raise MinerUError(f"PDF exceeds MinerU size limit: {size} bytes")
        page_count = pdf_page_count(pdf)
        if page_count > self.config.max_pages:
            raise MinerUError(f"PDF exceeds MinerU page limit: {page_count}")

        self._retry_events = []
        health = self.health()
        reported_version = _version_from_payload(health)
        if reported_version and reported_version != self.config.version:
            raise MinerUError(
                f"unsupported MinerU version {reported_version!r}; expected "
                f"exactly {self.config.version}"
            )
        version = reported_version or self.config.version

        form = {
            "backend": self.config.backend,
            "effort": self.config.effort,
            "parse_method": "auto",
            "formula_enable": "true",
            "table_enable": "true",
            "image_analysis": "true",
            "return_md": "true",
            "return_middle_json": "false",
            "return_model_output": "false",
            "return_content_list": "true",
            "return_images": "true",
            "response_format_zip": "true",
            "return_original_file": "false",
        }
        try:
            with pdf.open("rb") as handle:
                response = self.session.post(
                    f"{self.base_url}/tasks",
                    files={"files": (pdf.name, handle, "application/pdf")},
                    data=form,
                    timeout=self._timeout,
                )
            response.raise_for_status()
            submitted = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise MinerUError(f"MinerU submit failed [submission]: {exc}") from exc
        task_id = str(submitted.get("task_id") or submitted.get("id") or "")
        if not task_id:
            raise MinerUError("MinerU task response did not include task_id")

        deadline = time.monotonic() + self.config.parse_timeout_seconds
        last_status = "submitted"
        queued_ahead: int | None = None
        while time.monotonic() < deadline:
            status_response = self._get(
                f"{self.base_url}/tasks/{task_id}", "task_status"
            )
            if status_response.status_code == 404:
                raise MinerUError(f"MinerU task lost [task_lost]: {task_id}")
            try:
                status_response.raise_for_status()
                snapshot = status_response.json()
            except (requests.RequestException, ValueError) as exc:
                raise MinerUError(
                    f"MinerU status failed [task_status] task={task_id}: {exc}"
                ) from exc
            last_status = str(snapshot.get("status") or "unknown").lower()
            queued_ahead = snapshot.get("queued_ahead")
            if last_status in {"completed", "succeeded", "success", "done"}:
                break
            if last_status in {"failed", "error", "cancelled", "canceled"}:
                message = snapshot.get("error") or snapshot.get("message") or last_status
                raise MinerUError(str(message))
            time.sleep(self.config.poll_interval_seconds)
        else:
            raise MinerUError(
                f"MinerU task timed out after {self.config.parse_timeout_seconds}s"
            )

        result = self._get(
            f"{self.base_url}/tasks/{task_id}/result", "task_result", stream=True
        )
        if result.status_code == 404:
            raise MinerUError(f"MinerU task result lost [task_lost]: {task_id}")
        try:
            result.raise_for_status()
        except requests.RequestException as exc:
            raise MinerUError(
                f"MinerU result failed [task_result] task={task_id}: {exc}"
            ) from exc
        max_bytes = int(self.config.max_output_mb * 1024 * 1024)
        content = self._bounded_body(result, max_bytes)
        output_dir.mkdir(parents=True, exist_ok=True)
        content_type = result.headers.get("content-type", "").lower()
        if "zip" in content_type or content.startswith(b"PK\x03\x04"):
            _safe_extract_zip(content, output_dir, self.config)
        else:
            (output_dir / "task_result.json").write_bytes(content)
        return output_dir, {
            "task_id": task_id,
            "status": last_status,
            "queued_ahead": queued_ahead,
            "version": version,
            "page_count": page_count,
            "retries": list(self._retry_events),
        }

    @property
    def _timeout(self) -> tuple[float, float]:
        return (
            self.config.connect_timeout_seconds,
            self.config.request_timeout_seconds,
        )

    def _get(
        self, url: str, phase: str, *, stream: bool = False
    ) -> requests.Response:
        """Retry only idempotent GET requests and retain bounded retry trace."""

        last_error: requests.RequestException | None = None
        for attempt in range(self.config.max_request_retries + 1):
            try:
                response = self.session.get(url, timeout=self._timeout, stream=stream)
            except requests.RequestException as exc:
                last_error = exc
                retryable = True
                reason = type(exc).__name__
            else:
                retryable = response.status_code == 429 or response.status_code >= 500
                reason = f"HTTP {response.status_code}"
                if not retryable or attempt >= self.config.max_request_retries:
                    return response
                response.close()
            if attempt >= self.config.max_request_retries:
                break
            self._retry_events.append(
                {"phase": phase, "attempt": attempt + 1, "reason": reason[:160]}
            )
            time.sleep(min(2.0, 0.25 * (2**attempt)))
        assert last_error is not None
        raise MinerUError(f"MinerU GET failed [{phase}]: {last_error}") from last_error

    @staticmethod
    def _bounded_body(response: requests.Response, max_bytes: int) -> bytes:
        declared = response.headers.get("content-length")
        if declared:
            try:
                declared_size = int(declared)
            except ValueError:
                declared_size = 0
            if declared_size > max_bytes:
                response.close()
                raise MinerUError("MinerU result exceeds configured output limit")
        body = bytearray()
        for chunk in response.iter_content(chunk_size=1 << 20):
            body.extend(chunk)
            if len(body) > max_bytes:
                response.close()
                raise MinerUError("MinerU result exceeds configured output limit")
        response.close()
        return bytes(body)


class MinerUAdapter:
    """Convert MinerU content_list output into the stable project sidecar."""

    def __init__(self, config: MinerUConfig) -> None:
        self.config = config

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(
            {
                "schema": SIDECAR_SCHEMA_VERSION,
                "backend": self.config.backend,
                "effort": self.config.effort,
                "image_analysis": True,
                "formula": True,
                "table": True,
                "version_prefix": self.config.supported_version_prefix,
                "version": self.config.version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def canonicalize(
        self,
        pdf: Path,
        raw_dir: Path,
        parser_trace: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a versioned, page-aware canonical document."""

        content_path = _find_content_list(raw_dir)
        try:
            raw = json.loads(content_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MinerUError(f"invalid MinerU content list: {exc}") from exc
        items = _unwrap_content_list(raw)
        page_count = pdf_page_count(pdf)
        pages: list[dict[str, Any]] = [
            {"page_number": page, "text": "", "blocks": []}
            for page in range(1, page_count + 1)
        ]
        sections: list[dict[str, Any]] = []
        elements: list[dict[str, Any]] = []
        current_title = "Body"
        current_heading_path = "Body"
        current_heading_level: int | None = None
        heading_stack: dict[int, str] = {}
        section_lines: list[str] = []
        section_start = 1
        section_end = 1

        def flush_section() -> None:
            text = "\n".join(section_lines).strip()
            if text:
                sections.append(
                    {
                        "title": current_title,
                        "text": text,
                        "page_start": section_start,
                        "page_end": section_end,
                        "modality": "text",
                        "heading_path": current_heading_path,
                        "heading_level": current_heading_level,
                    }
                )

        for order, item in enumerate(items):
            page = _page_number(item)
            if not 1 <= page <= page_count:
                raise MinerUError(f"MinerU block page out of range: {page}")
            kind = str(item.get("type") or "text").lower()
            bbox = normalized_bbox(item.get("bbox"))
            text = _content_text(item, kind)
            text_level = _int_or_zero(item.get("text_level"))
            pages[page - 1]["blocks"].append(
                {
                    "type": kind,
                    "text": text,
                    "bbox": list(bbox) if bbox else None,
                    "order": order,
                }
            )
            if text:
                separator = "\n" if pages[page - 1]["text"] else ""
                pages[page - 1]["text"] += separator + text

            if kind == "text" and text_level > 0 and text:
                flush_section()
                current_title = text[:200]
                current_heading_level = text_level
                heading_stack[text_level] = current_title
                heading_stack = {
                    level: title
                    for level, title in heading_stack.items()
                    if level <= text_level
                }
                current_heading_path = " > ".join(
                    heading_stack[level] for level in sorted(heading_stack)
                )
                section_lines = []
                section_start = page
                section_end = page
                continue
            if text:
                section_lines.append(text)
                section_end = page

            if kind in {"table", "equation", "image", "chart", "code", "list"}:
                body = _element_body(item, kind)
                stable = hashlib.sha1(
                    f"{page}|{order}|{kind}|{text}".encode("utf-8")
                ).hexdigest()[:16]
                elements.append(
                    {
                        "element_id": f"p{page}-{kind}-{stable}",
                        "element_type": kind,
                        "page_start": page,
                        "page_end": page,
                        "text": body,
                        "markdown": body or None,
                        "modality": {
                            "table": "table",
                            "equation": "formula",
                            "image": "figure",
                            "chart": "chart",
                            "code": "code",
                        }.get(kind, "text"),
                        "bbox": list(bbox) if bbox else None,
                        "caption": _joined(
                            item.get(f"{kind}_caption") or item.get("caption")
                        )
                        or None,
                        "footnote": _joined(item.get(f"{kind}_footnote")) or None,
                        "summary": (
                            clean_text(item.get("content"))
                            if kind in {"image", "chart"}
                            else None
                        ),
                        "source": content_path.name,
                        "heading_path": current_heading_path,
                        "order": order,
                    }
                )
        flush_section()
        if not sections:
            combined = "\n".join(page["text"] for page in pages).strip()
            if combined:
                sections.append(
                    {
                        "title": "Body",
                        "text": combined,
                        "page_start": 1,
                        "page_end": page_count,
                        "modality": "text",
                        "heading_path": "Body",
                        "heading_level": None,
                    }
                )
        version = str(parser_trace.get("version") or self.config.version)
        title = next(
            (
                str(section["title"])
                for section in sections
                if section.get("title") and section["title"] != "Body"
            ),
            pdf.stem,
        )
        return {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "paper_id": pdf.stem,
            "title": title,
            "input_sha256": file_sha256(pdf),
            "parser": {
                "name": "mineru",
                "version": version,
                "backend": self.config.backend,
                "effort": self.config.effort,
                "fingerprint": self.fingerprint,
                "options": {
                    "formula": True,
                    "table": True,
                    "image_analysis": True,
                },
            },
            "page_count": page_count,
            "pages": pages,
            "sections": sections,
            "elements": elements,
            "trace": parser_trace,
        }

    def load(self, payload: dict[str, Any], pdf: Path) -> ParsedPDF:
        """Validate and load one canonical sidecar."""

        if payload.get("schema_version") != SIDECAR_SCHEMA_VERSION:
            raise MinerUError("unsupported MinerU sidecar schema")
        if payload.get("input_sha256") != file_sha256(pdf):
            raise MinerUError("stale MinerU sidecar")
        parser = payload.get("parser") or {}
        if parser.get("fingerprint") != self.fingerprint:
            raise MinerUError("MinerU parser fingerprint changed")
        if parser.get("version") != self.config.version:
            raise MinerUError("MinerU sidecar version does not match the pinned version")
        pages = [
            PaperPage(
                page_number=int(page["page_number"]),
                text=clean_text(page.get("text")),
                blocks=[
                    TextBlock(
                        page_number=int(page["page_number"]),
                        text=clean_text(block.get("text")),
                        bbox=normalized_bbox(block.get("bbox")),
                    )
                    for block in page.get("blocks", [])
                    if clean_text(block.get("text"))
                ],
            )
            for page in payload.get("pages", [])
        ]
        sections = [PaperSection(**section) for section in payload.get("sections", [])]
        elements = [
            PaperElement(
                **{
                    **element,
                    "bbox": normalized_bbox(element.get("bbox")),
                }
            )
            for element in payload.get("elements", [])
        ]
        trace = payload.get("trace")
        return ParsedPDF(
            pages=pages,
            sections=sections,
            elements=elements,
            trace=trace if isinstance(trace, list) else [trace or {}],
            parser_metadata=dict(parser),
        )


class MinerUParseProvider:
    """Load a valid canonical sidecar or create it through the MinerU service."""

    name = "mineru"

    def __init__(self, config: MinerUConfig) -> None:
        self.config = config
        self.adapter = MinerUAdapter(config)
        self.client = MinerUClient(config)
        self.cache_root = (BASE_DIR / config.cache_root).resolve()

    def parse(self, file: Path) -> ParsedPDF:
        """Parse one PDF without silently falling back to another parser."""

        sidecar = file.with_suffix(".mineru.json")
        if sidecar.exists():
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                return self.adapter.load(payload, file)
            except (OSError, json.JSONDecodeError, MinerUError):
                pass

        digest = file_sha256(file)
        raw_dir = self.cache_root / digest / self.adapter.fingerprint
        if has_content_list(raw_dir):
            trace: dict[str, Any] = {
                "task_id": None,
                "status": "raw_cache",
                "version": "unknown",
            }
        else:
            self.cache_root.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self.cache_root.parent) as temp_dir:
                temp_output = Path(temp_dir) / "output"
                _, trace = self.client.parse(file, temp_output)
                raw_dir.parent.mkdir(parents=True, exist_ok=True)
                if raw_dir.exists():
                    shutil.rmtree(raw_dir)
                shutil.move(str(temp_output), str(raw_dir))
        payload = self.adapter.canonicalize(file, raw_dir, trace)
        atomic_json(sidecar, payload)
        return self.adapter.load(payload, file)


def provider_from_config(config: MinerUConfig | None = None) -> MinerUParseProvider:
    """Create the configured MinerU provider without parser fallback."""

    return MinerUParseProvider(config or load_settings().mineru)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish one JSON file using an atomic same-directory replace."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def normalized_bbox(value: Any) -> tuple[float, float, float, float] | None:
    """Validate a MinerU 0..1000 bounding box."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise MinerUError("invalid MinerU bbox")
    try:
        x0, y0, x1, y1 = (float(part) for part in value)
    except (TypeError, ValueError) as exc:
        raise MinerUError("invalid MinerU bbox") from exc
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise MinerUError(f"MinerU bbox out of normalized range: {value}")
    return x0, y0, x1, y1


def clean_text(value: Any) -> str:
    """Remove control characters from untrusted parser content."""

    if not isinstance(value, str):
        return ""
    return _CONTROL_CHARS.sub("", value).strip()


def pdf_page_count(pdf: Path) -> int:
    """Validate a PDF and return its bounded page count."""

    try:
        import fitz
    except ImportError as exc:
        raise MinerUError("PyMuPDF is required for bounded PDF validation") from exc
    try:
        with fitz.open(str(pdf)) as document:
            return len(document)
    except Exception as exc:
        raise MinerUError(f"invalid PDF: {exc}") from exc


def has_content_list(root: Path) -> bool:
    """Return whether a raw MinerU cache contains usable structured output."""

    try:
        _find_content_list(root)
        return True
    except MinerUError:
        return False


def _safe_extract_zip(data: bytes, output_dir: Path, config: MinerUConfig) -> None:
    total = 0
    limit = int(config.max_output_mb * 1024 * 1024)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) > config.max_archive_entries:
            raise MinerUError("MinerU archive contains too many entries")
        root = output_dir.resolve()
        for member in members:
            total += member.file_size
            if total > limit:
                raise MinerUError("MinerU archive exceeds uncompressed size limit")
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise MinerUError("MinerU archive contains an unsafe path")
        archive.extractall(root)


def _find_content_list(root: Path) -> Path:
    candidates = [
        path
        for path in root.rglob("*.json")
        if path.name.endswith("_content_list.json")
        or path.name == "content_list.json"
    ]
    if candidates:
        return sorted(candidates)[0]
    task_result = root / "task_result.json"
    if task_result.exists():
        return task_result
    raise MinerUError("MinerU output did not contain content_list.json")


def _unwrap_content_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            return _unwrap_content_list(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise MinerUError("MinerU content_list string is not valid JSON") from exc
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("content_list", "data", "result", "results"):
            value = raw.get(key)
            if isinstance(value, list):
                if value and isinstance(value[0], dict):
                    nested = value[0].get("content_list")
                    if isinstance(nested, list):
                        return [item for item in nested if isinstance(item, dict)]
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                try:
                    return _unwrap_content_list(value)
                except MinerUError:
                    continue
    raise MinerUError("unsupported MinerU content list envelope")


def _page_number(item: dict[str, Any]) -> int:
    try:
        return int(item.get("page_idx", 0)) + 1
    except (TypeError, ValueError) as exc:
        raise MinerUError("invalid MinerU page_idx") from exc


def _content_text(item: dict[str, Any], kind: str) -> str:
    values: tuple[Any, ...]
    if kind == "table":
        values = (
            item.get("table_caption"),
            item.get("table_body"),
            item.get("table_footnote"),
        )
    elif kind in {"image", "chart"}:
        values = (
            item.get(f"{kind}_caption"),
            item.get("content"),
            item.get(f"{kind}_footnote"),
        )
    elif kind == "code":
        values = (item.get("code_caption"), item.get("code_body"))
    elif kind == "list":
        values = (item.get("list_items"), item.get("text"))
    else:
        values = (item.get("text"), item.get("content"), item.get("latex"))
    return "\n".join(part for part in (_joined(value) for value in values) if part)


def _element_body(item: dict[str, Any], kind: str) -> str:
    if kind == "table":
        return _joined(item.get("table_body"))
    if kind in {"image", "chart"}:
        return _joined(item.get("content"))
    if kind == "code":
        return _joined(item.get("code_body"))
    if kind == "list":
        return _joined(item.get("list_items") or item.get("text"))
    return _joined(item.get("text") or item.get("latex") or item.get("content"))


def _joined(value: Any) -> str:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        return "\n".join(clean_text(item) for item in value if clean_text(item))
    if isinstance(value, dict):
        for key in ("text", "markdown", "content", "summary"):
            if key in value:
                return _joined(value[key])
    return ""


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _version_from_payload(payload: dict[str, Any]) -> str:
    for key in ("version", "version_name", "mineru_version"):
        value = payload.get(key)
        if value:
            return str(value).lstrip("v")
    return ""
