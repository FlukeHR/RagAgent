from __future__ import annotations

import json
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import fitz
import requests

from config.settings import MinerUConfig
from retrieval.mineru import MinerUAdapter, MinerUClient, MinerUError, _safe_extract_zip


def _response(status: int, payload: dict) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode("utf-8")
    response.headers["content-type"] = "application/json"
    response.url = "http://127.0.0.1:8001/test"
    return response


class MinerUAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pdf = self.root / "paper.pdf"
        with fitz.open() as document:
            document.new_page()
            document.new_page()
            document.save(self.pdf)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.adapter = MinerUAdapter(MinerUConfig())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, items: list[dict]) -> None:
        (self.raw / "paper_content_list.json").write_text(
            json.dumps(items, ensure_ascii=False), encoding="utf-8"
        )

    def test_canonicalizes_pages_headings_and_multimodal_elements(self) -> None:
        self._write(
            [
                {
                    "type": "text",
                    "text": "Introduction",
                    "text_level": 1,
                    "page_idx": 0,
                    "bbox": [10, 10, 900, 60],
                },
                {
                    "type": "text",
                    "text": "The method starts here.",
                    "page_idx": 0,
                    "bbox": [10, 80, 900, 180],
                },
                {
                    "type": "text",
                    "text": "Method",
                    "text_level": 2,
                    "page_idx": 1,
                    "bbox": [10, 10, 900, 60],
                },
                {
                    "type": "table",
                    "table_caption": ["Table 1"],
                    "table_body": "A | B\n1 | 2",
                    "page_idx": 1,
                    "bbox": [100, 100, 800, 500],
                },
                {
                    "type": "equation",
                    "text": "E = mc^2",
                    "page_idx": 1,
                    "bbox": [100, 520, 600, 620],
                },
                {
                    "type": "image",
                    "image_caption": ["Figure 1"],
                    "content": "A rising accuracy curve.",
                    "page_idx": 1,
                    "bbox": [100, 630, 900, 950],
                },
            ]
        )
        payload = self.adapter.canonicalize(
            self.pdf,
            self.raw,
            {"task_id": "task-1", "status": "completed", "version": "3.4.4"},
        )
        parsed = self.adapter.load(payload, self.pdf)

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["parser"]["backend"], "hybrid-engine")
        self.assertEqual(len(parsed.pages), 2)
        self.assertEqual(parsed.sections[0].title, "Introduction")
        self.assertEqual(parsed.sections[1].heading_path, "Introduction > Method")
        self.assertEqual(
            {item.element_type for item in parsed.elements},
            {"table", "equation", "image"},
        )
        self.assertTrue(all(item.element_id for item in parsed.elements))
        self.assertEqual(parsed.elements[0].bbox, (100.0, 100.0, 800.0, 500.0))

    def test_rejects_out_of_range_bbox(self) -> None:
        self._write(
            [
                {
                    "type": "text",
                    "text": "bad",
                    "page_idx": 0,
                    "bbox": [-1, 0, 100, 100],
                }
            ]
        )
        with self.assertRaises(MinerUError):
            self.adapter.canonicalize(
                self.pdf,
                self.raw,
                {"status": "completed", "version": "3.4.4"},
            )

    def test_rejects_unknown_sidecar_fingerprint(self) -> None:
        self._write(
            [{"type": "text", "text": "body", "page_idx": 0, "bbox": None}]
        )
        payload = self.adapter.canonicalize(
            self.pdf,
            self.raw,
            {"status": "completed", "version": "3.4.4"},
        )
        payload["parser"]["fingerprint"] = "stale"
        with self.assertRaises(MinerUError):
            self.adapter.load(payload, self.pdf)

    def test_handles_rotated_multimodal_and_empty_page_fixture(self) -> None:
        self._write(
            [
                {
                    "type": "chart",
                    "content": "Quarterly revenue rises.",
                    "chart_caption": ["Chart 1"],
                    "page_idx": 0,
                    "angle": 90,
                },
                {
                    "type": "code",
                    "code_body": "print('ok')",
                    "page_idx": 0,
                    "bbox": [20, 20, 400, 300],
                },
                {
                    "type": "list",
                    "list_items": ["first", "second"],
                    "page_idx": 0,
                    "bbox": [20, 320, 400, 500],
                },
            ]
        )
        payload = self.adapter.canonicalize(
            self.pdf,
            self.raw,
            {"task_id": "task-2", "status": "completed"},
        )
        parsed = self.adapter.load(payload, self.pdf)
        self.assertEqual(payload["parser"]["version"], "3.4.4")
        self.assertEqual(
            {item.element_type for item in parsed.elements},
            {"chart", "code", "list"},
        )
        self.assertEqual(parsed.pages[1].text, "")

    def test_rejects_unpinned_mineru_version(self) -> None:
        self._write(
            [{"type": "text", "text": "body", "page_idx": 0, "bbox": None}]
        )
        payload = self.adapter.canonicalize(
            self.pdf,
            self.raw,
            {"status": "completed", "version": "3.4.4"},
        )
        payload["parser"]["version"] = "3.5.0"
        with self.assertRaises(MinerUError):
            self.adapter.load(payload, self.pdf)

    def test_rejects_zip_path_traversal(self) -> None:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as writer:
            writer.writestr("../escape.json", "{}")
        with self.assertRaises(MinerUError):
            _safe_extract_zip(archive.getvalue(), self.raw, MinerUConfig())

    def test_classifies_lost_mineru_task(self) -> None:
        client = MinerUClient(MinerUConfig(max_request_retries=0))
        responses = iter(
            [
                _response(200, {"version": "3.4.4"}),
                _response(404, {"detail": "not found"}),
            ]
        )
        client.session.get = lambda *args, **kwargs: next(responses)
        client.session.post = lambda *args, **kwargs: _response(
            200, {"task_id": "lost-task"}
        )
        with self.assertRaisesRegex(MinerUError, "task_lost"):
            client.parse(self.pdf, self.root / "output")


if __name__ == "__main__":
    unittest.main()
