from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import fitz

from config.settings import load_settings
from services.app_store import AppStore
from services.documents import DocumentService
from services.evidence_agent import EvidenceAgent, parse_json_response, validate_agent_result


class DocumentMvpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        base = load_settings()
        self.settings = replace(
            base,
            app=replace(
                base.app,
                database_path=str(root / "app.sqlite3"),
                documents_root=str(root / "documents"),
            ),
        )
        self.store = AppStore(self.settings)
        self.user = self.store.create_user("viewer", "Viewer", "unused")
        self.documents = DocumentService(self.settings, self.store)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_pdf_is_rendered_and_searchable(self) -> None:
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "Figure 1: Model B latency is 42 ms")
        result = self.documents.ingest(self.user.user_id, "chart.pdf", pdf.tobytes())
        pdf.close()

        self.assertEqual(result["status"], "ready")
        candidates = self.documents.search_pages(
            self.user.user_id, str(result["document_id"]), "Model B latency"
        )
        self.assertEqual(candidates[0]["page_number"], 1)
        image = self.documents.page_image(
            self.user.user_id, str(result["document_id"]), 1
        )
        self.assertIsNotNone(image)
        assert image is not None
        self.assertTrue(image.is_file())

    def test_visual_evidence_is_bound_to_supplied_pages(self) -> None:
        pages = [{"page_number": 8}, {"page_number": 11}]
        result = validate_agent_result(
            {
                "action": "answer",
                "answer": "Model B has the lowest latency [E1].",
                "evidence": [
                    {
                        "claim": "Model B: 42 ms",
                        "image_index": 2,
                        "bbox_2d": [100, 200, 700, 800],
                    }
                ],
            },
            pages,
        )
        self.assertEqual(result["sources"][0]["page"], 11)

        invented = validate_agent_result(
            {
                "action": "answer",
                "answer": "Unsupported [E1].",
                "evidence": [
                    {"claim": "x", "image_index": 3, "bbox_2d": [1, 1, 2, 2]}
                ],
            },
            pages,
        )
        self.assertEqual(invented["status"], "insufficient_evidence")

    def test_fenced_model_json_is_accepted(self) -> None:
        parsed = parse_json_response('```json\n{"action":"refuse"}\n```')
        self.assertEqual(parsed["action"], "refuse")

    def test_agent_can_request_one_more_page_search(self) -> None:
        class FakeDocuments:
            calls: list[str] = []

            def search_pages(
                self,
                user_id: str,
                document_id: str,
                query: str,
                exclude: set[int] | None = None,
            ) -> list[dict[str, Any]]:
                del user_id, document_id, exclude
                self.calls.append(query)
                return [{"page_number": len(self.calls), "text": query}]

        fake = FakeDocuments()
        agent = EvidenceAgent(self.settings, fake)  # type: ignore[arg-type]
        with patch.object(
            agent,
            "_inspect_pages",
            side_effect=[
                {"action": "search_more", "search_query": "Figure 2 latency"},
                {
                    "action": "answer",
                    "answer": "Model B is fastest [E1].",
                    "evidence": [
                        {
                            "claim": "Model B: 42 ms",
                            "image_index": 1,
                            "bbox_2d": [10, 20, 300, 400],
                        }
                    ],
                },
            ],
        ):
            result = agent.ask("u", "d", "Which model is fastest?", {}, "key")

        self.assertEqual(fake.calls, ["Which model is fastest?", "Figure 2 latency"])
        self.assertEqual(result["sources"][0]["page"], 2)


if __name__ == "__main__":
    unittest.main()
