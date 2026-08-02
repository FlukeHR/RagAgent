from __future__ import annotations

import unittest

from fastapi import HTTPException

from api.main import app
from api.routes import ingest_arxiv
from api.schemas import ArxivProposalRequest


class ApiContractTests(unittest.TestCase):
    def test_legacy_direct_ingest_is_hidden_and_gone(self) -> None:
        self.assertNotIn("/ingest_arxiv", app.openapi()["paths"])
        with self.assertRaises(HTTPException) as caught:
            ingest_arxiv(ArxivProposalRequest(query="test", max_results=None))
        self.assertEqual(caught.exception.status_code, 410)

    def test_async_ingest_routes_are_public(self) -> None:
        paths = app.openapi()["paths"]
        self.assertIn("/api/arxiv/ingest/proposals", paths)
        self.assertIn("/api/arxiv/ingest/confirm", paths)
        self.assertIn("/api/ingest/jobs/{job_id}", paths)

    def test_multi_user_product_routes_are_namespaced(self) -> None:
        paths = app.openapi()["paths"]
        for path in (
            "/api/auth/register",
            "/api/auth/login",
            "/api/dashboard",
            "/api/model-profiles",
            "/api/sessions/{conversation_id}/ask",
            "/api/papers/upload",
        ):
            self.assertIn(path, paths)
        self.assertNotIn("/ask", paths)
        self.assertNotIn("/sessions/{session_id}", paths)


if __name__ == "__main__":
    unittest.main()
