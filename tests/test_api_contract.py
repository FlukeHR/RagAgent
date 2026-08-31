from __future__ import annotations

import unittest

from api.main import app


class ApiContractTests(unittest.TestCase):
    def test_mvp_routes_are_exposed_without_legacy_agent_routes(self) -> None:
        paths = app.openapi()["paths"]
        for path in (
            "/",
            "/health",
            "/api/auth/register",
            "/api/auth/login",
            "/api/dashboard",
            "/api/model-profiles",
            "/api/documents",
            "/api/documents/{document_id}/pages/{page_number}",
            "/api/documents/{document_id}/ask",
        ):
            self.assertIn(path, paths)
        for removed in (
            "/api/sessions",
            "/api/sessions/{conversation_id}/ask",
            "/api/sessions/{conversation_id}/ask/stream",
            "/api/papers",
            "/api/papers/upload",
            "/api/arxiv/ingest/proposals",
            "/api/arxiv/ingest/confirm",
        ):
            self.assertNotIn(removed, paths)


if __name__ == "__main__":
    unittest.main()
