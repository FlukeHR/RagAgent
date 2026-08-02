from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from config.settings import load_settings
from services.app_store import AppStore
from services.security import AuthService


class AuthApiTests(unittest.TestCase):
    """Exercise cookie, CSRF, revocation, and tenant boundaries through HTTP."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        base = load_settings()
        self.settings = replace(
            base,
            app=replace(
                base.app,
                database_path=str(root / "app.sqlite3"),
                users_root=str(root / "users"),
                secrets_root=str(root / "secrets"),
            ),
        )
        self.store = AppStore(self.settings)
        self.auth = AuthService(self.settings, self.store)
        self.stack = ExitStack()
        for target, value in (
            ("api.dependencies.settings", self.settings),
            ("api.dependencies.auth_service", self.auth),
            ("api.auth_routes.auth_service", self.auth),
            ("api.auth_routes.store", self.store),
            ("api.app_routes.store", self.store),
        ):
            self.stack.enter_context(patch(target, return_value=value))
        self.stack.enter_context(patch("api.main.ingest_manager", return_value=None))
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.stack.close()
        self.temporary.cleanup()

    def register(self, client: TestClient, username: str) -> dict[str, object]:
        response = client.post(
            "/api/auth/register",
            json={
                "username": username,
                "password": f"long-password-{username}",
                "display_name": username.title(),
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_write_requires_same_origin_and_session_csrf(self) -> None:
        auth = self.register(self.client, "alice")
        csrf = str(auth["csrf_token"])

        missing_origin = self.client.post(
            "/api/sessions",
            headers={"X-CSRF-Token": csrf},
            json={"title": "Private"},
        )
        self.assertEqual(missing_origin.status_code, 403)

        missing_csrf = self.client.post(
            "/api/sessions",
            headers={"Origin": "http://testserver"},
            json={"title": "Private"},
        )
        self.assertEqual(missing_csrf.status_code, 403)

        created = self.client.post(
            "/api/sessions",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": csrf,
            },
            json={"title": "Private"},
        )
        self.assertEqual(created.status_code, 201, created.text)

    def test_cross_user_resource_is_hidden_and_logout_revokes_cookie(self) -> None:
        first_auth = self.register(self.client, "alice")
        created = self.client.post(
            "/api/sessions",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": str(first_auth["csrf_token"]),
            },
            json={"title": "Alice only"},
        )
        conversation_id = created.json()["conversation_id"]

        with TestClient(app) as second:
            self.register(second, "bob")
            hidden = second.get(f"/api/sessions/{conversation_id}")
            self.assertEqual(hidden.status_code, 404)

        logged_out = self.client.post(
            "/api/auth/logout",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": str(first_auth["csrf_token"]),
            },
        )
        self.assertEqual(logged_out.status_code, 204)
        self.assertEqual(self.client.get("/api/auth/session").status_code, 401)


if __name__ == "__main__":
    unittest.main()
