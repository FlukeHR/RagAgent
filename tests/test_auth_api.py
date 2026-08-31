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
from services.security import AuthService, SecretBox


class AuthApiTests(unittest.TestCase):
    """Check that the remaining credential APIs keep their security boundary."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        base = load_settings()
        self.settings = replace(
            base,
            app=replace(
                base.app,
                database_path=str(root / "app.sqlite3"),
                secrets_root=str(root / "secrets"),
            ),
        )
        self.store = AppStore(self.settings)
        self.auth = AuthService(self.settings, self.store)
        self.secret_box = SecretBox(self.settings, key=b"k" * 32)
        self.stack = ExitStack()
        for target, value in (
            ("api.dependencies.settings", self.settings),
            ("api.dependencies.auth_service", self.auth),
            ("api.auth_routes.auth_service", self.auth),
            ("api.auth_routes.store", self.store),
            ("api.app_routes.settings", self.settings),
            ("api.app_routes.store", self.store),
            ("api.app_routes.secret_box", self.secret_box),
        ):
            self.stack.enter_context(patch(target, return_value=value))
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

    def test_model_profile_write_requires_same_origin_and_csrf(self) -> None:
        auth = self.register(self.client, "alice")
        payload = {
            "name": "Local",
            "api_base": "https://example.com/v1",
            "model_name": "test-model",
            "api_key": "test-key",
        }
        missing_origin = self.client.post(
            "/api/model-profiles",
            headers={"X-CSRF-Token": str(auth["csrf_token"])},
            json=payload,
        )
        self.assertEqual(missing_origin.status_code, 403)

        created = self.client.post(
            "/api/model-profiles",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": str(auth["csrf_token"]),
            },
            json=payload,
        )
        self.assertEqual(created.status_code, 201, created.text)
        self.assertEqual(created.json()["key_last4"], "-key")

    def test_logout_revokes_cookie(self) -> None:
        auth = self.register(self.client, "alice")
        response = self.client.post(
            "/api/auth/logout",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": str(auth["csrf_token"]),
            },
        )
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.client.get("/api/auth/session").status_code, 401)


if __name__ == "__main__":
    unittest.main()
