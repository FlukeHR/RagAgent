from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from agent.evidence import AgentAnswer
from config.settings import load_settings
from services.app_store import AppStore
from services.security import AuthService, SecretBox


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
        self.secret_box = SecretBox(self.settings, key=b"k" * 32)
        self.stack = ExitStack()
        for target, value in (
            ("api.dependencies.settings", self.settings),
            ("api.dependencies.auth_service", self.auth),
            ("api.auth_routes.auth_service", self.auth),
            ("api.auth_routes.store", self.store),
            ("api.app_routes.store", self.store),
            ("api.app_routes.secret_box", self.secret_box),
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

    def test_streaming_answer_emits_tokens_and_persists_final_message(self) -> None:
        class FakeAgent:
            @staticmethod
            def ask(*args: object, **kwargs: object) -> AgentAnswer:
                del args
                sink = kwargs["token_sink"]
                assert callable(sink)
                sink("快速")
                sink("回答")
                return AgentAnswer("快速回答", steps=["fast"], trace=[])

        class FakePool:
            @staticmethod
            def schedule_prewarm(*args: object) -> None:
                del args

            @staticmethod
            def get(*args: object) -> FakeAgent:
                del args
                return FakeAgent()

        auth = self.register(self.client, "streamer")
        headers = {
            "Origin": "http://testserver",
            "X-CSRF-Token": str(auth["csrf_token"]),
        }
        pool = FakePool()
        with patch("api.app_routes.agent_pool", return_value=pool):
            profile = self.client.post(
                "/api/model-profiles",
                headers=headers,
                json={
                    "name": "Fast",
                    "api_base": "https://example.com/v1",
                    "model_name": "fast-model",
                    "api_key": "test-key",
                    "is_default": True,
                },
            )
            self.assertEqual(profile.status_code, 201, profile.text)
            session = self.client.post(
                "/api/sessions",
                headers=headers,
                json={
                    "title": "Stream",
                    "model_profile_id": profile.json()["profile_id"],
                },
            )
            conversation_id = session.json()["conversation_id"]
            with self.client.stream(
                "POST",
                f"/api/sessions/{conversation_id}/ask/stream",
                headers=headers,
                json={"question": "question"},
            ) as response:
                self.assertEqual(response.status_code, 200)
                events = [json.loads(line) for line in response.iter_lines() if line]

        self.assertEqual(
            [event["type"] for event in events],
            ["start", "token", "token", "final"],
        )
        self.assertEqual(events[-1]["result"]["answer"], "快速回答")
        saved = self.client.get(f"/api/sessions/{conversation_id}").json()
        self.assertEqual(saved["messages"][-1]["content"], "快速回答")


if __name__ == "__main__":
    unittest.main()
