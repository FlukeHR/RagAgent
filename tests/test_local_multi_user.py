from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cryptography.exceptions import InvalidTag

from api.schemas import ModelProfileResponse, SourceItem
from config.settings import load_settings
from services.app_store import AppStore
from services.security import AuthService, SecretBox, validate_model_endpoint
from services.user_scope import user_paths


class LocalMultiUserTests(unittest.TestCase):
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
                password_min_length=10,
                max_login_failures=2,
                allowed_local_llm_endpoints=["http://127.0.0.1:11434/v1"],
            ),
        )
        self.store = AppStore(self.settings)
        self.auth = AuthService(self.settings, self.store)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_password_session_and_cross_user_isolation(self) -> None:
        first = self.auth.register("alice", "long-password-a", "Alice", "test")
        second = self.auth.register("bob", "long-password-b", "Bob", "test")

        self.assertNotIn(first.token, first.session.token_hash)
        authenticated = self.auth.authenticate(first.token)
        self.assertIsNotNone(authenticated)
        conversation = self.store.create_conversation(
            first.user.user_id, "Private", None
        )
        self.assertIsNotNone(
            self.store.get_conversation(
                first.user.user_id, str(conversation["conversation_id"])
            )
        )
        self.assertIsNone(
            self.store.get_conversation(
                second.user.user_id, str(conversation["conversation_id"])
            )
        )

    def test_login_lockout_and_password_change_revoke_other_sessions(self) -> None:
        registered = self.auth.register(
            "researcher", "long-password-a", "Researcher", "first"
        )
        other = self.auth.login("researcher", "long-password-a", "second")
        with self.assertRaisesRegex(ValueError, "用户名或密码错误"):
            self.auth.login("researcher", "wrong", "bad")
        with self.assertRaisesRegex(ValueError, "用户名或密码错误"):
            self.auth.login("researcher", "wrong", "bad")
        with self.assertRaisesRegex(ValueError, "登录尝试过多"):
            self.auth.login("researcher", "long-password-a", "blocked")

        self.store.update_login_state(
            registered.user.user_id, failed_attempts=0, locked_until=0
        )
        refreshed = self.store.get_user(registered.user.user_id)
        assert refreshed is not None
        self.auth.change_password(
            refreshed,
            "long-password-a",
            "long-password-new",
            self.auth.token_hash(registered.token),
        )
        self.assertIsNone(self.auth.authenticate(other.token))
        self.assertIsNotNone(self.auth.authenticate(registered.token))

    def test_model_key_is_encrypted_and_bound_to_owner_and_profile(self) -> None:
        user = self.auth.register("alice", "long-password-a", "Alice", "test").user
        box = SecretBox(self.settings, key=b"k" * 32)
        profile_id = "a" * 32
        nonce, ciphertext = box.encrypt(user.user_id, profile_id, "secret-api-key")
        self.assertNotIn("secret-api-key", ciphertext)
        self.assertEqual(
            box.decrypt(user.user_id, profile_id, nonce, ciphertext),
            "secret-api-key",
        )
        with self.assertRaises(InvalidTag):
            box.decrypt("b" * 32, profile_id, nonce, ciphertext)

        stored = self.store.create_model_profile(
            user.user_id,
            {
                "profile_id": profile_id,
                "name": "Local",
                "provider": "openai-compatible",
                "api_base": "http://127.0.0.1:11434/v1",
                "model_name": "demo",
                "key_nonce": nonce,
                "key_ciphertext": ciphertext,
                "key_last4": "-key",
                "is_default": True,
            },
        )
        public = ModelProfileResponse(
            **{key: stored[key] for key in ModelProfileResponse.model_fields}
        ).model_dump()
        self.assertNotIn("key_ciphertext", public)
        self.assertNotIn("key_nonce", public)

    def test_user_paths_and_safe_source_response_do_not_leak_local_path(self) -> None:
        first = self.auth.register("alice", "long-password-a", "Alice", "test").user
        second = self.auth.register("bob", "long-password-b", "Bob", "test").user
        first_paths = user_paths(self.settings, first.user_id)
        second_paths = user_paths(self.settings, second.user_id)
        self.assertNotEqual(first_paths.papers, second_paths.papers)
        self.assertEqual(first_paths.root.parent, second_paths.root.parent)

        source = SourceItem(
            id="S1",
            paper_id="paper",
            paper_title="Paper",
            section="Methods",
            source=str(first_paths.papers / "paper.pdf"),
            source_kind="library_pdf",
            preview_kind="pdf",
        )
        self.assertNotIn("source", source.model_dump())

    def test_local_model_endpoint_requires_exact_allowlist(self) -> None:
        self.assertEqual(
            validate_model_endpoint("http://127.0.0.1:11434/v1", self.settings),
            "http://127.0.0.1:11434/v1",
        )
        with self.assertRaisesRegex(ValueError, "公开 HTTPS"):
            validate_model_endpoint("http://127.0.0.1:8001", self.settings)

    def test_proxy_fake_ip_does_not_block_https_hostname_by_default(self) -> None:
        fake_ip_result = [(2, 1, 6, "", ("198.18.0.112", 443))]
        with patch("services.security.socket.getaddrinfo", return_value=fake_ip_result):
            self.assertEqual(
                validate_model_endpoint("https://api.deepseek.com", self.settings),
                "https://api.deepseek.com",
            )

            strict = replace(
                self.settings,
                app=replace(
                    self.settings.app,
                    enforce_public_dns_for_model_endpoints=True,
                ),
            )
            with self.assertRaisesRegex(ValueError, "域名解析到了"):
                validate_model_endpoint("https://api.deepseek.com", strict)

    def test_explicit_private_ip_still_requires_exact_allowlist(self) -> None:
        with self.assertRaisesRegex(ValueError, "本机或私有 IP"):
            validate_model_endpoint("https://127.0.0.1/v1", self.settings)


if __name__ == "__main__":
    unittest.main()
