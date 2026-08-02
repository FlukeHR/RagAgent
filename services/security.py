from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import os
import re
import secrets
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from config.settings import BASE_DIR, Settings
from services.app_store import AppStore, AuthSessionRecord, UserRecord


_USERNAME = re.compile(r"^[A-Za-z0-9_\-.]{3,32}$")


@dataclass(frozen=True)
class LoginResult:
    """A newly authenticated user and the raw cookie value returned once."""

    user: UserRecord
    session: AuthSessionRecord
    token: str


class AuthService:
    """Local Argon2id password authentication with revocable server sessions."""

    def __init__(self, settings: Settings, store: AppStore) -> None:
        self.settings = settings
        self.store = store
        self.hasher = PasswordHasher()
        self._dummy_hash = self.hasher.hash("not-a-real-user-password")

    def register(
        self, username: str, password: str, display_name: str, user_agent: str
    ) -> LoginResult:
        """Create an account and return its first authenticated session."""

        normalized = username.strip()
        if not _USERNAME.fullmatch(normalized):
            raise ValueError("用户名需为 3–32 位字母、数字、点、横线或下划线")
        self.validate_password(password)
        name = display_name.strip()[:80] or normalized
        try:
            user = self.store.create_user(normalized, name, self.hasher.hash(password))
        except Exception as exc:
            if "UNIQUE constraint failed" in str(exc):
                raise ValueError("用户名已存在") from exc
            raise
        return self._new_session(user, user_agent)

    def login(self, username: str, password: str, user_agent: str) -> LoginResult:
        """Verify credentials, enforce lockout, and rotate into a new session."""

        user = self.store.get_user_by_username(username.strip())
        if user is None:
            try:
                self.hasher.verify(self._dummy_hash, password)
            except VerifyMismatchError:
                pass
            raise ValueError("用户名或密码错误")
        now = time.time()
        if user.locked_until > now:
            raise ValueError("登录尝试过多，请稍后再试")
        try:
            valid = self.hasher.verify(user.password_hash, password)
        except (VerifyMismatchError, InvalidHashError):
            valid = False
        if not valid:
            failures = user.failed_attempts + 1
            locked_until = (
                now + self.settings.app.login_lock_seconds
                if failures >= self.settings.app.max_login_failures
                else 0.0
            )
            self.store.update_login_state(
                user.user_id, failed_attempts=failures, locked_until=locked_until
            )
            raise ValueError("用户名或密码错误")
        self.store.update_login_state(user.user_id, failed_attempts=0, locked_until=0)
        refreshed = self.store.get_user(user.user_id)
        assert refreshed is not None
        if self.hasher.check_needs_rehash(refreshed.password_hash):
            self.store.update_password(refreshed.user_id, self.hasher.hash(password))
            refreshed = self.store.get_user(refreshed.user_id) or refreshed
        return self._new_session(refreshed, user_agent)

    def authenticate(self, raw_token: str | None) -> tuple[UserRecord, AuthSessionRecord] | None:
        """Resolve one raw session cookie without exposing it to persistent storage."""

        if not raw_token:
            return None
        session = self.store.get_auth_session(self.token_hash(raw_token))
        if session is None:
            return None
        user = self.store.get_user(session.user_id)
        if user is None:
            return None
        return user, session

    def change_password(
        self, user: UserRecord, current_password: str, new_password: str, token_hash: str
    ) -> None:
        """Change a password and revoke every other browser session."""

        try:
            if not self.hasher.verify(user.password_hash, current_password):
                raise ValueError("当前密码错误")
        except (VerifyMismatchError, InvalidHashError) as exc:
            raise ValueError("当前密码错误") from exc
        self.validate_password(new_password)
        self.store.update_password(user.user_id, self.hasher.hash(new_password))
        self.store.delete_other_auth_sessions(user.user_id, token_hash)

    def validate_password(self, password: str) -> None:
        if len(password) < self.settings.app.password_min_length:
            raise ValueError(
                f"密码至少需要 {self.settings.app.password_min_length} 个字符"
            )
        if len(password) > 256:
            raise ValueError("密码过长")

    def _new_session(self, user: UserRecord, user_agent: str) -> LoginResult:
        token = secrets.token_urlsafe(48)
        session = self.store.create_auth_session(
            user.user_id,
            self.token_hash(token),
            secrets.token_urlsafe(32),
            user_agent,
            time.time() + self.settings.app.session_ttl_seconds,
        )
        return LoginResult(user=user, session=session, token=token)

    @staticmethod
    def token_hash(raw_token: str) -> str:
        return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


class SecretBox:
    """AES-256-GCM encryption for user-supplied model credentials."""

    def __init__(self, settings: Settings, key: bytes | None = None) -> None:
        self.settings = settings
        self.key = key or self._load_or_create_key()
        if len(self.key) != 32:
            raise ValueError("model credential master key must contain 32 bytes")
        self.cipher = AESGCM(self.key)

    def encrypt(self, user_id: str, profile_id: str, secret: str) -> tuple[str, str]:
        """Encrypt a secret and return base64 nonce and ciphertext."""

        nonce = os.urandom(12)
        ciphertext = self.cipher.encrypt(
            nonce, secret.encode("utf-8"), self._aad(user_id, profile_id)
        )
        return self._encode(nonce), self._encode(ciphertext)

    def decrypt(
        self, user_id: str, profile_id: str, nonce: str, ciphertext: str
    ) -> str:
        """Decrypt a profile secret using its ownership-bound associated data."""

        plaintext = self.cipher.decrypt(
            self._decode(nonce),
            self._decode(ciphertext),
            self._aad(user_id, profile_id),
        )
        return plaintext.decode("utf-8")

    def _load_or_create_key(self) -> bytes:
        configured = os.getenv("PAPER_RAG_MASTER_KEY", "").strip()
        if configured:
            return self._decode(configured)
        root = Path(self.settings.app.secrets_root)
        root = root if root.is_absolute() else BASE_DIR / root
        root.mkdir(parents=True, exist_ok=True)
        path = root / "master.key"
        if path.exists():
            return self._decode(path.read_text(encoding="ascii").strip())
        key = os.urandom(32)
        encoded = self._encode(key)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(encoded)
        return key

    @staticmethod
    def _aad(user_id: str, profile_id: str) -> bytes:
        return f"paper-rag:model-profile:{user_id}:{profile_id}".encode("utf-8")

    @staticmethod
    def _encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii")

    @staticmethod
    def _decode(value: str) -> bytes:
        return base64.urlsafe_b64decode(value.encode("ascii"))


def validate_model_endpoint(value: str, settings: Settings) -> str:
    """Validate a model URL while remaining compatible with proxy fake-IP DNS."""

    raw = value.strip().rstrip("/")
    if not raw:
        raise ValueError("API Base 不能为空")
    allowed_local = {item.rstrip("/") for item in settings.app.allowed_local_llm_endpoints}
    if raw in allowed_local:
        return raw
    parsed = urlparse(raw)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("自定义 API Base 必须使用公开 HTTPS 地址")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API Base 不能包含账号、查询参数或片段")
    try:
        literal_ip = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal_ip = None
    if literal_ip is not None and not literal_ip.is_global:
        raise ValueError("API Base 不能使用未加入白名单的本机或私有 IP")
    if settings.app.enforce_public_dns_for_model_endpoints:
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)
            }
        except socket.gaierror as exc:
            raise ValueError("无法解析 API Base 主机名") from exc
        for address in addresses:
            if not ipaddress.ip_address(address).is_global:
                raise ValueError("API Base 域名解析到了本机或私有网络")
    clean = parsed._replace(path=parsed.path.rstrip("/"), params="", query="", fragment="")
    return urlunparse(clean)


def secure_compare(left: str, right: str) -> bool:
    """Constant-time comparison for CSRF tokens."""

    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))
