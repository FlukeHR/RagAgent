from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent.graph import PaperRAGAgent
from agent.memory import ConversationState
from config.settings import BASE_DIR, Settings
from services.security import SecretBox


_USER_ID = re.compile(r"^[a-f0-9]{32}$")


@dataclass(frozen=True)
class UserPaths:
    """Validated filesystem locations owned by one local application user."""

    root: Path
    papers: Path
    indexes: Path
    mineru_cache: Path


class StatelessAgentMemory:
    """Prevent the scoped Agent from creating a second conversation database."""

    @staticmethod
    def load(session_id: str) -> tuple[ConversationState, list[dict[str, Any]]]:
        del session_id
        return ConversationState(), []

    @staticmethod
    def save(
        session_id: str,
        state: ConversationState,
        history: list[dict[str, Any]],
    ) -> None:
        del session_id, state, history

    @staticmethod
    def delete(session_id: str) -> None:
        del session_id


def user_paths(settings: Settings, user_id: str, *, create: bool = True) -> UserPaths:
    """Resolve a UUID-scoped user root without accepting username/path input."""

    if not _USER_ID.fullmatch(user_id):
        raise ValueError("invalid user_id")
    configured = Path(settings.app.users_root)
    users_root = (configured if configured.is_absolute() else BASE_DIR / configured).resolve()
    root = (users_root / user_id).resolve()
    if root.parent != users_root:
        raise ValueError("user path escapes users_root")
    result = UserPaths(
        root=root,
        papers=root / "papers",
        indexes=root / "indexes",
        mineru_cache=root / "mineru-cache",
    )
    if create:
        for path in (result.papers, result.indexes, result.mineru_cache):
            path.mkdir(parents=True, exist_ok=True)
    return result


def scoped_settings(
    settings: Settings,
    user_id: str,
    *,
    model_name: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> Settings:
    """Clone centralized settings with only user-owned runtime paths changed."""

    paths = user_paths(settings, user_id)
    return replace(
        settings,
        project=replace(settings.project, data_root=str(paths.papers)),
        index=replace(settings.index, index_root=str(paths.indexes)),
        mineru=replace(settings.mineru, cache_root=str(paths.mineru_cache)),
        llm=replace(
            settings.llm,
            model_name=model_name or settings.llm.model_name,
            openai_api_base=api_base if api_base is not None else settings.llm.openai_api_base,
            openai_api_key=api_key if api_key is not None else settings.llm.openai_api_key,
        ),
    )


class AgentPool:
    """Bounded process-local cache of user/profile scoped paper agents."""

    def __init__(self, settings: Settings, secret_box: SecretBox, max_size: int = 16) -> None:
        self.settings = settings
        self.secret_box = secret_box
        self.max_size = max_size
        self._agents: OrderedDict[tuple[str, str, float], PaperRAGAgent] = OrderedDict()
        self._lock = threading.RLock()
        self._prewarming: set[tuple[str, str, float]] = set()
        self._prewarmed: set[tuple[str, str, float]] = set()

    def get(self, user_id: str, profile: dict[str, Any]) -> PaperRAGAgent:
        """Return a scoped Agent, decrypting its credential only in process memory."""

        profile_id = str(profile["profile_id"])
        key = (user_id, profile_id, float(profile["updated_at"]))
        with self._lock:
            existing = self._agents.pop(key, None)
            if existing is not None:
                self._agents[key] = existing
                return existing
            api_key = self.secret_box.decrypt(
                user_id,
                profile_id,
                str(profile["key_nonce"]),
                str(profile["key_ciphertext"]),
            )
            configured = scoped_settings(
                self.settings,
                user_id,
                model_name=str(profile["model_name"]),
                api_base=str(profile["api_base"]),
                api_key=api_key,
            )
            agent = PaperRAGAgent(configured, memory=StatelessAgentMemory())  # type: ignore[arg-type]
            self._agents[key] = agent
            while len(self._agents) > self.max_size:
                self._agents.popitem(last=False)
            return agent

    def schedule_prewarm(self, user_id: str, profile: dict[str, Any]) -> None:
        """Warm one scoped Agent in the background before its first question."""

        if not self.settings.agent.prewarm_on_startup:
            return
        key = (user_id, str(profile["profile_id"]), float(profile["updated_at"]))
        with self._lock:
            if key in self._prewarming or key in self._prewarmed:
                return
            self._prewarming.add(key)

        def run() -> None:
            succeeded = False
            try:
                self.get(user_id, profile).prewarm()
                succeeded = True
            except Exception:  # noqa: BLE001 - prewarm is opportunistic
                pass
            finally:
                with self._lock:
                    self._prewarming.discard(key)
                    if succeeded:
                        self._prewarmed.add(key)

        threading.Thread(
            target=run,
            name=f"rag-prewarm-{key[1][:8]}",
            daemon=True,
        ).start()

    def invalidate(self, user_id: str, profile_id: str | None = None) -> None:
        """Evict cached agents after a key, model, or index-affecting change."""

        with self._lock:
            for key in list(self._agents):
                if key[0] == user_id and (profile_id is None or key[1] == profile_id):
                    self._agents.pop(key, None)
            self._prewarmed = {
                key
                for key in self._prewarmed
                if not (
                    key[0] == user_id
                    and (profile_id is None or key[1] == profile_id)
                )
            }
