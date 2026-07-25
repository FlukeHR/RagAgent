from __future__ import annotations

import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory

from agent.conversation import ConversationState
from agent.memory import ConversationMemory
from config.settings import load_settings


class MemoryTests(unittest.TestCase):
    def test_save_load_delete(self) -> None:
        with TemporaryDirectory() as directory:
            settings = load_settings()
            settings = replace(
                settings,
                harness=replace(
                    settings.harness,
                    memory_db_path=f"{directory}/sessions.sqlite3",
                ),
            )
            memory = ConversationMemory(settings)
            state = ConversationState(goal="goal", cited_papers=["p"])
            history = [{"role": "user", "content": "hello"}]
            memory.save("session-1", state, history)
            loaded_state, loaded_history = memory.load("session-1")
            self.assertEqual(loaded_state.goal, "goal")
            self.assertEqual(loaded_history, history)
            memory.delete("session-1")
            self.assertEqual(memory.load("session-1")[1], [])

    def test_invalid_session_id(self) -> None:
        settings = load_settings()
        memory = ConversationMemory(settings)
        with self.assertRaises(ValueError):
            memory.load("../escape")


if __name__ == "__main__":
    unittest.main()
