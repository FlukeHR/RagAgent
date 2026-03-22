from __future__ import annotations

from pathlib import Path

from config.settings import BASE_DIR, Settings


class IndexManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.index_root = BASE_DIR / settings.index.index_root
        self.data_root = BASE_DIR / settings.project.data_root

    def get_repo_data_path(self, repo_name: str) -> Path:
        return self.data_root / repo_name

    def get_repo_index_path(self, repo_name: str) -> Path:
        return self.index_root / repo_name

    def list_repositories(self) -> list[str]:
        if not self.data_root.exists():
            return []
        excluded = {"indexes", "__pycache__"}
        return sorted(
            [
                p.name
                for p in self.data_root.iterdir()
                if p.is_dir() and p.name not in excluded and not p.name.startswith(".")
            ]
        )
