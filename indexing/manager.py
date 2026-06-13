from __future__ import annotations

from pathlib import Path

from config.settings import BASE_DIR, Settings


class IndexManager:
    """管理论文集合（collection = data/papers 下的子目录）。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.index_root = BASE_DIR / settings.index.index_root
        self.data_root = BASE_DIR / settings.project.data_root

    def get_collection_data_path(self, collection: str) -> Path:
        return self.data_root / collection

    def get_collection_index_path(self, collection: str) -> Path:
        return self.index_root / collection

    def list_collections(self) -> list[str]:
        if not self.data_root.exists():
            return []
        excluded = {"__pycache__"}
        return sorted(
            p.name
            for p in self.data_root.iterdir()
            if p.is_dir() and p.name not in excluded and not p.name.startswith(".")
        )
