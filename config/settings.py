from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    name: str
    data_root: str
    default_repo: str


@dataclass
class IndexConfig:
    index_root: str
    chunk_size: int
    chunk_overlap: int
    top_k_recall: int
    top_n_rerank: int


@dataclass
class EmbeddingConfig:
    model_name: str
    use_sentence_transformers: bool


@dataclass
class RerankConfig:
    model_name: str
    use_cross_encoder: bool


@dataclass
class LLMConfig:
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    openai_api_base: str
    openai_api_key: str


@dataclass
class Settings:
    project: ProjectConfig
    index: IndexConfig
    embedding: EmbeddingConfig
    rerank: RerankConfig
    llm: LLMConfig


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = _load_yaml(config_path)

    return Settings(
        project=ProjectConfig(**raw["project"]),
        index=IndexConfig(**raw["index"]),
        embedding=EmbeddingConfig(**raw["embedding"]),
        rerank=RerankConfig(**raw["rerank"]),
        llm=LLMConfig(**raw["llm"]),
    )
