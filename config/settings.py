from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    name: str
    data_root: str


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
class RetrievalConfig:
    low_confidence_threshold: float       # 强阈值：最高分经 sigmoid 归一化后的相关概率需 ≥ 此值
    max_corrections: int
    weak_confidence_threshold: float = 0.35  # 弱阈值：相关概率 ≥ 此值才算一条"够格"证据
    min_confident_sources: int = 1           # 数量判据：够格证据至少这么多条，否则判低置信
    answerability_min_sources: int = 1       # 硬闸：至少需要这么多有效来源才能生成实质答案
    answerability_min_score: float = 0.0     # 硬闸：可比分数需达到该 logit/融合分；None 来源按有效工具来源处理
    answerability_require_citation: bool = True


@dataclass
class PDFParseConfig:
    provider: str = "pymupdf"
    auto_ocr: bool = False
    timeout_seconds: float = 30.0


@dataclass
class ImageSearchConfig:
    enabled: bool = True
    max_pages: int = 80
    max_side: int = 256


@dataclass
class LLMConfig:
    model_name: str
    max_tokens: int
    max_tool_iters: int
    openai_api_base: str
    openai_api_key: str


@dataclass
class ArxivConfig:
    max_results: int
    max_ingest_papers: int = 3            # ingest_arxiv_papers 每轮下载上限（有界，护栏 #2）
    max_pdf_mb: float = 30.0              # 单篇 PDF 体积上限，超出跳过
    ingest_timeout_seconds: float = 120.0  # 下载+增量嵌入+重检索专属超时（含网络/推理）
    max_papers: int = 200                 # 在线入库论文容量上限；手动论文不自动淘汰；0 = 不限
    max_age_days: int = 0                 # 超过这么多天未被检索命中即淘汰；0 = 不按龄期淘汰


@dataclass
class HarnessConfig:
    tool_timeout_seconds: float
    tool_max_retries: int
    token_budget: int


@dataclass
class Settings:
    project: ProjectConfig
    index: IndexConfig
    embedding: EmbeddingConfig
    rerank: RerankConfig
    retrieval: RetrievalConfig
    pdf_parse: PDFParseConfig
    image_search: ImageSearchConfig
    llm: LLMConfig
    arxiv: ArxivConfig
    harness: HarnessConfig


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
MODELS_DIR = BASE_DIR / "models"


def resolve_model_path(name: str) -> str:
    """优先返回项目 `models/` 下的本地模型副本路径，否则原样返回（HF 名 / 路径）。

    config 里仍写 HF 全名（如 `sentence-transformers/all-MiniLM-L6-v2`）；若
    `models/all-MiniLM-L6-v2` 存在则用本地副本（离线、可移植），缺失则回退在线下载。
    """
    local = MODELS_DIR / Path(name).name
    if local.is_dir():
        return str(local)
    return name


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
        retrieval=RetrievalConfig(**raw["retrieval"]),
        pdf_parse=PDFParseConfig(**raw.get("pdf_parse", {})),
        image_search=ImageSearchConfig(**raw.get("image_search", {})),
        llm=LLMConfig(**raw["llm"]),
        arxiv=ArxivConfig(**raw["arxiv"]),
        harness=HarnessConfig(**raw["harness"]),
    )
