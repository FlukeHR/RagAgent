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
    rrf_k: int = 60
    max_chunks_per_parent: int = 2


@dataclass
class EmbeddingConfig:
    model_name: str
    use_sentence_transformers: bool
    fallback_dimension: int = 384


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
    answerability_min_confidence: float = 0.25
    cjk_ngram_size: int = 2
    deduplicate_evidence: bool = True
    conflict_detection: bool = True


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
    max_query_base64_chars: int = 2_000_000


@dataclass
class LLMConfig:
    model_name: str
    max_tokens: int
    max_tool_iters: int
    openai_api_base: str
    openai_api_key: str
    request_timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 30.0
    max_retries: int = 2


@dataclass
class ArxivConfig:
    max_results: int
    max_ingest_papers: int = 3            # ingest_arxiv_papers 每轮下载上限（有界，护栏 #2）
    max_pdf_mb: float = 30.0              # 单篇 PDF 体积上限，超出跳过
    ingest_timeout_seconds: float = 120.0  # 下载+增量嵌入+重检索专属超时（含网络/推理）
    max_papers: int = 200                 # 在线入库论文容量上限；手动论文不自动淘汰；0 = 不限
    max_age_days: int = 0                 # 超过这么多天未被检索命中即淘汰；0 = 不按龄期淘汰
    request_timeout_seconds: float = 60.0


@dataclass
class HarnessConfig:
    tool_timeout_seconds: float
    tool_max_retries: int
    token_budget: int
    tool_result_max_chars: int = 24000
    source_snippet_chars: int = 600
    image_base64_chars: int = 12000
    history_max_messages: int = 12
    history_max_chars: int = 12000
    history_summary_max_chars: int = 2000
    context_max_chars: int = 60000
    clarification_enabled: bool = True
    memory_ttl_seconds: int = 604800
    memory_max_sessions: int = 500
    memory_db_path: str = "./data/sessions.sqlite3"
    clarification_min_chars: int = 6
    recent_history_messages: int = 8
    claim_support_min_overlap: float = 0.12
    pdf_page_max_side: int = 2400
    pdf_region_max_side: int = 1600


@dataclass
class EvaluationConfig:
    """只用于离线评测的价格参数；默认 0 表示只统计 token、不换算费用。"""

    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0


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
    evaluation: EvaluationConfig


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

    settings = Settings(
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
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )
    _validate_settings(settings)
    return settings


def _validate_settings(settings: Settings) -> None:
    """启动时校验会影响安全、资源上限和索引一致性的配置。"""

    if settings.index.chunk_size <= 0:
        raise ValueError("index.chunk_size must be positive")
    if not 0 <= settings.index.chunk_overlap < settings.index.chunk_size:
        raise ValueError("index.chunk_overlap must be >= 0 and smaller than chunk_size")
    if settings.index.top_k_recall <= 0 or settings.index.top_n_rerank <= 0:
        raise ValueError("index top-k values must be positive")
    if settings.index.top_n_rerank > settings.index.top_k_recall:
        raise ValueError("index.top_n_rerank cannot exceed top_k_recall")
    if settings.llm.max_tool_iters <= 0 or settings.harness.token_budget <= 0:
        raise ValueError("LLM iteration and token budgets must be positive")
    if (
        settings.llm.request_timeout_seconds <= 0
        or settings.llm.connect_timeout_seconds <= 0
        or settings.llm.max_retries < 0
    ):
        raise ValueError("LLM timeout/retry settings are invalid")
    for name in (
        "tool_result_max_chars",
        "source_snippet_chars",
        "image_base64_chars",
        "history_max_messages",
        "history_max_chars",
        "context_max_chars",
        "recent_history_messages",
        "pdf_page_max_side",
        "pdf_region_max_side",
    ):
        if getattr(settings.harness, name) <= 0:
            raise ValueError(f"harness.{name} must be positive")
    if settings.arxiv.max_ingest_papers <= 0 or settings.arxiv.max_pdf_mb <= 0:
        raise ValueError("arxiv ingest limits must be positive")
    if settings.arxiv.request_timeout_seconds <= 0:
        raise ValueError("arxiv.request_timeout_seconds must be positive")
    if settings.embedding.fallback_dimension <= 0:
        raise ValueError("embedding.fallback_dimension must be positive")
    if settings.image_search.max_query_base64_chars <= 0:
        raise ValueError("image_search.max_query_base64_chars must be positive")
    if not 0.0 <= settings.harness.claim_support_min_overlap <= 1.0:
        raise ValueError("harness.claim_support_min_overlap must be in [0, 1]")
    if not 0.0 <= settings.retrieval.answerability_min_confidence <= 1.0:
        raise ValueError("retrieval.answerability_min_confidence must be in [0, 1]")
