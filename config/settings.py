from __future__ import annotations

from dataclasses import dataclass, field
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
    claim_support_min_overlap: float = 0.12


@dataclass
class MinerUConfig:
    api_base_url: str = "http://127.0.0.1:8001"
    version: str = "3.4.4"
    backend: str = "hybrid-engine"
    effort: str = "high"
    supported_version_prefix: str = "3.4."
    connect_timeout_seconds: float = 5.0
    request_timeout_seconds: float = 30.0
    max_request_retries: int = 2
    parse_timeout_seconds: float = 900.0
    poll_interval_seconds: float = 2.0
    max_pdf_mb: float = 30.0
    max_pages: int = 300
    max_output_mb: float = 250.0
    max_archive_entries: int = 3000
    cache_root: str = "./data/mineru-cache"
    proposal_ttl_seconds: int = 900
    job_db_path: str = "./data/ingest_jobs.sqlite3"
    max_pending_jobs: int = 8
    max_concurrent_jobs: int = 1


@dataclass
class LLMConfig:
    model_name: str
    max_tokens: int
    openai_api_base: str
    openai_api_key: str
    request_timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 30.0
    max_retries: int = 2


@dataclass
class ArxivConfig:
    max_results: int
    max_ingest_papers: int = 1            # 每个已确认 job 最多一篇论文
    max_pdf_mb: float = 30.0              # 单篇 PDF 体积上限，超出跳过
    max_papers: int = 200                 # 在线入库论文容量上限；手动论文不自动淘汰；0 = 不限
    max_age_days: int = 0                 # 超过这么多天未被检索命中即淘汰；0 = 不按龄期淘汰
    request_timeout_seconds: float = 60.0


@dataclass
class AgentConfig:
    """Bounded LangChain runtime, conversation, and evidence settings."""

    token_budget: int
    tool_result_max_chars: int = 8000
    source_snippet_chars: int = 600
    history_max_messages: int = 8
    history_max_chars: int = 8000
    history_summary_max_chars: int = 1600
    memory_ttl_seconds: int = 604800
    memory_max_sessions: int = 500
    memory_db_path: str = "./data/sessions.sqlite3"
    recent_history_messages: int = 6
    max_model_calls: int = 3
    max_graph_steps: int = 32
    max_tool_calls: int = 4
    max_local_search_calls: int = 1
    max_inspect_calls: int = 1
    max_arxiv_search_calls: int = 2
    max_total_sources: int = 8
    max_total_tool_result_chars: int = 16000
    final_max_sources: int = 5
    final_max_sources_per_paper: int = 3
    final_reuse_max_chars: int = 8000
    trace_value_max_chars: int = 500
    fast_local_enabled: bool = True
    fast_local_min_confidence: float = 0.5
    fast_local_max_sources: int = 3
    max_forced_tool_escalations: int = 1
    prewarm_on_startup: bool = True


@dataclass
class AppConfig:
    """Local multi-user application storage, authentication, and quota settings."""

    database_path: str = "./data/app.sqlite3"
    users_root: str = "./data/users"
    secrets_root: str = "./data/secrets"
    session_cookie_name: str = "paper_rag_session"
    session_ttl_seconds: int = 604800
    password_min_length: int = 10
    max_login_failures: int = 5
    login_lock_seconds: int = 900
    max_papers_per_user: int = 200
    max_pending_jobs_per_user: int = 3
    max_model_profiles_per_user: int = 10
    enforce_public_dns_for_model_endpoints: bool = False
    allowed_local_llm_endpoints: list[str] = field(default_factory=list)


@dataclass
class Settings:
    project: ProjectConfig
    index: IndexConfig
    embedding: EmbeddingConfig
    rerank: RerankConfig
    retrieval: RetrievalConfig
    mineru: MinerUConfig
    llm: LLMConfig
    arxiv: ArxivConfig
    agent: AgentConfig
    app: AppConfig


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
        mineru=MinerUConfig(**raw.get("mineru", {})),
        llm=LLMConfig(**raw["llm"]),
        arxiv=ArxivConfig(**raw["arxiv"]),
        agent=AgentConfig(**raw["agent"]),
        app=AppConfig(**raw.get("app", {})),
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
    if (
        settings.agent.max_model_calls <= 0
        or settings.agent.max_graph_steps <= 0
        or settings.agent.token_budget <= 0
    ):
        raise ValueError("agent model-call and token budgets must be positive")
    if (
        settings.llm.request_timeout_seconds <= 0
        or settings.llm.connect_timeout_seconds <= 0
        or settings.llm.max_retries < 0
    ):
        raise ValueError("LLM timeout/retry settings are invalid")
    for name in (
        "tool_result_max_chars",
        "source_snippet_chars",
        "history_max_messages",
        "history_max_chars",
        "recent_history_messages",
    ):
        if getattr(settings.agent, name) <= 0:
            raise ValueError(f"agent.{name} must be positive")
    if settings.arxiv.max_ingest_papers <= 0 or settings.arxiv.max_pdf_mb <= 0:
        raise ValueError("arxiv ingest limits must be positive")
    if settings.arxiv.request_timeout_seconds <= 0:
        raise ValueError("arxiv.request_timeout_seconds must be positive")
    if settings.embedding.fallback_dimension <= 0:
        raise ValueError("embedding.fallback_dimension must be positive")
    if settings.mineru.backend != "hybrid-engine" or settings.mineru.effort != "high":
        raise ValueError("MinerU baseline must use hybrid-engine with high effort")
    if not settings.mineru.version.startswith(settings.mineru.supported_version_prefix):
        raise ValueError("MinerU version must match supported_version_prefix")
    if settings.mineru.max_pages <= 0 or settings.mineru.max_pdf_mb <= 0:
        raise ValueError("MinerU PDF limits must be positive")
    if settings.mineru.max_request_retries < 0:
        raise ValueError("MinerU request retries cannot be negative")
    if settings.mineru.max_concurrent_jobs != 1:
        raise ValueError("MinerU ingestion must use one bounded worker")
    for name in (
        "max_tool_calls",
        "max_local_search_calls",
        "max_inspect_calls",
        "max_arxiv_search_calls",
        "max_total_sources",
        "max_total_tool_result_chars",
        "final_max_sources",
        "final_max_sources_per_paper",
        "final_reuse_max_chars",
        "trace_value_max_chars",
        "fast_local_max_sources",
        "max_forced_tool_escalations",
    ):
        if getattr(settings.agent, name) <= 0:
            raise ValueError(f"agent.{name} must be positive")
    if settings.agent.final_max_sources_per_paper > settings.agent.final_max_sources:
        raise ValueError(
            "agent.final_max_sources_per_paper cannot exceed final_max_sources"
        )
    if not 0.0 <= settings.agent.fast_local_min_confidence <= 1.0:
        raise ValueError("agent.fast_local_min_confidence must be in [0, 1]")
    if not 0.0 <= settings.retrieval.claim_support_min_overlap <= 1.0:
        raise ValueError("retrieval.claim_support_min_overlap must be in [0, 1]")
    if not 0.0 <= settings.retrieval.answerability_min_confidence <= 1.0:
        raise ValueError("retrieval.answerability_min_confidence must be in [0, 1]")
    if settings.app.session_ttl_seconds <= 0:
        raise ValueError("app.session_ttl_seconds must be positive")
    if settings.app.password_min_length < 8:
        raise ValueError("app.password_min_length must be at least 8")
    for name in (
        "max_login_failures",
        "login_lock_seconds",
        "max_papers_per_user",
        "max_pending_jobs_per_user",
        "max_model_profiles_per_user",
    ):
        if getattr(settings.app, name) <= 0:
            raise ValueError(f"app.{name} must be positive")
