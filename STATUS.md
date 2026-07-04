# STATUS.md — Paper RAG Agent 当前进展

> 改动频繁的东西放这里：当前状态 / 待办 / 已知问题 / 评估基线。
> 稳定事实（架构、命令、规范、护栏）见 `CLAUDE.md`。
> 最近盘点：2026-06-21。

## 一句话现状

检索 + agentic 循环 + API + 前端**端到端跑通**，全部核心不变量与护栏已落地（agentic 主路径 + 降级单跳 RAG 均接引用回查与二次检索）。能力已覆盖：
多源检索（本地稠密+BM25 RRF→CrossEncoder 重排）、引用回查、低置信二次检索、harness 护栏（schema 校验/超时重试/token 预算/结构化 trace）、
多轮对话（历史注入+指代消解）、多会话前端（自动标题+PDF 预览高亮+trace 可视化）、增量索引、arXiv 全文按需入库（同步单轮）+ 集合容量治理、
本地模型副本、生成侧 RAGAS 评估（显式授权）、CI（ruff+pytest）。
`pytest tests/` **48 用例全绿**（全离线 mock，不打网络/arXiv/LLM API）。

## 评估基线（仅检索侧，统一用官方集）

> 已删除手造的 `eval.py`（4 条 demo 样本、数据文件早被删、非官方）。评测一律用官方集。

- **QASPER**（`python3 evaluation/eval_qasper.py`，50 篇 / 144 题，CE+BM25）：
  Hit@5 **0.7222**、MRR **0.4959**、nDCG@5 **0.4185**、Recall@5 **0.4811**。
- 改检索 / agent 逻辑后必须重跑并与基线比对，**指标回退不算完成**。
- **生成侧（RAGAS，已接入官方 QASPER）**：`RAG_EVAL_ALLOW_API=1 python3 evaluation/eval_generation.py --qasper --limit 20 --record`。
  QASPER 是单文档 QA：在每题所属论文内检索→生成，以 gold answer 为 reference，跑 faithfulness / answer relevancy /
  context precision/recall / answer correctness（有参考指标，版本支持则自动加）。judge 复用项目 LLM 后端、嵌入用本地模型。
  需显式授权、独立于单测，尚未跑出基线数值（待填）。也支持自定义集 JSON（走真实 agent 生成）。

> **低置信阈值标定结论**（`eval_qasper.py --sweep`，144 题）：CrossEncoder 分数几乎不能区分检索命中/未命中
> （命中均值 0.588 vs 未命中 0.598，Youden's J≈0）。即「靠单分数判低置信」信号本就弱，故保持保守默认
> `low_confidence_threshold=0.5`（概率口径），真正的安全网是引用回查 + 数量判据，而非原始分数。

## 评估历史记录（2026-06-21）

- `evaluation/results_log.py`：每次带 `--record` 跑评估追加一行到 `evaluation/results/history.jsonl`
  （时间 + git 提交/分支/dirty + 配置快照 + 指标）。`eval_qasper.py --record` / `eval_generation.py --yes --record` 接入；
  `python3 evaluation/results_log.py --compare` 看最新两次增减（回归视角）。已种入 QASPER 检索基线一条。
- 生成侧评测集：**手造的 `generation_eval.json` 已删**，改用**官方 QASPER**（`--qasper`，单文档内检索，无需另建索引）。
  `qasper_reference` 从标注取 gold answer 作 reference；`collect_samples_qasper` 在论文段落内检索→生成，有离线单测。

## 待办 / 缺口

1. **生成侧 RAGAS 基线未落数**：脚本可跑（QASPER 已接入、ragas×langchain 垫片已修），需用真实 API 跑
   `--qasper --limit 20 --record` 落一条基线填进上面。
2. **eval 门禁未入 CI**：CI 已跑 ruff + pytest；eval 因需模型/数据/API 仍本地手动跑（mypy 已接入但非阻断）。

## 已知问题 / 限制

- **会话仅存浏览器 localStorage**（`rag_sessions` / `rag_active`）：无跨设备 / 后端持久化（已与用户确认的设计取舍）。
- **prune 跨进程**：集合级锁只保进程内并发；手动 `python3 indexing/prune.py` 是独立进程，建议在服务空闲时跑。
- **低置信阈值与 rerank 模式耦合**：默认阈值按 CrossEncoder 开（CE logit→sigmoid）标定；CE 关（降级 token+余弦）需另行重标。
- **额外 LLM 调用未计入 token 预算**：查询改写（仅有历史时）、标题生成（仅首轮）各一次轻量调用，量小但未纳入 `harness.token_budget`。
