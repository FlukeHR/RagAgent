# STATUS.md — Paper RAG Agent 当前进展

> 改动频繁的东西放这里：当前状态 / 待办 / 已知问题 / 评估基线。
> 稳定事实（架构、命令、规范、护栏）见 `AGENTS.md`。
> 最近盘点：2026-07-06。

## 一句话现状

检索 + agentic 循环 + API + 前端**端到端跑通**，全部核心不变量与护栏已落地（agentic 主路径 + 降级单跳 RAG 均接引用回查与二次检索）。能力已覆盖：
多源检索（本地稠密+BM25 RRF→CrossEncoder 重排）、引用回查、低置信二次检索、harness 护栏（schema 校验/超时重试/token 预算/结构化 trace）、
多轮对话（历史注入+指代消解）、多会话前端（自动标题+PDF 预览高亮+trace 可视化）、增量索引、arXiv 全文按需入库（同步单轮）+ 集合容量治理、
本地模型副本、生成侧 RAGAS 评估（显式授权）、CI（ruff+pytest）。
上次记录 `pytest tests/` **48 用例全绿**（全离线 mock，不打网络/arXiv/LLM API）；本轮新增 PDF loader / PDF page tool / PDF grounding eval 单测，当前执行环境缺少 `pytest`/`ruff`，已用 `python -m compileall` 做语法级验证。

## 本轮进展（2026-07-06）

- **PDF 页级解析地基已落地**：`PaperLoader` 现在按页解析 PDF，保留 `PageText.page_number` 与 `is_scanned_like`；章节和 chunk 会带 `page_start/page_end`，检索返回的 `sources` 与 API schema 也会暴露页码范围。
- **混合索引已落地**：构建索引时同时写入细粒度语义 text chunk 与页级 page chunk；检索继续走稠密向量 + BM25 RRF + 可选 CrossEncoder rerank，使跨页近似搜索和页码定位可共用现有检索链路。
- **按需 PDF 页工具已落地**：新增 `read_pdf_page`，按 `paper_id + page_number` 读取单页文本；`include_image=true` 时把该页按 `max_side` 限制渲染成 JPEG/PNG，并可返回受限 base64，供扫描件/OCR/VLM 后续链路使用。
- **PDF grounding 评测已补**：新增 `evaluation/eval_pdf_grounding.py`，离线计算 page hit、page recall、table/value consistency，并接入 `results_log.py --kind pdf_grounding`。
- **OCR/VLM 产物入索引链路已落地**：PDF 同目录 `paper.ocr.json` / `paper.vlm.json` 会被 loader 读取；扫描页即使没有内嵌文本，也可通过 sidecar 生成 `modality=ocr|vlm` 的可检索 chunk。sidecar 参与文件 hash，产物变更会触发增量重建。
- **索引兼容处理**：索引参数签名增加 `chunk_metadata_version=5`，下一次增量构建会触发旧索引重建，避免复用缺少页码/页级/OCR/VLM/element/context chunk 的旧索引。
- **测试补充**：新增 PDF loader、PDF page tool、PDF grounding eval 离线单测，覆盖页码范围传递、扫描样式页面检测、页面图片渲染、OCR sidecar 入索引并被检索命中、页码命中和表格/数值一致性。

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

## PDF 解析增强路线：借鉴 LlamaIndex / LiteParse / LlamaParse

> 结论：借鉴它们的**解析层与 metadata 抽象**，不要替换本项目的自研 harness。现有 harness 的 schema 校验、预算、超时、引用回查、低置信二次检索和 trace 仍是项目核心。

- **LiteParse 优先作为本地解析 provider**：用于补足本地 OCR、layout、bbox、markdown/结构化文本能力，适合默认离线/低成本路径。目标是把 LiteParse 结果转换为项目已有 sidecar：`paper.ocr.json`、`paper.vlm.json`、后续 `paper.layout.json`、`paper.tables.json`、`paper.bboxes.json`。
- **LlamaParse 作为可选高级 provider**：用于复杂扫描件、表格、图表、word/line/cell 级 bbox 或高质量 markdown。它涉及 API key、成本和隐私，应当 opt-in，不进入单测默认路径；测试里只 mock provider 输出。
- **LlamaIndex Framework 只借鉴抽象**：可参考 `Document / Node / metadata / ingestion pipeline / transformations` 的设计，把解析、切块、metadata 组装做成 provider/pipeline；不把 agent loop、retriever 决策和引用核查迁移到 LlamaIndex agent。
- **统一接入形态**：新增 `PDFParseProvider` 抽象，至少包含 `pymupdf`（现有默认）、`liteparse`（本地增强）、`llamaparse`（可选云端增强）三个实现。所有 provider 输出统一的 `ParsedPDF` / sidecar，不让索引、检索、工具层直接依赖某个第三方 SDK。
- **工程护栏**：外部 provider 调用必须有超时、错误降级和 trace；解析结果属于不可信数据，只能作为证据输入，不得影响 harness 护栏；API key 只走环境变量 / secrets。

## 原始需求差距 / 后续任务

> 2026-07-06 追加实现后，PDF 多模态相关的代码闭环已补到可测试地基；本节只保留外部运行/真实基线层面的限制。

1. **PDFParseProvider / OCR 运行时已接入**  
   新增 `retrieval/pdf_parse.py`：统一 `PDFParseProvider` / `ParsedPDF` / `ParsedElement` 抽象，默认 `pymupdf` provider 读取 PDF 文本、页级 OCR/VLM sidecar、layout/table/figure/formula/bbox sidecar；`tesseract` 为 opt-in 本地 OCR runtime，带超时与失败降级。索引阶段可通过 `pdf_parse.provider/auto_ocr/timeout_seconds` 配置，mock provider 单测覆盖“扫描页无 sidecar 时生成 OCR 内容并入索引”。

2. **图像检索地基已接入**  
   新增 `search_pdf_images` harness 工具，支持 `image_base64` 或 `paper_id + page_number (+ bbox)` 作为 query，召回相似 PDF 页面图像；source 返回页码、`element_type=page_image`、`modality=image`、分数和可预览图片 metadata。当前默认是离线图像签名 fallback，后续可替换为 CLIP/SigLIP 向量模型。

3. **表格 / 图表 / 公式元素入索引已接入**  
   索引输入 hash 已纳入 `.elements/.layout/.tables/.figures/.formulas/.bboxes.json`；chunker 新增 `split_elements`，可生成 `table/figure/formula/text_block` 等 chunk，保留页码、bbox、caption/summary、modality 与来源。sidecar-only 扫描 PDF 也不会因页文本为空被跳过。

4. **页内 bbox 定位已闭环到工具/API**  
   `read_pdf_region` 可按 PDF bbox 读取/渲染局部区域；`render_pdf_page` 支持 bbox clip；`/preview/meta` 支持直接用页码定位，`/preview/page` 支持 bbox 高亮，snippet 高亮仍保留为降级路径。

5. **chunk 上下文说明已补全**  
   `Chunk` / source schema / prompt 均新增 `chunk_context` 与 `heading_path`；本地检索、arXiv 全文入库、PDF page/region 工具都会返回该字段，方便展示和引用溯源。

6. **no-answer 硬闸已加强**  
   agent 主路径和降级 RAG 都接入 `answerability gate`：按有效来源数量、最低分、生成后有效引用检查决定是否允许实质回答；失败时统一输出“未检索到充分依据”，并在 trace 中记录 `answerability` 事件。

7. **PDF grounding 评测集与指标已补**  
   `eval_pdf_grounding.py` 新增 `bbox_hit`、`ocr_hit`、`visual_semantic_hit`；新增 `evaluation/data/pdf_grounding/offline_smoke.json`，CI 增加纯离线 smoke eval。当前环境已跑通该 smoke：page_hit=1.0、page_recall=1.0、value_consistency=0.5、bbox_hit=0.5、ocr_hit=0.5、visual_semantic_hit=0.5。

8. **生成侧 RAGAS 基线仍需真实环境运行**  
   代码入口仍是 `RAG_EVAL_ALLOW_API=1 python3 evaluation/eval_generation.py --qasper --limit 20 --record`。本轮执行环境缺少项目依赖（`PyYAML/numpy/PyMuPDF/pytest/ruff` 等）且未提供真实 LLM API key/计费授权，因此不能可靠落数；不得伪造该基线。拿到完整依赖和显式 API 授权后再写入数值。

## 已知问题 / 限制

- **会话仅存浏览器 localStorage**（`rag_sessions` / `rag_active`）：无跨设备 / 后端持久化（已与用户确认的设计取舍）。
- **prune 跨进程**：集合级锁只保进程内并发；手动 `python3 indexing/prune.py` 是独立进程，建议在服务空闲时跑。
- **低置信阈值与 rerank 模式耦合**：默认阈值按 CrossEncoder 开（CE logit→sigmoid）标定；CE 关（降级 token+余弦）需另行重标。
- **额外 LLM 调用未计入 token 预算**：查询改写（仅有历史时）、标题生成（仅首轮）各一次轻量调用，量小但未纳入 `harness.token_budget`。
- **当前仍不是完整多模态 RAG**：页码范围、text/page 混合索引、页面图片工具和 OCR/VLM sidecar 入索引已补齐第一层地基，但主检索仍是文本 embedding；扫描件若没有 sidecar 或外部 OCR/VLM provider，仍无法自动从图像生成可检索内容。
