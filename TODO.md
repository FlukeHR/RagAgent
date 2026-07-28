# TODO / 完成清单

> 更新时间：2026-07-25  
> 工程改造已经完成；未勾选项只包含必须由项目维护者使用真实数据、真实 API 和实际部署环境执行的评测。不要在没有实验结果时填写提升数字。

## P0：正确性与安全

- [x] 通过 `PaperRepository` 统一校验 `paper_id`、目录 containment 和扁平论文库归属。
- [x] 严格校验现代/旧版 arXiv ID，旧版含 `/` 的 ID 映射为安全存储名。
- [x] 工具 schema 增加未知字段拒绝、长度、数组数量、格式和范围约束。
- [x] `ToolPolicy` 声明 side effects、幂等性、超时、重试和进程隔离策略。
- [x] 写工具默认不重试；`ingest_arxiv_papers` 在独立进程中运行，超时后 terminate/kill，避免写任务后台继续。
- [x] 网络请求使用显式 timeout；下载采用大小限制、PDF 文件头检查、`.part` 临时文件和原子替换。
- [x] 保存实际 embedding backend、模型、维度、归一化方式和 fingerprint。
- [x] 查询 embedding 与索引 fingerprint 不一致时拒绝检索并要求重建，不再静默混用向量空间。
- [x] 向量索引采用 generation 文件和 manifest-last 原子发布，构建失败不替换当前 manifest。
- [x] Retriever 检查 generation 并热重载 VectorStore/BM25，解决 Agent 入库后的陈旧索引。
- [x] `Retriever.search()` 支持 paper、modality、element type 过滤；arXiv 入库只检索本轮目标论文。
- [x] API 与 tool adapter 共享 `ArxivSearchService`/`PaperLibraryService`，API 不再直接调用 tool class。

## P1：对话、记忆与上下文

- [x] 增加确定性的模糊度检测和 `needs_clarification` 状态。
- [x] 澄清问题覆盖研究对象、比较对象、评价标准、时间范围和期望输出。
- [x] 增加 ConversationState：目标、约束、事实、待办、引用论文和摘要。
- [x] 对含“继续/这个/它”等承接词但与原目标不匹配的输入进行主题偏移确认。
- [x] 历史使用最近窗口 + 较早消息滚动摘要，不再无界注入全部原文。
- [x] LLM 调用前预估 input + reserved output token，预计超预算时不发起调用。
- [x] history、tool result、source snippet、图片 base64 和总上下文分别设置配置上限。
- [x] tool result 进入下一轮前再次按总 context budget 截断。
- [x] 使用 SQLite 保存服务端会话，支持 TTL、容量治理、读取和删除接口。
- [x] 前端继续使用 `localStorage` 作为离线副本，并把安全 session ID 传给后端。

## P1：证据冲突、引用与幻觉护栏

- [x] 所有工具返回统一 `EvidenceSource`，不再自行分配 `[S编号]`。
- [x] `EvidenceRegistry` 统一去重、编号、替换 citation placeholder 和输出 source。
- [x] 重复内容按 normalized snippet/content hash 抑制，避免重复论文制造虚假多来源。
- [x] AnswerVerifier 除检查 source ID 外，还做 claim/source token 支持度检查。
- [x] 对高重叠来源检查数值差异和否定关系，标注潜在 conflict。
- [x] 冲突记录包含论文时间、证据质量和可选 preferred source；无法消解时提示分别陈述并降低结论强度。
- [x] Agentic 和 fallback 两条路径都执行 claim、citation 和 conflict 核查。
- [x] 提供 `evaluation/audit_citations.py` 导出 claim/source 对，供人工支持度审计。

> 当前 claim 与冲突检测是保守的词法/数值启发式，不等价于经过评测的 NLI 模型。真实准确率需按 `EVALUATION_GUIDE.md` 人工抽检。

## P1：retrieval 简化与重构

- [x] 用 `retrieval/models.py` 合并 ParsedPage/PageText、ParsedElement/PaperElement 等重复模型。
- [x] 将原 PaperLoader 职责拆成 PaperRepository、PDF parser、DocumentNormalizer 和 PDFService；PaperLoader 只保留兼容 facade。
- [x] section/page/element 统一为可组合 `ChunkStrategy`。
- [x] Chunk 增加 `parent_id`、`content_hash` 和 `granularity`。
- [x] native text 优先，OCR/VLM 作为补充或 fallback；精确重复内容只索引一次。
- [x] page chunk 不再错误使用首个 text block bbox。
- [x] `QueryAnalyzer` 统一英文 token 与中文 CJK n-gram，BM25、fallback、冲突核查和评估共用。
- [x] Dense、Sparse、RRF、ParentDiversifier、Reranker 和 ScoreCalibrator 拆成独立阶段。
- [x] `RetrievalResult` 分别记录 dense/sparse/fusion/rerank score、backend 和 confidence。
- [x] 生产和评估共同使用 `RetrievalPipeline`/`rank_in_memory`，减少实现漂移。
- [x] Retriever 支持同一 parent 数量上限与全局 content hash 去重。
- [x] PDF 页读取只解析目标页。
- [x] 页面像素签名在 build index 时预计算；查询时只做向量搜索并渲染最终 top-k。
- [x] 图片签名基于解码后像素直方图，不再比较 PNG/JPEG 压缩字节。
- [x] 实际使用 14 个本地文件完成全量构建 smoke test：1414 chunks，图片索引 80 页。

## P1：tool 与 Agent 简化

- [x] 建立 `Tool` Protocol、`ToolSpec`、`ToolPolicy` 和单一 `ToolRegistry`。
- [x] 工具实例、schema、白名单和执行策略不再在 Agent 中双重维护。
- [x] harness 从 `agent/graph.py` 拆到 `agent/harness.py`。
- [x] Evidence/Verifier、Conversation/Memory、FallbackRAG 拆成独立模块。
- [x] 建立 local/pdf/arxiv 的渐进式 tool profile：首轮只暴露相关 schema，观察一次工具结果后才允许策略扩展。
- [x] profile 只控制允许暴露的原子工具，任何调用仍经过 harness，不把 tool 包成可绕过权限的 opaque skill。
- [x] 评估 `read_pdf_page` 与 `read_pdf_region` 合并方案后决定保留两个工具：region 的必填 bbox 和更严格图像预算构成独立安全边界。
- [x] 文本、图片、历史、网络、LLM 和下载上限迁移到 `config/config.yaml` 并启动校验。
- [x] API service 与 tool adapter 分离。
- [x] 拆分 `requirements.txt`、`requirements-eval.txt`、`requirements-optional.txt` 和 `requirements-dev.txt`。

## P1：评估基础设施

- [x] `eval_qasper.py` 支持 Dense/BM25/Hybrid、是否 rerank、top-k/top-n 和 JSON/CSV 落盘。
- [x] `benchmark_retrieval.py` 执行 Dense only、BM25 only、Hybrid、Hybrid+CE 消融和 top-k/top-n sweep。
- [x] `benchmark_grounded.py` 在真实本地论文 gold 数据上做 chunk size、overlap 和 top-k sweep。
- [x] `eval_generation.py` 继续保持真实 API 显式授权，并保存 metadata、生成 usage 和可用时的 judge usage。
- [x] `benchmark_e2e.py` 记录总延迟、first-byte、p50/p95/p99、token、工具轮数和估算成本。
- [x] 所有 Agent 辅助生成、function-calling 主循环和请求总耗时写入 trace。
- [x] 评估配置支持 input/output 每百万 token 单价。
- [x] 评测结果记录 commit、dirty state、配置快照、模型 backend、Python、平台、处理器和时间；API key 自动脱敏。
- [x] 建立 `business_cases.jsonl` 的模糊、多轮、可答、不可答和冲突类别模板。
- [x] 建立 `badcases.jsonl` 字段模板和 `audit_citations.py` 人工审计导出。

## P2：测试与回归

- [x] 增加路径越界、恶意 arXiv ID、未知 schema 字段测试。
- [x] 增加隔离写工具硬超时和不重试测试。
- [x] 增加索引 fingerprint、不兼容拒绝、原子 generation、热重载和过滤检索测试。
- [x] 增加 CJK analyzer、重复 chunk、page bbox 和 metadata 测试。
- [x] 增加 Evidence 编号、去重、无效引用和数值冲突测试。
- [x] 增加模糊澄清、主题偏移、历史压缩和 SQLite 记忆测试。
- [x] CI 保留 Ruff 和非阻断 mypy；维护者决定移除 `tests/` 后同步删除 unittest 步骤。
- [x] 本地验证使用 Ruff、mypy、实际索引构建和检索 smoke evaluation。
- [x] RAGAS 保持显式付费授权，不进入自动 CI。

## 待维护者执行的真实评测

- [ ] 下载官方 QASPER 数据并运行完整 retrieval baseline/ablation。
- [ ] 在 `business_cases.jsonl` 中替换并启用真实业务问题、gold paper/page/evidence。
- [ ] 运行生产 grounded chunk/top-k sweep，选出正式 chunk 与召回参数。
- [ ] 在固定硬件和同一模型下运行 E2E baseline，记录冷/热启动和各工具路径。
- [ ] 显式授权运行 RAGAS 生成评估，核对 judge usage 与供应商账单。
- [ ] 随机抽检至少 50 条 claim/source，填写 supported/partially/unsupported/bad citation。
- [ ] 将真实 badcase 写入 `evaluation/data/badcases.jsonl`，修复后加入回归集。
- [ ] 用同一数据和环境比较 baseline/candidate，填写命中率、幻觉率、token、费用和延迟变化。

完整命令、指标解释和结果表见 `EVALUATION_GUIDE.md`。
