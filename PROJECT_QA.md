# 项目问题与处理方法

> 更新时间：2026-07-27
> 本文回答“当前代码如何处理”，并记录已经完成的 baseline。完整实验条件和原始结果解释见 `EVALUATION_GUIDE.md`。

## 1. 工具调用超时怎么办？

**状态：已处理。**

每个 tool 通过 `ToolPolicy` 声明：

- timeout；
- 最大重试次数；
- read/network/write 副作用；
- 是否幂等；
- 是否必须在隔离进程中执行。

`ToolHarness` 在执行前校验 schema 和白名单，再按 policy 执行并记录 attempts、duration、error、
source 数和内部 metadata。

具体策略：

- 普通只读工具超时后立即向模型返回 error outcome；
- 网络调用同时设置 requests/OpenAI/arXiv 客户端内部 timeout；
- 幂等只读工具才允许有界重试；
- 写工具默认不重试；
- `ingest_arxiv_papers` 在独立进程运行，超时后 terminate，必要时 kill，避免下载或索引写任务继续后台修改状态；
- 下载使用 `.part`、大小上限、PDF 文件头校验和原子替换；
- 工具失败不会中断整个 Agent，而是回注模型调整策略。

配置位于 `config/config.yaml` 的 `harness`、`arxiv` 和 `llm`。

## 2. 多个工具结果冲突怎么办？

**状态：已增加显式冲突处理；当前是启发式，不是经过评测的 NLI。**

`EvidenceRegistry` 会先按 normalized content 去重，再检查不同论文的高重叠 evidence：

- 数值集合不同；
- 一方存在否定关系而另一方不存在。

发现冲突后：

- source 标记为 `conflict`；
- trace 记录冲突的 `[S编号]`、原因、重叠度、发表时间和 evidence quality；
- 有明显时间或质量差异时给出 `preferred` 候选，但不会静默丢弃另一方；
- Agent 收到系统反馈，必须分别说明结论与条件；
- 无法消解时降低结论强度并保留分歧；
- fallback 路径也会在答案中追加冲突提醒。

最终是否准确识别冲突仍需人工审计，尤其是语义相反但没有明显否定词的情况。

## 3. 用户问题很模糊怎么追问？

**状态：已处理。**

- `ConversationManager` 检查过短问题和“这个、哪个更好、帮我看看、比较一下”等缺少对象的表达。
  若历史和会话状态不能补全：
- 返回 `status=needs_clarification`；
- 暂停工具调用；
- 追问研究对象、比较对象、评价标准；
- 提示可补充时间范围和期望输出格式；
- trace 记录 `ambiguous_question`。

有历史时仍会先做查询改写；能够从历史消解“它/该方法”的，不会重复追问。

## 4. 多轮对话偏移怎么拉回来？

**状态：已处理基础偏移场景。**

服务端保存当前 goal。新问题含“继续、这个、它、刚才”等承接词，但与原 goal 没有可识别交集时，
系统不会直接检索，而是询问：

```text
你是想继续原目标……，还是切换到一个新问题？
```

用户明确使用“换个话题、另一个问题、切换到”等表达时，系统更新 goal，不阻止正常主题切换。
查询改写、context compaction、clarification 和 drift 都写入 trace。

## 5. 多轮记忆怎么保留？

**状态：已实现浏览器副本和服务端有界记忆。**

- 前端 `localStorage` 保存多会话，作为本地/离线副本；
- 前端为每个会话生成高熵 session ID，并随 `/ask` 发送；
- SQLite 保存 ConversationState 和有界历史；
- state 包含 goal、constraints、facts、pending、cited papers 和 summary；
- 提供 `GET /sessions/{session_id}` 和 `DELETE /sessions/{session_id}`；
- 根据 TTL 删除过期会话；
- 根据 `memory_max_sessions` 淘汰最旧会话；
- session ID 做格式和长度校验。

当前定位仍是单用户本地部署，没有账号、租户和 ACL；若公开部署，需要在 API 前增加认证授权。

## 6. 如何避免上下文太长？

**状态：已实现多层预算。**

处理顺序：

1. 只保留最近若干原始消息；
2. 较早消息压缩为有界滚动摘要；
3. history 总字符数受限；
4. tool result、source snippet 和 image base64 分别受限；
5. Agent 工作历史超过总 context budget 时再次截断 tool content；
6. 每次调用 LLM 前估算 input token，并为最大 output token 预留空间；
7. 预计超过 token budget 时不发起该次调用，直接基于现有证据总结；
8. `max_tool_iters` 继续限制多跳轮数。

所有截断和预算停止事件写入 trace。辅助生成、function-calling 主循环的 token 也统一汇总为
`type=usage`。

## 7. 做 RAG 后问答命中率有没有提升？

**状态：已完成公开数据 baseline；尚无改动前后的 candidate 对照。**

QASPER dev 的 888 个问题已经完成同环境检索消融：

- Dense：Hit@5=0.6002、MRR=0.3582、Recall@5=0.4494；
- BM25：Hit@5=0.5721、MRR=0.3282、Recall@5=0.4227；
- Hybrid/RRF：Hit@5=0.6318、MRR=0.4063、Recall@5=0.4767；
- Hybrid/RRF/CrossEncoder：Hit@5=0.6959、MRR=0.4710、Recall@5=0.5375。

这证明当前同一评测链里 Hybrid + CrossEncoder 优于 Dense baseline，但不能表述为“项目改造后
提升”，因为还没有固定旧 commit 作为 candidate 的对照。统一复现命令是
`python evaluation/evaluate.py --profile key`。

## 8. token 成本有没有压？

**状态：已得到 smoke usage；没有价格和前后对照，不能声称成本下降。**

现在 trace 统一记录：

- Agent function-calling input/output token；
- rewrite、summary、fallback 等辅助 `generate()` usage；
- 每请求总 input/output token；
- 根据配置单价计算的 estimated cost。

QASPER 生成 smoke 的 5 个回答共使用 5,024 input token、1,391 output token，RAGAS judge
记录 94,533 total token。当前 `evaluation.input_price_per_million` 和 output price 都是 0，
旧 E2E 报告的 `estimated_cost=0` 只是配置结果，不是免费或成本已下降的证据。

`evaluation/evaluate.py --profile key --yes` 会统一记录生成、E2E 和 judge usage。

最终账单必须与供应商 billing 核对，不能只使用本地估算。

## 9. 回答延迟从多少降到了多少？

**状态：旧 E2E 数据无效，计时算法已修正，等待补齐 answerable 用例后重跑。**

现在记录：

- request 总耗时；
- 每次 LLM 调用耗时；
- 每个 tool 总耗时；
- retrieval 的 store、embedding、Dense、Sparse、fusion/rerank 分阶段耗时；
- E2E response headers、mean、p50、p95、p99；非流式接口不记录伪 TTFT。

旧报告只覆盖 10 次追问请求。原脚本把小样本分位数向下取整，p50 错报 44.8 ms；从原始记录
线性插值得到 p50=635.1 ms、p95=1878.8 ms。此外 `/ask` 是非流式 JSON，旧字段
`first_byte` 其实是响应头耗时，不是模型 TTFT。以上问题已修正，但旧结果不能用于宣称延迟改善。

需要分别比较冷启动、热启动、本地检索、二次检索、PDF、图片和 arXiv 路径。

## 10. 幻觉率有没有测？

**状态：已跑 5 样本 RAGAS smoke；人工引用审计尚未形成有效样本。**

在线处理：

- `[S编号]` 必须来自真实 EvidenceRegistry；
- 无效引用会删除并触发有界纠错；
- claim/source 词法支持不足会触发复核或纠错；
- 证据不足会拒答；
- potential conflict 不会被当作一致结论。

5 样本结果为 Faithfulness=0.7500、Answer Relevancy=0.5836、Context Precision=0.5000、
Context Recall=0.6500、Answer Correctness=0.4631、Factual Correctness=0.0000。样本太少，
不能外推为系统幻觉率。

旧 E2E 只有追问响应，没有 answered/source/citation，故审计 CSV 为 0 条；这不表示幻觉率为 0。
现在统一入口会自动接审计并为 0 条结果写明原因。最终仍由 CSV 人工标注 supported、
partially supported、unsupported 和 bad citation。

## 11. 是否出现过 badcase，是怎么处理的？

**状态：仓库没有历史真实 badcase 记录，因此不能声称出现过哪些；闭环工具已经建立。**

`evaluation/data/badcases.jsonl` 定义：

- 输入与历史；
- expected/actual；
- sources 和 trace；
- category；
- root cause；
- fix；
- fixed commit；
- regression status。

建议分类包括 ambiguity、drift、retrieval miss、wrong ranking、duplicate context、conflict、
unsupported claim、bad citation、false refusal、timeout、stale index 和 prompt injection。

处理流程：

```text
固定输入/配置/trace
→ 定位 parser/chunk/recall/rerank/context/generation/harness
→ 加最小回归测试
→ 单因素修复
→ Ruff + mypy + QASPER smoke
→ 必要时显式授权 RAGAS
```

## 12. 切片策略怎么选择的？

**状态：已完成 FinanceBench sweep；生产参数仍需结合生成 token/延迟确认。**

当前策略：

- section/page/element 为三个可组合 ChunkStrategy；
- section 使用字符窗口并尽量在段落、换行和句号处切分；
- page 用于页码和视觉工具路由；
- table、figure、formula、layout 使用 element chunk；
- native text 优先，OCR/VLM 是补充或 fallback；
- exact content hash 去重；
- Chunk 保留 paper、section、page、bbox、modality、parent、granularity 和 content hash；
- retrieval 限制同一 parent 的最大结果数。

FinanceBench 的 150 个问题上，解析 evidence coverage=0.9853、gold page token F1=0.9722。
已测组合中 1400/140、Top-20 得到 Page Hit=0.3867、Evidence Hit=0.3867、MRR=0.1837；
900/135、Top-20 对应 0.3267、0.3533、0.1609。配置已改为 chunk 1400/140、recall top-k=20、
rerank top-n=5；由于还没有同条件生成 token、延迟和本地论文 gold 回归，目前仍视为候选参数。

## 13. top-k 怎么调？

**状态：QASPER 与 FinanceBench 已完成 sweep；候选值已有，生产值尚未定稿。**

推荐先调 recall top-k，再调 rerank top-n，最后结合生成质量、延迟和 token 选择。

QASPER 的 recall top-k=24、rerank top-n=8 得到 Hit@8=0.7849、MRR=0.4872、Recall@8=0.6397；
Top-5 的 12/5 配置为 Hit@5=0.6959、MRR=0.4710、Recall@5=0.5375。不要只选最高 Recall：
top-k/top-n 增大也会增加 rerank 延迟、上下文 token 和噪声。

## 14. 找回效果有没有对比？

**状态：公开数据真实结果已跑，本地论文 gold 对比未跑。**

QASPER 消融覆盖 Dense、BM25、Hybrid 和 Hybrid+reranker。生产 grounded benchmark 覆盖 chunk、
overlap 和 top-k。生产与评估现在共用 `RetrievalPipeline`，避免评估脚本复制另一套 RRF/rerank。

报告必须同时给出：

- retrieval metrics；
- 实际 backend（CrossEncoder 还是 token-overlap fallback）；
- embedding fingerprint；
- p95 latency；
- token/context 数量。

## 15. 为什么这样设计工具链？

**状态：设计理由明确；优越性仍需实验验证。**

工具链采用“低成本、低副作用优先，证据不足再扩展”的渐进策略：

1. 首轮默认只暴露 local profile，减少 schema token；
2. 最新研究问题直接开放 arXiv profile；
3. 图表、页码和公式问题开放 PDF profile；
4. 一次工具观察后允许策略扩展；
5. arXiv 先查摘要，只下载少量已筛选论文；
6. section/page/region/image 按证据粒度逐步深入；
7. 每个动作仍经过 schema、权限、预算、超时和 trace；
8. 最后执行 citation、claim、conflict 和 answerability 核查。

tool 保持原子能力；profile/skill 只负责“允许模型看到哪些 tool”，不能直接执行工具或绕过 harness。
`read_pdf_page` 和 `read_pdf_region` 继续分开，因为 region 的必填 bbox 和更严格图片预算是独立安全边界。

这种设计理论上能降低无必要下载、embedding、schema token 和上下文噪声，但是否优于其他工具链，
必须由维护者完成真实消融、延迟、成本和幻觉评测后再下结论。
