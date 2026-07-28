# RAG 真实评测指南

本指南记录 2026-07-25 至 2026-07-27 已完成的 baseline，并说明如何复现。只有在同一数据、
模型、硬件和运行方式下比较，才能报告“提升了多少”。`evaluation/results/` 默认不提交 Git，
下表数字来自当前工作区保存的原始报告。

## 统一入口

日常评测只使用一个入口。默认运行 QASPER 检索消融和 FinanceBench 的解析、切片、检索指标，
不调用付费 LLM API：

```powershell
python evaluation/evaluate.py --profile key
```

快速检查使用 `--profile smoke`；完整参数网格使用 `--profile full`。OmniDocBench OCR 较慢，
可在 smoke/key 中显式加入：

```powershell
python evaluation/evaluate.py --profile smoke --omnidoc
```

生成、E2E 和引用审计需要真实 API。先启动 API，再显式授权；引用审计会在 E2E 后自动执行：

```powershell
uvicorn api.main:app
python evaluation/evaluate.py --profile key --yes
```

一次运行的分阶段结果和汇总写入 `evaluation/results/latest/`，总表是
`key_metrics.json`。`eval_qasper.py`、`eval_financebench.py`、`eval_omnidocbench.py`、
`eval_generation.py`、`benchmark_e2e.py` 和 `audit_citations.py` 是统一入口调用的底层适配器，
只在调试单项时直接运行。

## 已完成测评数据

所有有效检索实验均使用 `sentence-transformers/all-MiniLM-L6-v2`；启用重排时实际 backend 为
`cross_encoder`（`cross-encoder/ms-marco-MiniLM-L-6-v2`）。报告对应 commit
`7b9cc79c037a0d70d99396d7420f60cb21aed798`，工作区为 dirty，因此这些数字是当前工作区
baseline，不是可跨 commit 复用的发布基准。

QASPER dev 共评测 281 篇论文、888 个有 gold evidence 的问题，输出 Top-5：

| 检索链 | Hit@5 | MRR | nDCG@5 | Recall@5 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.6002 | 0.3582 | 0.3387 | 0.4494 |
| BM25 | 0.5721 | 0.3282 | 0.3096 | 0.4227 |
| Hybrid / RRF | 0.6318 | 0.4063 | 0.3732 | 0.4767 |
| Hybrid / RRF / CrossEncoder | **0.6959** | **0.4710** | **0.4392** | **0.5375** |

在同一 Top-5 口径下，Hybrid + CrossEncoder 相比 Dense 的 Hit@5 增加 9.57 个百分点，MRR
增加 0.1128；相比不重排 Hybrid，Hit@5 增加 6.42 个百分点，MRR 增加 0.0647。扩大到
recall top-k=24、rerank top-n=8 时，Hit@8=0.7849、MRR=0.4872、Recall@8=0.6397，
但上下文更多，不能与 Top-5 的指标直接当作同口径提升。

FinanceBench 使用 150 个问题、84 份 PDF。解析层
`gold_page_nonempty_rate=1.0000`、`evidence_text_coverage=0.9853`、
`gold_page_token_f1=0.9722`；900/135 切片的 evidence preservation 为 0.9630，页码 metadata
准确率为 1.0000。固定 900/135、Top-20 时的检索消融为：

| 检索链 | Paper Hit | Page Hit | Evidence Hit | Evidence Recall | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.6800 | 0.1800 | 0.1533 | 0.1478 | 0.0772 |
| Dense | **0.9267** | 0.3067 | **0.3733** | **0.3478** | 0.1185 |
| Hybrid / RRF | 0.8933 | **0.3267** | 0.3533 | 0.3322 | 0.1239 |
| Hybrid / RRF / CrossEncoder | 0.8933 | **0.3267** | 0.3533 | 0.3322 | **0.1609** |

CrossEncoder 在此配置没有提高 Top-20 命中率，但把 MRR 从 0.1239 提高到 0.1609，说明收益
主要来自顺序。chunk sweep 中 1400/140、Top-20 得到 Page Hit=0.3867、Evidence Hit=0.3867、
MRR=0.1837，是已测组合中较好的检索结果。当前生产候选已改为 chunk 1400/140、
recall top-k=20、rerank top-n=5；仍需用本地论文 gold、生成 token 和延迟做上线前复核。

生成侧 smoke 使用 QASPER 5 个问题：Faithfulness=0.7500、Answer Relevancy=0.5836、
Context Precision=0.5000、Context Recall=0.6500、Answer Correctness=0.4631、
Factual Correctness=0.0000。回答生成共 5,024 input token、1,391 output token；RAGAS judge
记录 94,533 total token。样本只有 5 个，且配置单价为 0，不能据此报告总体质量或真实费用。

MMLongBench-Doc 只跑了 10 个问题、2 份 PDF 的 smoke：Top-10 Page Hit=0.6250、
Page Recall=0.4375、all-gold-pages Hit=0.3750、MRR=0.2583。gold evidence 文本没有接入，
所以 Evidence Hit=0 不代表检索完全失败。

OmniDocBench 原报告显示 `pages=0`，不是 0 分 baseline：旧代码把 `english` 错映射成 `en`，
导致 206 个英文 academic_literature 页面全部被过滤。映射和空样本检查已修复，必须重跑后才能
填写解析成绩。

旧 E2E 报告的 10 次请求只覆盖两个 `needs_clarification` 样本，状态命中 10/10，但没有
answerable 样本、source 或引用；回答文本也检测到乱码。因此它只能说明追问分支被执行，不能
作为 E2E 问答、引用、成本或幻觉 baseline。旧脚本还用向下取整计算分位数，把 p50 错报成
44.8 ms；由原始 10 条 latency 线性插值得到 p50=635.1 ms、p95=1878.8 ms。新入口已修正
分位数算法，并把旧称 `first_byte` 的非流式响应指标改为 `response_headers_ms`。

`citation_audit.csv` 为 0 字节的原因同样是没有 `answered` 回答可审计。新审计器即使 0 条也会
写出固定表头和 summary，并把无引用陈述标成 `missing_citation`；自动 overlap 仍不能替代人工
`supported / partially_supported / unsupported / bad_citation` 标注。

## 1. 固定实验条件

每次评测记录：

- Git commit 与工作区是否 dirty；
- `config/config.yaml`；
- Python 和依赖版本；
- embedding/reranker/LLM 的名称与实际 backend；
- CPU/GPU、内存、操作系统；
- 数据集版本和样本数；
- 冷启动或热启动；
- API 单价；
- 评测时间。

评测脚本会自动保存大部分 metadata，并脱敏 `openai_api_key`。建议实验名称：

```text
2026-07-25_baseline_hybrid-rrf-ce
```

每组实验只改变一个因素。评测结果默认写入 `evaluation/results/`，该目录不提交 Git。

## 2. 安装与本地索引

运行依赖：

```powershell
pip install -r requirements.txt
```

全量构建生产索引：

```powershell
python indexing/build_index.py --full
```

检查 `data/indexes/manifest.json` 中的：

```text
embedding.backend
embedding.model_name
embedding.fingerprint
dim
generation
```

若实际 backend 是 `hashing` 或 reranker 是 `token_overlap`，报告中必须明确说明，不能写成
SentenceTransformer/CrossEncoder 的实验结果。

下面两个 PDF 数据集入口都不调用 `LLMClient`，不会产生大模型 API 费用。本地
SentenceTransformer/CrossEncoder 首次使用时可能从 Hugging Face 下载模型；之后可以离线运行。

## 3. 两套 PDF 基准的目录

数据集本体不提交仓库。下载后整理成：

```text
evaluation/data/
├─ omnidocbench/
│  ├─ OmniDocBench.json
│  └─ images/
├─ financebench/
│  ├─ data/
│  │  └─ financebench_open_source.jsonl
│  └─ pdfs/
```

官方来源：

- [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- [FinanceBench](https://github.com/patronus-ai/financebench)

脚本会递归查找结构化数据和 PDF。若自动发现失败，使用各入口的 `--questions`、`--pdf-dir`
显式传入。

不再把 MMLongBench-Doc 作为必跑项，也不为了凑满三套引入另一个大文件或受限数据集。
OmniDocBench 负责解析质量，FinanceBench 负责解析、切片和 evidence 召回，最后用本地论文
gold set 验证学术领域效果，这三层已经覆盖当前项目需要的选型依据。

### 先确认本地模型 backend

运行 FinanceBench 评测后，检查报告中的：

```text
embedding.backend
embedding_load_error
reranker_backend
reranker_load_error
```

正式实验应优先得到：

```text
sentence_transformers
cross_encoder
```

若得到 `hashing`/`token_overlap`，脚本仍能完成离线基线，但不能把结果写成真实语义模型实验。

## 4. OmniDocBench：PDF 页面解析

OmniDocBench 主要提供页面图像和结构化标注。当前项目的 PyMuPDF 原生文本路径不能直接从页面图像
取得文字，因此有两种运行方式。

### 4.1 使用项目的本地 OCR 路径

先安装 Tesseract，并确认：

```powershell
tesseract --version
```

Smoke test：

```powershell
python evaluation/eval_omnidocbench.py `
  evaluation/data/omnidocbench/OmniDocBench.json `
  --images-root evaluation/data/omnidocbench/images `
  --language english `
  --data-source academic_literature `
  --ocr `
  --limit 20 `
  --export-predictions evaluation/results/omnidocbench_predictions `
  --output evaluation/results/omnidocbench_smoke.json
```

扩大到筛选后的全部页面：

```powershell
python evaluation/eval_omnidocbench.py `
  evaluation/data/omnidocbench/OmniDocBench.json `
  --images-root evaluation/data/omnidocbench/images `
  --language english `
  --data-source academic_literature `
  --ocr `
  --export-predictions evaluation/results/omnidocbench_predictions `
  --output evaluation/results/omnidocbench_baseline.json
```

`--ocr` 只调用本地 Tesseract，不调用 LLM。若不加 `--ocr` 且输入只有页面图像，PyMuPDF
解析为空是预期结果，这只能说明“原生文本 parser 不适用于扫描页面”。

### 4.2 评测已有 Markdown parser 输出

如果以后接入 MinerU、Marker 或其他 parser，将每页输出保存成与页面图像同名的 `.md`：

```powershell
python evaluation/eval_omnidocbench.py `
  evaluation/data/omnidocbench/OmniDocBench.json `
  --images-root evaluation/data/omnidocbench/images `
  --predictions-dir path/to/page_markdown `
  --language english `
  --data-source academic_literature `
  --output evaluation/results/omnidocbench_external_parser.json
```

项目桥接报告包含：

- `text_token_f1`；
- `text_token_recall`；
- `ordered_token_similarity`；
- `table_content_coverage`；
- `formula_content_coverage`；
- `empty_prediction_rate`；
- 每页错误和明细。

这些是项目内的快速、确定性指标，不等同于 OmniDocBench 官方的 TEDS、CDM、COCODet。
`--export-predictions` 会导出逐页 Markdown；需要论文可比的正式解析成绩时，再按 OmniDocBench
官方仓库的 `end2end`/`md2md` 配置运行官方 evaluator。官方 evaluator 本身也不需要 LLM API。

## 5. FinanceBench：解析、切片和 evidence 召回

FinanceBench 开源样本带原始 PDF、0-based evidence 页码、原文 evidence 和整页文本。适合同时检查：

```text
PDF 是否解析出 evidence
→ evidence 是否被 chunk 保留
→ evidence chunk 是否进入 Top-K
```

Smoke test：

```powershell
python evaluation/eval_financebench.py `
  evaluation/data/financebench `
  --limit 10 `
  --chunk-sizes 500 `
  --overlap-ratios 0.15 `
  --top-k 5,10 `
  --output evaluation/results/financebench_smoke.json
```

完整 chunk sweep：

```powershell
python evaluation/eval_financebench.py `
  evaluation/data/financebench `
  --chunk-sizes 500,900,1400 `
  --overlap-ratios 0.10,0.15,0.20 `
  --top-k 5,10,20 `
  --mode hybrid `
  --evidence-threshold 0.80 `
  --output evaluation/results/financebench_baseline.json
```

入口会自动把 FinanceBench 的 0-based `evidence_page_num` 加一，与项目内部 1-based
`page_start/page_end` 对齐。报告分三层：

- parsing：`gold_page_nonempty_rate`、`evidence_text_coverage`、`gold_page_token_f1`；
- chunking：evidence preservation/split/lost、页码 metadata、索引/解析字符比；
- retrieval：Paper/Page/Evidence Hit@K、跨页全命中率、MRR。

`evidence_threshold=0.8` 表示一个 chunk 至少覆盖 gold evidence 的 80% token 才算命中。
归一化会处理 Unicode、空白、PDF 断词和连字符，但不会用 LLM 做语义判定。

## 6. FinanceBench 检索消融顺序

固定数据、chunk size、overlap 和 top-k，只改变检索链，依次运行：

```text
--mode sparse --no-rerank
--mode dense --no-rerank
--mode hybrid --no-rerank
--mode hybrid
```

调参顺序：

1. 先用 evidence/page Recall 和 MRR 选择 recall top-k；
2. 再比较 reranker 前后排序、延迟和 backend；
3. 再比较 chunk size 与 overlap；
4. 最后才接生成模型测试答案质量。

不同 chunk size 的 Top-K 所含字符/token 数不同，报告时同时观察
`indexed_to_parsed_char_ratio`，不要只选 Recall 最高的一组。

建议汇总：

| dataset | parser | chunk/overlap | top-k | Page Hit | Evidence Hit | MRR | backend |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| OmniDocBench | 旧结果空过滤，待重跑 | N/A | N/A | N/A | N/A | N/A | Tesseract |
| FinanceBench | PyMuPDF | 1400/140 | 20 | 0.3867 | 0.3867 | 0.1837 | MiniLM + CrossEncoder |

QASPER 仍可作为 clean-text retrieval 回归，但不再作为 PDF parser/chunker 的主评测：

统一入口的 smoke/key profile 会同时运行 QASPER。

## 7. 本地论文 Gold Set

公开数据集不能替代项目真实论文。请编辑：

```text
evaluation/data/business_cases.jsonl
```

至少填写并启用：

```json
{
  "id": "case-001",
  "category": "answerable",
  "question": "真实业务问题",
  "history": [],
  "expected_status": "answered",
  "gold_paper_ids": ["真实 paper_id"],
  "gold_pages": [3],
  "gold_evidence": ["原文中的关键证据子串"],
  "enabled": true
}
```

数据集建议包含：

- answerable；
- unanswerable；
- ambiguous；
- multi-turn reference；
- conflicting evidence；
- table/figure/formula；
- 中英文查询；
- prompt injection。

运行 chunk、overlap 和 top-k sweep：

```powershell
python evaluation/benchmark_grounded.py `
  --chunk-sizes 500,900,1400 `
  --overlap-ratios 0.10,0.15,0.20 `
  --top-k 8,12,24 `
  --output evaluation/results/grounded_chunks.json
```

该脚本使用独立的 `evaluation/results/index-*`，不会覆盖正式 `data/indexes`。

选型标准：

- Recall/MRR 不明显下降；
- 重复 evidence 少；
- page/section metadata 正确；
- p95 latency 可接受；
- 输入 token 和平均费用更低；
- Faithfulness/Correctness 不下降；
- 可答与不可答拒答行为稳定。

当前 `benchmark_grounded.py` 仍以论文级命中为主；本地 gold 的正式比较应参照 FinanceBench
入口使用相同的 evidence/page 指标。公开数据集选完参数后，再用本地用例确认领域迁移没有退化。

## 8. 生成质量与幻觉

该步骤调用真实生成 API 和 RAGAS judge，会计费。先确认：

```powershell
$env:RAG_EVAL_ALLOW_API = "1"
python evaluation/eval_generation.py `
  --limit 5 `
  --output evaluation/results/generation_smoke.json
```

确认正常后扩大：

```powershell
$env:RAG_EVAL_ALLOW_API = "1"
python evaluation/eval_generation.py `
  --limit 20 `
  --output evaluation/results/generation_baseline.json
```

根据安装的 RAGAS 版本，指标可能包括：

- Faithfulness；
- Response Relevancy；
- Context Precision；
- Context Recall；
- Answer Correctness；
- Factual Correctness。

报告必须写实际 RAGAS 版本和实际启用指标。`judge_usage` 为空时，以供应商账单为准。

## 9. 端到端延迟、token 和成本

启动：

```powershell
uvicorn api.main:app
```

评测脚本可能调用真实 API，因此需要显式授权：

```powershell
python evaluation/evaluate.py --profile key --yes
```

记录：

- total latency mean/p50/p95/p99；
- 非流式 `response_headers_ms` p50/p95；
- input/output token；
- estimated cost；
- tool call 数；
- 每个 tool duration；
- retrieval 的 store/embed/dense/sparse/fusion-rerank duration；
- request status。

分别准备以下用例：

- 冷启动第一问；
- 热启动本地检索；
- 二次检索；
- 模糊澄清；
- PDF page/region/image；
- arXiv 摘要；
- arXiv 下载和增量索引；
- 20 轮以上长历史。

当前 `/ask` 是非流式接口，无法测模型首 token。`response_headers_ms` 仅表示客户端收到 HTTP
响应头的时间；要测 TTFT，必须先实现 SSE/流式响应。

### 成本公式

在 `config/config.yaml` 填写：

```yaml
evaluation:
  input_price_per_million: 0
  output_price_per_million: 0
```

计算：

```text
cost =
  input_tokens / 1,000,000 × input_price_per_million
  + output_tokens / 1,000,000 × output_price_per_million
```

最终同时报告 trace estimated cost 和供应商真实账单。

## 10. 人工引用与幻觉审计

从 E2E 结果导出 claim/source CSV：

```powershell
python evaluation/audit_citations.py `
  evaluation/results/e2e_baseline.json `
  --output evaluation/results/citation_audit.csv
```

随机抽检至少 50 条，填写：

```text
supported
partially_supported
unsupported
bad_citation
unanswerable_but_answered
over_refusal
```

计算：

```text
引用编号错误率 = bad_citation / 总样本
不受支持陈述率 = (partially_supported + unsupported) / 总样本
错误作答率     = unanswerable_but_answered / 不可答样本
过度拒答率     = over_refusal / 可答样本
```

自动 token overlap 只用于筛出 `needs_review`，不能替代人工语义判断。

## 11. Badcase 闭环

把失败样本写入：

```text
evaluation/data/badcases.jsonl
```

字段模板已经存在。推荐分类：

- `ambiguous_question`
- `conversation_drift`
- `retrieval_miss`
- `wrong_ranking`
- `duplicate_context`
- `conflicting_evidence`
- `unsupported_claim`
- `bad_citation`
- `false_refusal`
- `tool_timeout`
- `stale_index`
- `prompt_injection`

流程：

```text
固定输入、配置、sources、trace
→ 定位 parser/chunk/recall/rerank/context/generation/harness
→ 添加最小回归样本
→ 单因素修复
→ Ruff + mypy + 对应数据集 smoke evaluation
→ OmniDocBench / FinanceBench / 本地论文 gold set
→ 必要时显式授权 RAGAS
→ 记录修复 commit 和新旧指标
```

## 12. 最终对照表

| 指标 | baseline | candidate | 变化 | 样本数 |
| --- | ---: | ---: | ---: | ---: |
| QASPER Hit@5 | 0.6959 | 待测 | 待测 | 888 |
| QASPER MRR | 0.4710 | 待测 | 待测 | 888 |
| QASPER Recall@5 | 0.5375 | 待测 | 待测 | 888 |
| Faithfulness | 0.7500 | 待测 | 待测 | 5 |
| Answer Correctness | 0.4631 | 待测 | 待测 | 5 |
| 不受支持陈述率 | 无有效人工审计 | 待测 | 待测 | 0 |
| 正确拒答率 | 未覆盖 rejected 样本 | 待测 | 待测 | 0 |
| 回答生成平均 input token | 1004.8 | 待测 | 待测 | 5 |
| 平均费用 | 单价未配置 | 待测 | 待测 | 0 |
| 有效 E2E p50 延迟 | 待重跑 | 待测 | 待测 | 0 |
| 有效 E2E p95 延迟 | 待重跑 | 待测 | 待测 | 0 |

只有同条件 baseline/candidate 对照完成后，才能回答命中率提升、token 压缩、延迟下降和幻觉率变化。
