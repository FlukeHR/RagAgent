# RAG 真实评测指南

本指南由项目维护者执行。代码和评测入口已经准备好，但仓库没有真实 baseline/candidate 结果。
第一次运行只能称为 baseline；只有在同一数据、模型、硬件和运行方式下比较，才能报告“提升了多少”。

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

评测依赖：

```powershell
pip install -r requirements-eval.txt
```

可选 FAISS：

```powershell
pip install -r requirements-optional.txt
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

## 3. QASPER 检索评测

把官方 QASPER dev 数据放到：

```text
evaluation/data/qasper/qasper-dev-v0.3.json
```

先跑小样本：

```powershell
python evaluation/eval_qasper.py `
  --limit 10 `
  --mode hybrid `
  --output evaluation/results/qasper_smoke.json
```

完整 baseline：

```powershell
python evaluation/eval_qasper.py `
  --limit 0 `
  --mode hybrid `
  --output evaluation/results/qasper_baseline.json
```

核心指标：

- Hit@k：top-k 是否至少命中一个 gold evidence；
- MRR：第一个正确 evidence 的倒数排名；
- nDCG@k：正确 evidence 的排序质量；
- Recall@k：gold evidence 找回比例；
- Mean top confidence：最高结果的校准后置信度，仅用于后续阈值标定。

## 4. Retrieval 消融与 top-k/top-n

一次运行 Dense、BM25、Hybrid、Hybrid+reranker：

```powershell
python evaluation/benchmark_retrieval.py `
  --limit 50 `
  --top-k 8,12,24 `
  --top-n 3,5,8 `
  --output evaluation/results/retrieval_ablation.json
```

候选链：

```text
Dense only
BM25 only
Dense + BM25 + RRF
Dense + BM25 + RRF + reranker
```

调参顺序：

1. 先用 Recall@k/MRR 选 recall top-k；
2. 再用 nDCG、生成质量、延迟和 token 选 rerank top-n；
3. 不要先跑完整参数笛卡尔积；
4. 不要只选 Recall 最高的配置。

建议报告：

| mode | top-k | top-n | Hit@k | MRR | Recall@k | backend |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dense | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| BM25 | 待测 | 待测 | 待测 | 待测 | 待测 | BM25 |
| Hybrid | 待测 | 待测 | 待测 | 待测 | 待测 | RRF |
| Hybrid+rerank | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

## 5. 生产 chunk 策略评测

QASPER 使用标注段落作为 chunk，不能证明生产 900/150 最优。请编辑：

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

## 6. 生成质量与幻觉

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

## 7. 端到端延迟、token 和成本

启动：

```powershell
uvicorn api.main:app
```

评测脚本可能调用真实 API，因此需要显式授权：

```powershell
python evaluation/benchmark_e2e.py `
  --yes `
  --repeat 5 `
  --output evaluation/results/e2e_baseline.json
```

记录：

- total latency mean/p50/p95/p99；
- first-byte p50/p95；
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

`first-byte` 是 HTTP 首字节，不等价于流式模型首 token；当前 `/ask` 是非流式接口，报告中需使用正确名称。

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

## 8. 人工引用与幻觉审计

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

## 9. Badcase 闭环

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
→ Ruff + unittest + mypy
→ QASPER
→ 必要时显式授权 RAGAS
→ 记录修复 commit 和新旧指标
```

## 10. 最终对照表

| 指标 | baseline | candidate | 变化 | 样本数 |
| --- | ---: | ---: | ---: | ---: |
| Hit@k | 待测 | 待测 | 待测 | 待测 |
| MRR | 待测 | 待测 | 待测 | 待测 |
| Recall@k | 待测 | 待测 | 待测 | 待测 |
| Faithfulness | 待测 | 待测 | 待测 | 待测 |
| Answer Correctness | 待测 | 待测 | 待测 | 待测 |
| 不受支持陈述率 | 待测 | 待测 | 待测 | 待测 |
| 正确拒答率 | 待测 | 待测 | 待测 | 待测 |
| 平均 input token | 待测 | 待测 | 待测 | 待测 |
| 平均费用 | 待测 | 待测 | 待测 | 待测 |
| p50 延迟 | 待测 | 待测 | 待测 | 待测 |
| p95 延迟 | 待测 | 待测 | 待测 | 待测 |

只有同条件 baseline/candidate 对照完成后，才能回答命中率提升、token 压缩、延迟下降和幻觉率变化。

