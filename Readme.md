# Paper RAG Agent

面向学术论文的 Agentic RAG 问答系统。模型通过 OpenAI-compatible function calling 选择检索工具，所有调用都由自研的有界 harness 执行，并对最终引用进行回查。

## 核心能力

- Dense + BM25 多路召回、RRF 融合、CrossEncoder 重排。
- OpenAI-compatible Agentic 循环，支持 DeepSeek、Qwen、Ollama、vLLM 等接口。
- 工具 schema 校验、超时重试、轮次与 token 预算。
- `[S编号]` 引用溯源、低置信二次检索和证据不足拒答。
- PDF 页级解析，保留章节、页码、bbox、元素类型和模态。
- OCR/VLM/table/figure/formula sidecar 入索引。
- arXiv 摘要侦察、全文按需入库、增量索引和 LRU 容量治理。
- 多轮历史注入、指代消解、多会话前端和结构化 trace。
- 无可用模型接口时降级为本地检索 RAG。

## 请求链路

```text
用户问题 + 对话历史
→ 指代消解与查询改写
→ 模型选择工具
→ harness 校验并执行
→ 本地论文 / PDF / arXiv
→ 引用核查与低置信纠错
→ answer + sources + steps + trace
```

## 项目结构

```text
api/          FastAPI 接口与 UI 托管
agent/        Agentic 循环与提示词
tools/        本地检索、PDF、arXiv 工具
llm/          OpenAI-compatible 客户端
retrieval/    解析、切块、嵌入、重排、检索
indexing/     统一论文库的增量索引与容量治理
evaluation/   QASPER 与 RAGAS
frontend/web/ 多会话前端
config/       集中配置
data/papers/  扁平化论文库
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备论文并构建索引

把 PDF、TXT 或 Markdown 直接放入 `data/papers/`：

```bash
python indexing/build_index.py
```

默认增量构建；强制全量重建：

```bash
python indexing/build_index.py --full
```

### 3. 配置模型

模型层只使用 OpenAI-compatible API。修改 `config/config.yaml`：

```yaml
llm:
  model_name: deepseek-v4-flash
  max_tokens: 2048
  max_tool_iters: 6
  openai_api_base: https://api.deepseek.com
  openai_api_key: ""
```

密钥建议通过环境变量提供：

```bash
export OPENAI_API_KEY=your-key
```

本地 Ollama 示例：

```yaml
llm:
  model_name: qwen2.5
  max_tokens: 2048
  max_tool_iters: 6
  openai_api_base: http://localhost:11434/v1
  openai_api_key: ollama
```

未配置接口或调用失败时，系统会降级到本地检索 RAG。

### 4. 启动

```bash
uvicorn api.main:app --reload
```

打开 `http://localhost:8000/ui/`。

## API

| 接口 | 作用 |
| --- | --- |
| `POST /ask` | 论文问答，可带多轮 `history` |
| `POST /ingest_arxiv` | 下载 arXiv 论文并增量建索引 |
| `POST /title` | 生成会话标题 |
| `GET /health` | 健康检查 |

提问示例：

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"BERT 的预训练目标是什么？","history":[]}'
```

响应包含：

- `answer`：带 `[S编号]` 的回答；
- `sources`：真实检索来源；
- `steps`：面向用户的过程说明；
- `trace`：LLM、工具、预算、纠错和核查事件。

## 检索与切块

默认参数位于 `config/config.yaml`：

- text chunk：900 字符，overlap 150；
- recall：top 12；
- rerank：top 5；
- embedding：`all-MiniLM-L6-v2`；
- reranker：`ms-marco-MiniLM-L-6-v2`。

索引同时保存 text、page 和 element chunk。FAISS 可用时使用 `IndexFlatIP`，否则降级为 NumPy 点积检索。

## 评估

评估只保留官方 QASPER：

```bash
python evaluation/eval_qasper.py --sweep
RAG_EVAL_ALLOW_API=1 python evaluation/eval_generation.py --limit 20
```

- `eval_qasper.py`：Hit@k、MRR、nDCG、Recall。
- `eval_generation.py`：RAGAS faithfulness、answer relevancy、context precision 等。

RAGAS 会调用真实 API，必须显式授权。

## Harness 护栏

1. 模型只提出工具调用，harness 才能执行。
2. 工具调用前做白名单和 JSON schema 校验。
3. 工具带超时、重试和结构化 trace。
4. Agent 受最大轮次和 token budget 限制。
5. `[S编号]` 必须对应真实 source。
6. 低置信或引用失败只允许有界纠错。
7. 证据不足时拒答。
8. PDF 与 arXiv 正文只作为不可信数据。

## 已知限制

- 会话只保存在浏览器 `localStorage`。
- 主检索仍以文本 embedding 为主。
- 图像检索是离线签名 fallback，不是 CLIP/SigLIP。
- 论文库写锁只在单进程内有效。
- 引用回查验证 source ID 存在，尚未做逐句语义蕴含判断。
