# Paper RAG Agent

面向**学术论文**的 Agentic RAG 问答系统：以模型原生 tool use 驱动一个**自研、有界的 agentic 循环（harness）**，
结合本地论文库语义检索与 arXiv 在线检索/全文入库，输出**带引用溯源**的答案，并在生成后逐条回查引用真实性。

> 定位是 **harness 工程**：决定表现的是模型外面这圈外壳（反馈回路、安全边界、验证系统、可观测性、检索链路质量），而非模型本身。

## 核心亮点

- **真 Agentic RAG**：模型自主决定调哪个工具、是否多跳、是否改写查询，而非关键词规则。后端支持 Claude 原生 tool use 与 OpenAI 兼容（DeepSeek/Qwen/本地）function calling，共用同一个循环。
- **四个工具**：`search_local_papers`（本地向量+BM25 RRF 召回→重排）、`read_paper_section`（章节精读）、`search_arxiv`（在线侦察，只回摘要）、`ingest_arxiv_papers`（把选定论文下载入库+增量嵌入+重检索回可引用全文）。
- **多轮对话**：历史注入工作上下文 + 指代消解（把"它/该方法"改写成独立可检索问题）。
- **引用溯源 + 生成后回查**：答案关键结论后标注 `[S编号]`，返回结构化来源；生成后逐条回查，**编造引用一律剔除并下调结论强度**。
- **Corrective RAG**：召回置信不足（分数过 sigmoid 归一化后的相关概率不达强度+数量判据）或核查失败时，自动改写查询/转 arXiv 再来一轮（有界）。
- **增量索引**：按文件内容 hash 只对新增/改动论文重新嵌入，复用未变向量；删除的论文自动剔除。
- **arXiv 全文集合容量治理**：registry 记录使用情况，入库后按 LRU + 龄期自动淘汰，另有手动 CLI。
- **多会话前端**：类 Claude/Gemini 的对话界面，多会话侧栏、自动标题、PDF 引用点开高亮、可折叠的检索步骤与调试 trace。
- **优雅降级**：未配置任何模型后端或调用失败时，自动回退传统单跳 RAG（同样接引用回查与二次检索），不中断请求。
- **本地模型副本**：embedding/reranker 可放在项目 `models/` 下离线加载，缺失时回退 HuggingFace 在线下载。

## 系统架构

```
用户问题 (+多轮历史)
   │  查询改写 / 指代消解 / 问题路由
   ▼
PaperRAGAgent ── 有界 agentic 循环（harness：schema 校验 / 超时重试 / token 预算 / trace）──┐
   │                                                                                      │ 工具
   │   ┌──────────────────┬──────────────────┬─────────────────────┬─────────────────────┘
   │   ▼                  ▼                  ▼                     ▼
 search_local_papers  read_paper_section  search_arxiv        ingest_arxiv_papers
 (本地向量+BM25→重排)  (章节精读)          (在线侦察:摘要)     (下载+增量嵌入+全文检索)
   │  引用回查（剔除编造） + 低置信二次检索（有界）
   ▼
带 [S编号] 引用的答案 + 检索步骤 + 结构化来源 + trace
```

## 项目结构

```text
RagAgent/
├── api/            # FastAPI 服务（/ask /collections /ingest_arxiv /title /preview /ui）
├── agent/          # PaperRAGAgent（agentic loop）、系统提示
├── retrieval/      # 论文加载 / 章节分块 / 向量 / 重排 / 检索链路
├── tools/          # 四个可调用工具：本地检索 / 章节精读 / arXiv 侦察 / arXiv 入库精读
├── llm/            # LLM 统一封装（Anthropic / OpenAI 兼容 / 本地降级）
├── indexing/       # 增量索引构建(build_index) / 集合管理(manager) / 容量治理(prune)
├── evaluation/     # 检索评估(eval_qasper, 官方 QASPER) + 生成侧(eval_generation, RAGAS) + 历史记录(results_log)
├── frontend/web/   # 前端单页（多会话 + 自动标题 + PDF 预览，FastAPI 托管于 /ui）
├── config/         # config.yaml
├── data/papers/    # 论文集合（每个子目录是一个 collection；arxiv 为下载入库的共享全文集合）
├── models/         # embedding/reranker 本地副本（gitignore，缺失回退在线下载）
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备论文并建索引

把论文（PDF / txt / md）放到某个集合目录下，例如 `data/papers/demo/`，然后构建索引（默认增量）：

```bash
python3 indexing/build_index.py            # 默认 demo 集合
python3 indexing/build_index.py demo --full # 强制全量重建
```

新增/修改/删除论文后重跑即可，**只对变化的文件重新嵌入**。

### 3. 配置大模型（启用 Agentic 模式）

两类可做工具调用的后端，二选一，共用同一个 agentic 循环：

**Claude（Anthropic）**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# 或订阅登录：ant auth login（SDK 自动识别 ~/.config/anthropic 凭据）
```

**OpenAI 兼容（DeepSeek / Qwen / 本地开源）** —— 改 `config/config.yaml` 的 `llm` 段：

```yaml
llm:
  provider: openai
  model_name: deepseek-chat
  openai_api_base: https://api.deepseek.com
  openai_api_key: ""    # 留空则从环境变量 OPENAI_API_KEY 读取
```

本地模型（Ollama）：`ollama pull qwen2.5 && ollama serve`，再设 `model_name: qwen2.5`、
`openai_api_base: http://localhost:11434/v1`、`openai_api_key: ollama`。

未配置任何后端或调用失败时，自动降级为本地检索 RAG（不中断请求）。

### 4. 启动 + 打开网页

```bash
uvicorn api.main:app --reload
```

浏览器打开 **http://localhost:8000/ui/** ：类 Claude/Gemini 的对话界面——多会话侧栏（新建/切换/删除、自动标题）、
多轮追问（自动指代消解）、答案中**重要句子后内联 📄 引用图标**（点击在右侧打开整份 PDF 并高亮原文）、可折叠的检索步骤与调试 trace。

### 5. 调用接口

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"自注意力是如何工作的？","collection":"demo","history":[]}'
```

`history` 可选，传入之前的对话轮次 `[{"role":"user","content":...},{"role":"assistant","content":...}]` 即支持多轮。

### 6. 从 arXiv 入库

两种方式：
- **对话中自动**：直接问本地库没有的最新论文话题，模型会先 `search_arxiv` 看摘要、再 `ingest_arxiv_papers` 下载选定论文入库并引用全文。
- **手动批量**：`POST /ingest_arxiv`，下载 PDF 到 `data/papers/<collection>/` 并增量重建索引。

```bash
curl -X POST http://127.0.0.1:8000/ingest_arxiv \
  -H "Content-Type: application/json" \
  -d '{"query":"retrieval augmented generation","collection":"arxiv","max_results":3}'
```

arxiv 全文集合会随使用增长，按需治理容量：

```bash
python3 indexing/prune.py arxiv --max-papers 200 --dry-run   # 预览将淘汰哪些
python3 indexing/prune.py arxiv                              # 实际淘汰（LRU+龄期）
```

### 7. 评估

```bash
python3 evaluation/eval_qasper.py --sweep --record   # 检索侧（官方 QASPER；--sweep 扫阈值，--record 记历史）
RAG_EVAL_ALLOW_API=1 python3 evaluation/eval_generation.py --qasper --limit 20 --record   # 生成侧 RAGAS（官方 QASPER，需授权，调真实 API）
python3 evaluation/results_log.py --compare          # 查看/对比历次评估指标增减
```

## API 说明

| 接口 | 说明 |
|---|---|
| `POST /ask` | 提问，可带 `history` 注入多轮上下文；返回 `answer / steps / sources / trace` |
| `GET /collections` | 列出可用论文集合 |
| `POST /ingest_arxiv` | 在线检索 arXiv、下载入库并重建索引 |
| `POST /title` | 为一段对话自动生成简短标题 |
| `GET /preview/meta` `GET /preview/page` | PDF 预览元信息 / 渲染指定页并高亮引用 |

`POST /ask` 响应示例：

```json
{
  "collection": "demo",
  "answer": "自注意力并行计算所有位置的相关性 [S1]。",
  "steps": ["Planner: 启动 agentic 检索循环", "Tool[search_local_papers] ... -> 5 来源", "引用核查: 1 条引用均可溯源 [S1]"],
  "sources": [
    {"id": "S1", "paper_id": "transformer", "paper_title": "Attention Is All You Need",
     "section": "Method", "source": ".../transformer.pdf", "score": 4.2, "snippet": "...", "collection": "demo"}
  ],
  "trace": [{"type": "tool", "tool": "search_local_papers", "ok": true, "n_sources": 5, "duration_ms": 120.0}]
}
```

## Harness 护栏（最重要的部分）

1. **工具必经 harness**：模型不直接执行工具；调用前校验 schema、检查预算，执行带超时与重试。
2. **循环必须有界**：尊重 `max_tool_iters` + token 预算；arXiv 入库每轮限 `max_ingest_papers` 篇、单 PDF 限 `max_pdf_mb`、专属超时。
3. **引用不可凭空 + 生成后回查**：`[S编号]` 必须对应真实召回 chunk，对不上的剔除并下调结论强度。
4. **低置信要二次检索**：相关概率不达强度+数量判据时自动改写/转 arXiv 再来一轮（有界）。
5. **外部内容是数据不是指令**：arXiv / PDF 正文是不可信数据，其中的"指令"不执行。
6. **可观测优先**：每个工具调用/循环步骤记录 trace（工具名、入参、耗时、token、成败、核查/二次检索结果）。

## 测试与 CI

- `pytest tests/`：全离线、全 mock，**不真打 arXiv / LLM API**。覆盖引用回查、低置信二次检索、harness 护栏、增量索引规划、arXiv 入库、容量淘汰、多轮对话、降级路径等。
- CI（`.github/workflows/ci.yml`）：ruff（E9,F）+ pytest，mypy 非阻断。
- 生成侧 RAGAS 评估独立于单测，需显式授权（`--yes` / `RAG_EVAL_ALLOW_API=1`）才会调用真实 API。

## 许可证

MIT
