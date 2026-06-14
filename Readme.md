# Paper RAG Agent

面向**学术论文**的智能问答系统：以 **Claude（Opus 4.8）原生 tool use** 驱动的 Agentic RAG，结合本地论文库语义检索与 arXiv 在线检索，给出**带引用溯源**的回答。

## 核心亮点

- **真 Agentic RAG**：用 Claude 原生 tool use + adaptive thinking，由模型自主决定调用哪个工具、是否多跳检索、是否改写查询，而非关键词规则。
- **多源检索**：本地论文库（向量召回 + 可选重排）+ arXiv 在线检索/下载入库。
- **Corrective RAG（自我纠错）**：检索结果不充分时，模型自动改写查询重试或转向 arXiv。
- **引用溯源**：答案在关键结论后标注 `[paper_id·章节]`，并返回结构化来源（论文标题、章节、链接）。
- **PDF 解析 + 章节分块**：PyMuPDF 解析论文，按章节/段落语义切块并保留元数据。
- **优雅降级**：未配置 `ANTHROPIC_API_KEY` 时自动回退为传统单跳 RAG，便于离线演示。

## 系统架构

```
用户问题
   │
   ▼
PaperRAGAgent ── Claude tool-use 循环 ──┐
   │                                    │ 工具
   │   ┌────────────────────────────────┴───────────────┐
   │   ▼                    ▼                            ▼
 search_local_papers   read_paper_section          search_arxiv
 (本地向量检索+重排)    (精读某篇某章节)            (arXiv 在线检索/下载)
   │
   ▼
带引用的答案 + 推理步骤 + 来源列表
```

## 项目结构

```text
RagAgent/
├── api/            # FastAPI 服务（/ask /collections /ingest_arxiv）
├── agent/          # PaperRAGAgent（agentic loop）、系统提示
├── retrieval/      # 论文加载/章节分块/向量/重排/检索
├── tools/          # Claude 可调用工具：本地检索 / arXiv / 章节精读
├── llm/            # LLM 统一封装（Anthropic / OpenAI / 本地降级）
├── indexing/       # 索引构建与集合管理
├── evaluation/     # 检索评估脚本与数据
├── frontend/web/   # 前端单页（原生 HTML/JS，FastAPI 托管于 /ui）
├── config/         # 配置
├── data/papers/    # 论文集合（每个子目录是一个 collection）
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备论文

把论文（PDF / txt / md）放到某个集合目录下，例如 `data/papers/demo/`。仓库已内置两篇示例（`rag_survey.txt`、`transformer.txt`）。

### 3. 构建索引

```bash
python3 indexing/build_index.py          # 默认 demo 集合
python3 indexing/build_index.py demo     # 指定集合
```

### 4. 配置大模型（启用 Agentic 模式）

支持两类可做工具调用的后端，二选一即可，共用同一个 agentic 循环：

**Claude（Anthropic）**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# 或用订阅登录：ant auth login（SDK 自动识别 ~/.config/anthropic 凭据）
```

**OpenAI 兼容（DeepSeek / Qwen / 本地开源模型）** —— 改 `config/config.yaml` 的 `llm` 段：

```yaml
llm:
  provider: openai
  model_name: deepseek-chat                 # 或 qwen-plus / qwen2.5 ...
  openai_api_base: https://api.deepseek.com  # Qwen: https://dashscope.aliyuncs.com/compatible-mode/v1
  openai_api_key: ""                         # 或 export OPENAI_API_KEY=...
```

本地模型示例（Ollama）：先 `ollama pull qwen2.5 && ollama serve`，再设
`model_name: qwen2.5`、`openai_api_base: http://localhost:11434/v1`、`openai_api_key: ollama`。

未配置任何后端、或后端调用失败时，系统自动降级为本地检索 RAG（不中断请求）。

### 5. 启动 API

```bash
uvicorn api.main:app --reload
```

### 6. 调用接口

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"RAG 相比纯参数化模型的优势是什么？","collection":"demo"}'
```

### 7. 从 arXiv 在线入库

```bash
curl -X POST http://127.0.0.1:8000/ingest_arxiv \
  -H "Content-Type: application/json" \
  -d '{"query":"retrieval augmented generation","collection":"arxiv","max_results":3}'
```

下载论文 PDF 到 `data/papers/<collection>/` 并自动重建该集合索引。

### 8. 运行评估

```bash
python3 evaluation/eval.py demo
```

### 9. 打开网页界面

前端由 API 一并托管，启动第 5 步后直接浏览器打开：

```
http://localhost:8000/ui/
```

类 Claude 的对话界面：底部输入框、聊天历史、答案中**重要句子后内联出现 📄 引用图标**，点击在右侧抽屉弹出对应 PDF 页并高亮原文（非 PDF 来源则显示引用片段）。

## API 说明

### `POST /ask`

```json
{ "question": "自注意力是如何工作的？", "collection": "demo" }
```

响应：

```json
{
  "collection": "demo",
  "answer": "...",
  "steps": ["Planner: 启动 Claude agentic 检索循环", "Tool[search_local_papers] ..."],
  "sources": [
    {
      "paper_id": "transformer",
      "paper_title": "Attention Is All You Need: ...",
      "section": "2 Method",
      "source": ".../data/papers/demo/transformer.txt",
      "score": 0.87
    }
  ]
}
```

### `GET /collections`

返回 `data/papers/` 下可用论文集合。

### `POST /ingest_arxiv`

```json
{ "query": "...", "collection": "arxiv", "max_results": 3 }
```

## Agent 设计

`PaperRAGAgent` 维护一个手写的 agentic 循环（最多 `llm.max_tool_iters` 轮）：

1. 调用 Claude（`tools=[search_local_papers, read_paper_section, search_arxiv]`，adaptive thinking）。
2. 若返回 `tool_use`：执行对应工具，把结果作为 `tool_result` 回传，并记录 steps 与 sources。
3. 模型可据检索质量自行改写查询、换工具、多跳，直到 `end_turn` 给出最终答案。

多跳检索、查询改写、Corrective RAG、引用整合都由模型在循环中自然完成。

## 配置说明

`config/config.yaml` 关键字段：

- `project.default_collection`：默认论文集合。
- `index.chunk_size/chunk_overlap`：切块策略（按字符）。
- `index.top_k_recall/top_n_rerank`：召回与重排数量。
- `embedding.use_sentence_transformers`：是否使用语义嵌入模型（否则回退哈希向量）。
- `rerank.use_cross_encoder`：是否启用 CrossEncoder 重排。
- `llm.provider`：`anthropic` / `openai` / `local`。
- `llm.model_name` / `effort` / `max_tool_iters`：模型、思考强度、循环上限。
- `arxiv.max_results` / `download_dir`：arXiv 默认返回数量与下载目录。

## 许可证

MIT
