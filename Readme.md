# Code RAG Agent

面向代码仓库的智能问答系统，基于 RAG（检索增强生成）+ Agent 多步推理，实现对本地代码的检索、分析和回答。

## 功能特性

- 代码语义检索：向量召回 + 可选重排。
- 多步推理流程：Planner -> Retriever -> Judge -> Generator。
- 工程化结构：API、Agent、Retrieval、LLM 解耦。
- 多仓库扩展：支持在 `data/` 下放置多个代码仓库。
- 评估能力：内置 Recall/MRR/Hit Rate 评估脚本。

## 系统流程

1. 用户提交问题。
2. Agent 规划是否需要多跳检索。
3. 检索模块完成召回和重排。
4. LLM 根据上下文生成答案。
5. 返回答案、步骤、证据来源。

## 项目结构

```text
RagAgent/
├── api/                  # FastAPI 服务
├── agent/                # 推理流程与状态
├── retrieval/            # 加载、切块、向量、重排、检索
├── tools/                # Agent 可调用工具
├── llm/                  # LLM 统一封装
├── indexing/             # 索引构建与管理
├── evaluation/           # 评估脚本与数据
├── frontend/             # Streamlit 页面
├── config/               # 配置
├── data/                 # 代码仓库样例与索引
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 构建索引

```bash
python3 indexing/build_index.py
```

### 3. 启动 API

```bash
uvicorn api.main:app --reload
```

### 4. 调用接口

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"FastAPI 路由是如何定义的？"}'
```

### 5. 查看仓库列表

```bash
curl http://127.0.0.1:8000/repos
```

### 6. 运行评估

```bash
python3 evaluation/eval.py
```

### 7. 启动前端（可选）

```bash
streamlit run frontend/streamlit_app.py
```

## API 说明

### `POST /ask`

请求示例：

```json
{
  "question": "FastAPI 中间件是如何工作的？",
  "repo_name": "fastapi"
}
```

响应示例：

```json
{
  "repo_name": "fastapi",
  "answer": "...",
  "steps": ["Planner: multi-hop=True", "Retriever: hop=1, got=5"],
  "sources": [
    {
      "file_path": ".../data/fastapi/middleware.py",
      "start_line": 1,
      "end_line": 10,
      "score": 0.91
    }
  ]
}
```

### `GET /repos`

返回 `data/` 下可用仓库列表。

## 配置说明

主配置文件是 `config/config.yaml`，关键字段：

- `project.default_repo`: 默认检索仓库。
- `index.chunk_size/chunk_overlap`: 切块策略。
- `index.top_k_recall/top_n_rerank`: 召回与返回数量。
- `embedding.use_sentence_transformers`: 是否使用语义嵌入模型。
- `rerank.use_cross_encoder`: 是否启用 CrossEncoder 重排。
- `llm.provider`: `local` 或 `openai`。

## 许可证

MIT