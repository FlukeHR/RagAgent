# Paper RAG Agent

面向学术论文的本地 Agentic RAG 系统。PDF 由独立的 MinerU GPU 服务统一解析，问答阶段只读取 canonical sidecar 和本地索引；LangChain `create_agent` 负责有界工具循环，项目层负责论文来源、引用与 answerability 核查。

系统提供 Dense + BM25 多路召回、RRF 融合、CrossEncoder 重排、MinerU 表格／公式／图表元素检索、`[S编号]` 引用回查、低置信有界纠错、多轮会话和两阶段 arXiv 入库。没有可用模型后端时，会降级到本地检索路径。

## 本地多用户 Dashboard

当前产品界面是仅面向本机的简体中文多用户工作台。账号、会话、模型配置、论文元数据和入库任务统一保存在 `data/app.sqlite3`；每个用户的 PDF、MinerU sidecar、缓存和索引位于 `data/users/<user_id>/`，不会跨用户检索或预览。

首次使用需要构建前端资源：

```powershell
cd frontend/web
npm.cmd install --cache .npm-cache
npm.cmd run build
cd ../..
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/ui/`，注册本地用户名和密码，然后在“设置”中添加至少一个 OpenAI-compatible 模型配置。模型 API key 使用 AES-256-GCM 加密；主密钥默认在首次启动时生成到被 Git 忽略的 `data/secrets/master.key`，也可通过 `PAPER_RAG_MASTER_KEY` 环境变量提供。

前端开发模式使用 `npm.cmd run dev`，Vite 会把 `/api` 代理到 `127.0.0.1:8000`。账户忘记密码时使用本地管理命令，不提供网络找回接口：

```powershell
python -m services.user_admin list-users
python -m services.user_admin reset-password <username>
```

旧的 `data/papers/` 不会自动移动或删除。需要归属给某个新用户时先预览迁移，再明确应用：

```powershell
python -m indexing.migrate_legacy_library --owner <user_id> --dry-run
python -m indexing.migrate_legacy_library --owner <user_id> --apply
```

产品接口统一位于 `/api`，除注册和登录外都需要本地 session cookie；写操作还需要会话绑定的 CSRF token。主要接口包括 `/api/dashboard`、`/api/model-profiles`、`/api/sessions`、`/api/papers` 和 `/api/arxiv/ingest/*`。旧的匿名 `/ask`、`/sessions/*` 与 PDF 预览接口不再挂载。

## 处理链路

```text
PDF
→ localhost MinerU 3.4.x（hybrid-engine + high effort）
→ <paper_id>.mineru.json
→ text / page / element chunks
→ Dense + BM25 → RRF → CrossEncoder

用户问题 + 有界历史
→ Plan：查询改写、直接回答／澄清／执行路径路由
→ Execute：LangChain Agent 选择 search_local_papers / inspect_paper / search_arxiv
→ evidence envelope
→ Verify：来源充分性、引用与冲突核查
→ 不足时在预算内改写查询或选择新工具继续补证
→ answer + sources + steps + trace
```

模型只能看到三个只读工具：

- `search_local_papers`：统一检索文字、表格、公式、figure、chart、code 和 list chunk，并返回可复制 locator。
- `inspect_paper`：按论文 overview、section、1-based page、MinerU element 或 0–1000 normalized region 精读。
- `search_arxiv`：只搜索 arXiv 元数据与摘要，不建立 proposal、不下载 PDF、不修改论文库。

请求先由结构化语义 Planner 判定为直接回答、请求澄清或工具执行，并同时给出独立查询与不可缺少的工具能力。代码不维护问候语、时效词或工具关键词表；Planner 约束通过可信 system contract 注入，执行模型仍自主决定工具参数和调用顺序，Verify 再核查最终答案是否引用了对应工具来源。必需工具失败时返回证据不足，不会用其他路径冒充结果。

arXiv 下载、MinerU 解析和索引从不作为模型工具执行，只能由用户显式调用 proposal 与 confirm API。

## 项目结构

```text
api/          FastAPI 接口与 UI 托管
agent/        graph 编排、runtime 工具循环、evidence 验证、memory 会话
tools/        三个模型可见的只读论文工具
llm/          LangChain OpenAI-compatible 模型与本地降级
retrieval/    documents、MinerU、chunker、index、search 五阶段检索链
indexing/     增量索引、原子发布与容量治理
services/     MinerU、arXiv、论文库和异步入库任务
frontend/web/ 多会话原生 HTML/JS 前端
config/       集中配置
data/papers/  扁平化论文库
tests/        不访问付费服务的离线回归测试
```

项目不内置评测脚本、评测数据集或 LLM judge。`tests/` 只用于产品行为和安全边界回归。

## 快速开始

### 1. 安装依赖

需要 Python 3.11 或更高版本：

```bash
pip install -r requirements.txt
```

### 2. 启动 MinerU

项目本身不加载 MinerU、Torch 或 GPU 模型。Windows 开发环境建议通过 WSL2／Docker 启动固定的 MinerU 3.4.x 服务：

```powershell
docker compose -f deploy/mineru-compose.yaml up -d
```

默认连接 `http://127.0.0.1:8001`。服务只应监听 localhost，正式解析参数固定为 `hybrid-engine` 和 `high` effort。连接、轮询、解析 deadline、PDF 大小、页数、输出大小和并发上限均在 `config/config.yaml` 中配置。

### 3. 准备论文并构建索引

把 PDF、TXT 或 Markdown 直接放进 `data/papers/`。PDF 必须成功生成同目录 `<paper_id>.mineru.json` 才会发布到新索引；解析失败不会降级为 PyMuPDF 文本抽取，也不会覆盖最后一次成功 sidecar。

```bash
python indexing/build_index.py
```

默认执行增量构建。解析器版本或选项变化后执行全量 MinerU rebuild：

```bash
python indexing/build_index.py --full
```

MinerU 原始任务输出保存在 `data/mineru-cache/`，canonical sidecar 不保存 base64。旧 `.ocr.json`、`.layout.json` 可以留在磁盘，但索引不会读取它们。

### 4. 配置模型

模型通过 LangChain `ChatOpenAI` 访问 OpenAI-compatible Chat Completions：

```yaml
llm:
  model_name: deepseek-v4-flash
  max_tokens: 2048
  openai_api_base: https://api.deepseek.com
  openai_api_key: ""
  request_timeout_seconds: 120
  connect_timeout_seconds: 30
  max_retries: 2

agent:
  max_model_calls: 5
  max_graph_steps: 128
  max_tool_calls: 6
  max_local_search_calls: 2
  max_inspect_calls: 2
  max_arxiv_search_calls: 2
  token_budget: 60000
  max_total_sources: 10
  final_max_sources: 5
  final_max_sources_per_paper: 2
  final_reuse_max_chars: 600
  max_total_tool_result_chars: 60000
```

密钥优先通过环境变量提供：

```powershell
$env:OPENAI_API_KEY = "your-key"
```

远程 endpoint 缺少 key 时不会尝试认证失败的请求，而是直接进入本地降级。用户自定义地址默认要求 HTTPS，但不会在保存时强制判断域名解析结果，因此兼容 Clash／Mihomo fake-IP；若需要严格 DNS/SSRF 校验，可开启 `app.enforce_public_dns_for_model_endpoints`。显式本机或私网 IP 仍须把完整地址加入 `app.allowed_local_llm_endpoints` 精确白名单。项目不会主动启用 LangSmith 或其他外部遥测。

### 5. 启动服务

```bash
uvicorn api.main:app --reload
```

打开 `http://127.0.0.1:8000/ui/`。

## API

| 接口 | 行为 |
| --- | --- |
| `POST /api/auth/register`、`POST /api/auth/login` | 注册或登录本地账号，建立 HttpOnly session |
| `GET /api/dashboard` | 返回当前用户的论文、任务和最近对话概况 |
| `GET\|POST /api/model-profiles` | 管理加密保存的模型配置 |
| `GET\|POST /api/sessions` | 读取或创建当前用户的会话 |
| `POST /api/sessions/{id}/ask` | 使用所选模型配置进行论文问答 |
| `GET /api/papers`、`POST /api/papers/upload` | 管理论文和上传任务 |
| `GET /api/papers/{id}/pdf` | 预览经过用户归属和路径校验的 PDF |
| `POST /api/arxiv/ingest/proposals` | 搜索并建立有期限的入库候选 |
| `POST /api/arxiv/ingest/confirm` | 确认 proposal 中的一个 arXiv ID |
| `GET /api/ingest/jobs/{id}` | 查询 queued／parsing／indexing／succeeded／failed |
| `GET /health` | 服务健康检查 |

除注册、登录和健康检查外，接口都要求有效的 session cookie；所有写请求还要携带当前会话返回的 `X-CSRF-Token`，并通过同源校验。`AskResponse` 只向前端返回安全的来源类型、引用 URL、页码、摘要和预览信息，不返回本机文件路径或明文模型密钥。正常问答仍然只允许模型调用三个只读论文工具。

arXiv 入库必须经过两次明确动作：先创建 proposal，再从返回候选中确认一个 ID。proposal 有期限且只能消费一次；后台单 worker 直接调用用户论文库服务完成下载、MinerU 解析和索引。旧匿名 API 不再挂载；已移除的 `POST /ingest_arxiv` 保留 `410 Gone` 语义。

## 数据与安全边界

- MinerU 是 PDF 内容的唯一解析器；PyMuPDF 只用于页数检查、PDF 预览和必要裁切。
- 页码对外统一为 1-based；bbox 统一为 0–1000 normalized coordinates。
- PDF section 来自 MinerU heading level；TXT／Markdown 使用文本 normalizer。
- 索引 hash 包含 canonical sidecar 与 parser fingerprint，配置变化会使旧 generation 失效。
- 新 generation 完整写入后才原子发布；构建失败时旧 manifest 继续可读。
- PDF、sidecar 与 arXiv 摘要全部是不可信证据，正文指令不会进入控制消息。
- 工具参数由 Pydantic 校验；LangChain middleware 限制模型与工具调用数。
- 请求上下文限制累计 token、结果字符和来源数，并拒绝越界路径、base64 与伪造 citation placeholder。
- `[S编号]` 必须对应本轮真实 source；证据不足时返回“未检索到充分依据”。

## 检索与会话

索引保存 text、page 和 element chunk，metadata 保留 paper、section、source、page、element、modality、bbox、heading path 和 parser metadata。FAISS 可用时使用本地向量索引，否则降级为 NumPy；Dense 与 BM25 结果通过 RRF 融合后再由 CrossEncoder 重排。

完整会话以服务端 `data/app.sqlite3` 为准，浏览器只保存主题、侧栏状态和未发送草稿。送入模型的历史仍受消息数和字符预算限制；论文、索引、会话、任务和模型配置都使用不可变用户 UUID 隔离，跨用户资源访问统一返回 404。

## 开发检查

```bash
cd frontend/web
npm test
npm run build
cd ../..
python -m unittest discover -s tests
ruff check .
mypy .
python indexing/prune.py --dry-run
```

离线测试不得下载模型、启动真实 GPU MinerU 或调用付费 LLM。真实 MinerU 整合只能通过显式环境旗标运行。

## 已知限制

- 主检索仍以文本 embedding 为主；图像和图表依赖 MinerU high-effort 分析生成的文字／Markdown element。
- 论文库写锁只保证单进程内互斥。
- 引用检查包含 ID、词面支持与冲突启发式，不等同于完整语义蕴含证明。
- OpenAI-compatible 第三方后端必须支持标准 tool calling；供应商私有 reasoning 字段不会被依赖。
