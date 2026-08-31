# FigureLens

FigureLens 是一个受 [Look As You Think](https://arxiv.org/abs/2511.12003) 启发的最小视觉文档问答 Agent。它直接保留 PDF 页面图像，通过文本召回候选页，再让视觉模型回答并返回可点击验证的页面 bbox 证据。

```text
PDF → 页面文本 + 页面 PNG → 有界页面召回
    → 视觉模型回答 / 请求再搜索一次
    → 服务端校验 page + bbox + [E#]
    → 页面高亮证据
```

## 当前能力

- 本地账号、CSRF 和加密模型配置；
- 上传文本型 PDF，限制为20MB、30页；
- PyMuPDF 提取页面文本并渲染页面图像；
- 英文词与中文 bigram 的小规模线性页面检索；
- 最多两轮视觉检查，支持一次 `search_more`；
- 归一化 bbox、证据 claim 和 `[E#]` 引用校验；
- 单文件原生 Web UI，点击证据即可高亮页面区域。

当前不支持 OCR、多文档问答、对话历史、向量库、模型训练或 LAT 的 SFT/GRPO 复现。

## 运行

```powershell
pip install -r requirements.txt
uvicorn api.main:app --reload
```

打开 `http://127.0.0.1:8000/ui/`：

1. 注册本地账号；
2. 添加一个支持图片输入的 OpenAI-compatible 模型；
3. 上传 PDF；
4. 选择文档并询问图表问题。

模型必须能接受 Chat Completions 的 `image_url` data URL，并按提示返回 JSON。API key 使用 AES-256-GCM 加密保存在本地数据库，主密钥默认写入被 Git 忽略的 `data/secrets/master.key`。

## 验证

```powershell
python -m unittest discover -s tests
ruff check .
mypy .
git diff --check
git ls-files data
```

测试不会调用真实模型。`git ls-files data` 的正常结果只能包含 `data/__init__.py`。
