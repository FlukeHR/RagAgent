from __future__ import annotations

import requests
import streamlit as st

# 绕过系统代理(Clash/V2Ray 等)直连本地后端，否则发往 localhost 的请求会被代理拦截返回 502
session = requests.Session()
session.trust_env = False

st.set_page_config(page_title="Paper RAG Agent", page_icon="📚", layout="wide")
st.title("📚 Paper RAG Agent")
st.caption("基于 Claude agentic RAG 的论文问答：本地论文库 + arXiv 在线检索 + 引用溯源")

api_base = st.sidebar.text_input("API 地址", value="http://localhost:8000")


@st.cache_data(ttl=10)
def fetch_collections(base: str) -> list[str]:
    try:
        resp = session.get(f"{base}/collections", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("collections", [])
    except requests.RequestException:
        return []
    return []


collections = fetch_collections(api_base)
collection = st.sidebar.selectbox(
    "论文集合", options=collections or ["demo"], index=0
)

st.sidebar.divider()
st.sidebar.subheader("从 arXiv 入库")
arxiv_query = st.sidebar.text_input("arXiv 关键词", placeholder="例如：retrieval augmented generation")
ingest_collection = st.sidebar.text_input("入库集合名", value="arxiv")
if st.sidebar.button("检索并下载入库"):
    if arxiv_query.strip():
        with st.spinner("正在从 arXiv 下载并建索引..."):
            r = session.post(
                f"{api_base}/ingest_arxiv",
                json={"query": arxiv_query, "collection": ingest_collection},
                timeout=300,
            )
        if r.status_code == 200:
            data = r.json()
            st.sidebar.success(
                f"已入库 {len(data['downloaded'])} 篇 -> 集合 {data['collection']}"
                f"（{data['indexed_chunks']} chunks）"
            )
            fetch_collections.clear()
        else:
            st.sidebar.error(f"入库失败: {r.status_code} {r.text}")
    else:
        st.sidebar.warning("请输入 arXiv 关键词")

question = st.text_area(
    "请输入你的问题", height=120, placeholder="例如：RAG 相比纯参数化模型的主要优势是什么？"
)

if st.button("提问", type="primary"):
    if not question.strip():
        st.warning("请输入问题")
    else:
        with st.spinner("Agent 正在检索并生成答案..."):
            response = session.post(
                f"{api_base}/ask",
                json={"question": question, "collection": collection},
                timeout=300,
            )
        if response.status_code != 200:
            st.error(f"请求失败: {response.status_code} {response.text}")
        else:
            data = response.json()
            st.subheader("回答")
            st.write(data.get("answer", ""))

            with st.expander("🔎 Agent 推理步骤", expanded=False):
                for step in data.get("steps", []):
                    st.write(f"- {step}")

            st.subheader("📎 引用来源")
            for src in data.get("sources", []):
                score = src.get("score")
                score_txt = f" · score={score}" if score is not None else ""
                title = f"**《{src['paper_title']}》** · {src['section']} "
                meta = f"`{src['paper_id']}`{score_txt}"
                src_link = src.get("source", "")
                if src_link.startswith("http"):
                    st.markdown(f"- {title}{meta} — [{src_link}]({src_link})")
                else:
                    st.markdown(f"- {title}{meta} — `{src_link}`")
