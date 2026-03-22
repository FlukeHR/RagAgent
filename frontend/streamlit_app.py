from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="Code RAG Agent", page_icon="🤖", layout="wide")
st.title("Code RAG Agent")
st.caption("面向代码仓库的智能问答")

api_base = st.sidebar.text_input("API 地址", value="http://localhost:8000")
repo_name = st.sidebar.text_input("仓库名", value="fastapi")
question = st.text_area("请输入你的问题", height=140, placeholder="例如：FastAPI 中间件是如何工作的？")

if st.button("提问", type="primary"):
    if not question.strip():
        st.warning("请输入问题")
    else:
        with st.spinner("正在检索并生成答案..."):
            response = requests.post(
                f"{api_base}/ask",
                json={"question": question, "repo_name": repo_name},
                timeout=60,
            )

        if response.status_code != 200:
            st.error(f"请求失败: {response.status_code} {response.text}")
        else:
            data = response.json()
            st.subheader("回答")
            st.write(data.get("answer", ""))

            st.subheader("推理步骤")
            for step in data.get("steps", []):
                st.write(f"- {step}")

            st.subheader("来源片段")
            for src in data.get("sources", []):
                st.write(
                    f"- {src['file_path']}:{src['start_line']}-{src['end_line']} (score={src['score']})"
                )
