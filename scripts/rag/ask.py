from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import CrossEncoder
import torch
# ======================
# 1. 加载 embedding 和 FAISS 向量库（只需一次）
# ======================
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

db = FAISS.load_local(
    "faiss_index", 
    embedding, 
    allow_dangerous_deserialization=True  
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")


# ======================
# 2. 加载 LLM 和 tokenizer（只需一次！）
# ======================
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     dtype="auto",  # 显式指定 dtype
#     device_map="auto"           # 自动分配（GPU + CPU offload）
# )
# print("✅ 模型和向量库加载完成！")

# # ======================
# # 3. 循环提问：每次重新 RAG！
# # ======================
# while True:
#     query = input("\n🔍 Ask about FastAPI (type 'quit' to exit): ").strip()
    
#     if not query or query.lower() == "quit":
#         print("👋 Goodbye!")
#         break

#     candidate_docs = db.similarity_search(query, k=20)  # 初筛 20 个
#     pairs = [(query, doc.page_content) for doc in candidate_docs]
#     scores = reranker.predict(pairs)

#     scored = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
#     docs = [doc for doc, _ in scored[:4]]  # 最终 4 个
    
#     # 构建上下文
#     context = "\n\n".join(
#         f"File: {d.metadata['path']}\n{d.page_content[:800]}"
#         for d in docs
#     )

    
#     prompt = f"""
# You are a senior software engineer specializing in FastAPI.

# Use ONLY the following code snippets to answer the question. 
# If the snippets don't contain relevant information, say "I don't know based on the provided code."

# Code snippets:
# {context}

# Question: {query}
# Answer:
# """

#     # Tokenize 并推理
#     inputs = tokenizer(
#         prompt, 
#         return_tensors="pt",
#         truncation=True,
#         max_length=32768  # 防止超长
#     ).to(model.device)

#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=512,
#         temperature=0.2,
#         # do_sample=False,      # 确定性输出（适合技术问答）
#         # pad_token_id=tokenizer.eos_token_id
#     )

#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     final_answer = answer.split("Answer:")[-1].strip()

    
#     print("\n💬 Answer:")
#     print(final_answer)



MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

db = FAISS.load_local(
    "faiss_index", 
    embedding, 
    allow_dangerous_deserialization=True  
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # quantization_config=quantization_config,
    dtype=torch.float16,
    device_map="cuda"
)
print("✅ 模型加载完成！")

query = "How to install FastAPI?"

candidate_docs = db.similarity_search(query, k=20)  # 初筛 20 个
pairs = [(query, doc.page_content) for doc in candidate_docs]
scores = reranker.predict(pairs)

scored = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
docs = [doc for doc, _ in scored[:4]]  # 最终 4 个

# 构建上下文
context = "\n\n".join(
    f"File: {d.metadata['path']}\n{d.page_content[:800]}"
    for d in docs
)
docs = db.similarity_search(query, k=4)
context = "\n\n".join(
    f"File: {d.metadata['path']}\n{d.page_content[:800]}"
    for d in docs
)

prompt = f"""
You are a senior software engineer.

Use the following code snippets to answer the question.

{context}

Question: {query}
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.2
)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)