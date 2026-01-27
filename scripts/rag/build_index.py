from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from chunk_code import chunks

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

db = FAISS.from_documents(chunks, embedding)
db.save_local("faiss_index")

print("FAISS index built and saved.")
