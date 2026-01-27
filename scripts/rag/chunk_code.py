from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_code import load_codebase

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

docs = load_codebase("data/fastapi")
chunks = splitter.split_documents(docs)

print(f"Original files: {len(docs)}")
print(f"Code chunks: {len(chunks)}")
