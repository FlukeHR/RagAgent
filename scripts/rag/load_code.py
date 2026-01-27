from pathlib import Path
from langchain_core.documents import Document

def load_codebase(repo_path: str, exts=(".py",)):
    docs = []
    for path in Path(repo_path).rglob("*"):
        if path.suffix in exts:
            try:
                text = path.read_text(encoding="utf-8")
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"path": str(path)}
                    )
                )
            except Exception:
                pass
    return docs


if __name__ == "__main__":
    docs = load_codebase("data/fastapi")
    print(f"Loaded {len(docs)} code files")

