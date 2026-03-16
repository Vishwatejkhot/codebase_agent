from pathlib import Path
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.ingest import load_code_documents



BASE_DIR    = Path(__file__).resolve().parent.parent   
INDEX_DIR   = BASE_DIR / "data" / "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(repo_path: str | Path, force: bool = False) -> FAISS:
    """
    Build and save a FAISS index from the cloned repo.

    Args:
        repo_path : Path to the cloned repo (inside gitcode/)
        force     : Rebuild even if index already exists

    Returns:
        FAISS vectorstore
    """
    repo_path = Path(repo_path)
    repo_name = repo_path.name
    save_path = INDEX_DIR / repo_name

    if save_path.exists() and not force:
        print(f"✅ Index already exists for '{repo_name}'. Loading...")
        return load_index(repo_name)

    print(f"\n🔨 Building FAISS index for: {repo_name}")
    docs = load_code_documents(repo_path)

    if not docs:
        raise ValueError(f"No code documents found in {repo_path}")

    print(f"🔗 Embedding {len(docs)} chunks (this may take a minute)...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    save_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_path))
    print(f"💾 Index saved to: {save_path}")

    return vectorstore


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_index(repo_name: str) -> FAISS:
    """Load an existing FAISS index by repo name."""
    save_path = INDEX_DIR / repo_name

    if not save_path.exists():
        raise FileNotFoundError(
            f"No index found for '{repo_name}'. "
            f"Run build_index() first."
        )

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(save_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"✅ Loaded index for: {repo_name}")
    return vectorstore


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------
def list_indexed_repos() -> list[str]:
    """Return names of all repos that have been indexed."""
    if not INDEX_DIR.exists():
        return []
    return [d.name for d in INDEX_DIR.iterdir() if d.is_dir()]


# ---------------------------------------------------------------------------
# Search helper
# ---------------------------------------------------------------------------
def search_codebase(vectorstore: FAISS, query: str, k: int = 5) -> list:
    """Run a similarity search and return top-k chunks with metadata."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vectorstore.py <path_to_cloned_repo>")
        sys.exit(1)

    vs = build_index(sys.argv[1])
    results = search_codebase(vs, "authentication function")
    for doc, score in results:
        print(f"\n📄 {doc.metadata['file_path']} (line {doc.metadata['start_line']}) | score: {score:.3f}")
        print(doc.page_content[:200])
