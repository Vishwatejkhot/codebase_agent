import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


EXTENSION_MAP = {
    ".py":    "python",
    ".js":    "javascript",
    ".ts":    "typescript",
    ".jsx":   "javascript",
    ".tsx":   "typescript",
    ".java":  "java",
    ".cpp":   "cpp",
    ".c":     "c",
    ".cs":    "csharp",
    ".go":    "go",
    ".rb":    "ruby",
    ".rs":    "rust",
    ".php":   "php",
    ".swift": "swift",
    ".kt":    "kotlin",
    ".md":    "markdown",
    ".yaml":  "yaml",
    ".yml":   "yaml",
    ".json":  "json",
    ".sh":    "bash",
    ".html":  "html",
    ".css":   "css",
    ".sql":   "sql",
}

# Directories to always skip
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", ".env", "dist", "build", ".idea", ".vscode",
    "*.egg-info", "coverage", ".pytest_cache",
}

# Files to always skip
SKIP_FILES = {
    "package-lock.json", "yarn.lock", "poetry.lock",
    "Pipfile.lock", ".DS_Store",
}

MAX_FILE_SIZE_KB = 200   

def get_splitter(language: str) -> RecursiveCharacterTextSplitter:
    """
    Returns a code-aware text splitter.
    Tries to split on class/function boundaries first.
    """
    separators = {
        "python":     ["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
        "javascript": ["\nfunction ", "\nclass ", "\nconst ", "\n\n", "\n", " ", ""],
        "typescript": ["\nfunction ", "\nclass ", "\nconst ", "\n\n", "\n", " ", ""],
        "java":       ["\nclass ", "\npublic ", "\nprivate ", "\n\n", "\n", " ", ""],
        "go":         ["\nfunc ", "\ntype ", "\n\n", "\n", " ", ""],
    }
    chosen = separators.get(language, ["\n\n", "\n", " ", ""])

    return RecursiveCharacterTextSplitter(
        separators=chosen,
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )


def should_skip_dir(dir_name: str) -> bool:
    return dir_name in SKIP_DIRS or dir_name.startswith(".")


def load_code_documents(repo_path: str | Path) -> list[Document]:
    """
    Walk the repo, read every supported code file, split into chunks.

    Returns:
        List of LangChain Document objects with metadata:
        {
            file_path   : relative path inside repo
            language    : detected language
            start_line  : approximate start line of chunk
            repo_name   : name of the repository
        }
    """
    repo_path = Path(repo_path)
    repo_name = repo_path.name
    all_docs  = []
    skipped   = 0
    processed = 0

    print(f"\n📂 Scanning repo: {repo_path}")

    for root, dirs, files in os.walk(repo_path):
        # Prune unwanted directories in-place
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]

        for filename in files:
            if filename in SKIP_FILES:
                continue

            ext = Path(filename).suffix.lower()
            if ext not in EXTENSION_MAP:
                skipped += 1
                continue

            file_path = Path(root) / filename

            # Skip large files
            size_kb = file_path.stat().st_size / 1024
            if size_kb > MAX_FILE_SIZE_KB:
                print(f"   ⚠️  Skipping large file ({size_kb:.0f} KB): {filename}")
                skipped += 1
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"   ❌ Could not read {filename}: {e}")
                skipped += 1
                continue

            if not content.strip():
                skipped += 1
                continue

            language     = EXTENSION_MAP[ext]
            relative_path = str(file_path.relative_to(repo_path))
            splitter     = get_splitter(language)
            chunks       = splitter.split_text(content)

            for i, chunk in enumerate(chunks):
                # Approximate line numbers
                chars_before = sum(len(c) for c in chunks[:i])
                start_line   = content[:chars_before].count("\n") + 1

                doc = Document(
                    page_content=chunk,
                    metadata={
                        "file_path":  relative_path,
                        "language":   language,
                        "start_line": start_line,
                        "repo_name":  repo_name,
                        "filename":   filename,
                    },
                )
                all_docs.append(doc)

            processed += 1

    print(f"✅ Processed {processed} files → {len(all_docs)} chunks")
    print(f"   Skipped {skipped} files (unsupported / too large / empty)")
    return all_docs


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "gitcode"
    docs = load_code_documents(path)
    print(f"\nSample chunk:\n{docs[0].page_content[:300]}")
    print(f"Metadata: {docs[0].metadata}")
