from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_community.vectorstores.faiss import FAISS


def get_llm() -> ChatGroq:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Global vectorstore holder — set by agent.py before running
# ---------------------------------------------------------------------------
_vectorstore: FAISS | None = None


def set_vectorstore(vs: FAISS):
    global _vectorstore
    _vectorstore = vs


def _retrieve(query: str, k: int = 5) -> tuple[str, list[dict]]:
    """
    Internal helper: runs similarity search and formats chunks.
    Returns (formatted_context_string, list_of_metadata_dicts)
    """
    if _vectorstore is None:
        return "No codebase indexed yet.", []

    results = _vectorstore.similarity_search_with_score(query, k=k)

    context_parts = []
    sources       = []

    for doc, score in results:
        meta = doc.metadata
        header = (
            f"📄 File: {meta.get('file_path', 'unknown')} "
            f"| Language: {meta.get('language', '?')} "
            f"| Line: ~{meta.get('start_line', '?')} "
            f"| Score: {score:.3f}"
        )
        context_parts.append(f"{header}\n```{meta.get('language','')}\n{doc.page_content}\n```")
        sources.append(meta)

    return "\n\n---\n\n".join(context_parts), sources



@tool
def search_code(query: str) -> str:
    """
    Search the indexed codebase for code related to a query.
    Returns the most relevant code chunks with file paths and line numbers.
    Use this first before any other tool to find relevant code.
    """
    context, sources = _retrieve(query, k=5)

    if not sources:
        return "No relevant code found for the query."

    source_list = "\n".join(
        f"  • {s.get('file_path')} (line ~{s.get('start_line')})"
        for s in sources
    )

    return f"Found in:\n{source_list}\n\n{context}"



@tool
def explain_code(query: str) -> str:
    """
    Finds code related to the query and explains what it does in plain English.
    Good for: understanding functions, classes, modules, or logic flows.
    """
    context, sources = _retrieve(query, k=4)
    llm = get_llm()

    if not sources:
        return "Could not find relevant code to explain."

    prompt = f"""You are a senior software engineer explaining code to a teammate.

Based on this code retrieved from the codebase:

{context}

Question: {query}

Provide a clear explanation covering:
1. **What it does** — purpose and responsibility
2. **How it works** — step-by-step logic
3. **Key inputs/outputs** — parameters, return values
4. **Where it fits** — how it connects to other parts (based on file paths)

Be concise but thorough. Reference specific file names."""

    return llm.invoke(prompt).content



@tool
def suggest_refactor(query: str) -> str:
    """
    Finds code related to the query and suggests concrete refactoring improvements.
    Good for: improving code quality, reducing complexity, fixing design issues.
    """
    context, sources = _retrieve(query, k=4)
    llm = get_llm()

    if not sources:
        return "Could not find relevant code to refactor."

    prompt = f"""You are a senior software engineer performing a code review.

Code retrieved from the codebase:

{context}

Refactoring request: {query}

Provide:
1. **Issues Found** — specific problems with the current code
2. **Refactored Version** — show the improved code with comments
3. **Why It's Better** — explain each improvement
4. **Risk Level** — low / medium / high (how risky is this change?)

Use markdown formatting. Show before/after code blocks."""

    return llm.invoke(prompt).content


@tool
def find_bugs(query: str) -> str:
    """
    Scans retrieved code for potential bugs, security issues, or edge cases.
    Good for: error handling, null checks, security vulnerabilities, race conditions.
    """
    context, sources = _retrieve(query, k=5)
    llm = get_llm()

    if not sources:
        return "Could not find relevant code to analyze."

    prompt = f"""You are a security-focused senior engineer doing a bug hunt.

Code retrieved from the codebase:

{context}

Bug hunt request: {query}

Identify and report:
1. **🔴 Critical Bugs** — crashes, data loss, security holes
2. **🟡 Warnings** — edge cases, missing validation, poor error handling
3. **🔵 Code Smells** — maintainability issues, anti-patterns
4. **✅ Fix Suggestions** — concrete code fixes for each issue found

If no bugs are found, say so clearly. Reference exact file and line numbers."""

    return llm.invoke(prompt).content



@tool
def generate_docs(query: str) -> str:
    """
    Finds a function/class/module and generates proper documentation for it.
    Good for: generating docstrings, README sections, API docs, inline comments.
    """
    context, sources = _retrieve(query, k=3)
    llm = get_llm()

    if not sources:
        return "Could not find relevant code to document."


    lang = sources[0].get("language", "python") if sources else "python"

    prompt = f"""You are a technical writer generating documentation for a codebase.

Code retrieved:

{context}

Documentation request: {query}

Generate:
1. **Docstring** — in the correct format for {lang} (Google style for Python, JSDoc for JS/TS)
2. **Function/Class Summary** — one-paragraph plain English description
3. **Parameters Table** — name | type | description for each param
4. **Returns** — what is returned and when
5. **Usage Example** — a realistic code snippet showing how to use it

Format cleanly with markdown."""

    return llm.invoke(prompt).content



@tool
def trace_flow(query: str) -> str:
    """
    Traces end-to-end logic across multiple files for a feature or workflow.
    Good for: understanding request/response flows, data pipelines, auth flows,
    event chains — anything that spans multiple files or modules.
    """
    context, sources = _retrieve(query, k=6)
    llm = get_llm()

    if not sources:
        return "Could not find relevant code to trace."

    file_list = list({s.get("file_path") for s in sources})

    prompt = f"""You are a senior engineer explaining a system's architecture and flow.

Code retrieved from these files:
{', '.join(file_list)}

Full context:
{context}

Trace request: {query}

Provide:
1. **Flow Diagram** (ASCII or numbered steps)
   Entry point → Step 1 → Step 2 → ... → Output
2. **File-by-File Breakdown** — what each file contributes to this flow
3. **Data Transformations** — how data changes at each step
4. **Integration Points** — external services, APIs, databases touched
5. **Potential Bottlenecks** — where performance or failures might occur

Be specific about which file handles which part of the flow."""

    return llm.invoke(prompt).content
