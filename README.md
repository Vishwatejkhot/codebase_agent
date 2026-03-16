# 🤖 Codebase Q&A Agent

An agentic RAG system that lets you **chat with any GitHub repository** using Groq model + LangChain + FAISS.

---

## 📁 Folder Structure

```
codebase_agent/
├── gitcode/                  ← Cloned GitHub repos live here (auto-created)
│   └── <repo-name>/          ← e.g. gitcode/langchain/
├── data/
│   └── faiss_index/          ← FAISS vector indexes (auto-created)
│       └── <repo-name>/
├── src/
│   ├── __init__.py
│   ├── cloner.py             ← Clone GitHub repo into gitcode/
│   ├── ingest.py             ← Walk repo, chunk code files
│   ├── vectorstore.py        ← Build & load FAISS index
│   ├── tools.py              ← LangChain agent tools
│   └── agent.py             ← Agent orchestrator + CLI
├── app.py                    ← Streamlit UI
├── main.py                   ← CLI entry point
├── requirements.txt
├── .env
└── README.md
```

---

## ⚡ Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key
```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at: https://console.groq.com

### 3a. Run CLI
```bash
# From GitHub URL (clones → indexes → chat)
python main.py --url https://github.com/user/repo

# Load previously indexed repo
python main.py --repo my-repo

# Interactive mode (prompts you)
python main.py
```

### 3b. Run Streamlit UI
```bash
streamlit run app.py
```

---

## 🤖 What the Agent Can Do

| Tool | What it does |
|---|---|
| `search_code` | Find relevant code chunks via RAG |
| `explain_code` | Explain functions, classes, modules |
| `suggest_refactor` | Improve code quality with before/after |
| `find_bugs` | Spot bugs, security holes, edge cases |
| `generate_docs` | Write docstrings and documentation |
| `trace_flow` | Trace end-to-end feature flows across files |

---

## 💬 Example Questions

```
"Give me an overview of this codebase"
"How does authentication work?"
"Explain what process_payment() does"
"Find security vulnerabilities in the API layer"
"Refactor the database module"
"Generate a docstring for the User class"
"Trace the request flow from HTTP → database"
```

---

## 🔧 Supported File Types

Python, JavaScript, TypeScript, Java, Go, Rust, C/C++, C#, Ruby, PHP, Swift, Kotlin, SQL, Bash, HTML, CSS, Markdown, YAML, JSON

---

## 📐 Architecture

```
GitHub URL
    ↓
cloner.py → gitcode/<repo>/
    ↓
ingest.py → code chunks with metadata
    ↓
vectorstore.py → FAISS index (data/faiss_index/<repo>/)
    ↓
User Question → LangChain Agent (Groq llama3-70b)
    ↓
Tool Selection → search_code → explain / refactor / bugs / docs / trace
    ↓
Answer with file references + line numbers
```
