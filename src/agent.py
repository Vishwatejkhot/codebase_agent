import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.faiss import FAISS 

from src.tools import (
    search_code,
    explain_code,
    suggest_refactor,
    find_bugs,
    generate_docs,
    trace_flow,
    set_vectorstore,
)
from src.vectorstore import build_index, load_index, list_indexed_repos
from src.cloner import clone_repo, get_repo_name
from src.tools import set_vectorstore 

load_dotenv()

BASE_DIR    = Path(__file__).resolve().parent
GITCODE_DIR = BASE_DIR / "gitcode"


SYSTEM_PROMPT = """You are an expert software engineer and code analyst with deep knowledge of software architecture, design patterns, and best practices.

You have full access to an indexed codebase and a set of specialized tools. When a user asks a question:

1. **Always start** with `search_code` to find relevant code chunks
2. **Then use** the most appropriate tool based on what they want:
   - Understanding code → `explain_code`
   - Improving code    → `suggest_refactor`
   - Finding issues    → `find_bugs`
   - Documentation     → `generate_docs`
   - End-to-end flows  → `trace_flow`

3. **Always cite** file names and line numbers in your final answer
4. **Be specific** — reference actual code, not vague descriptions
5. If a question needs multiple tools, use them in sequence

You are analyzing the repository: {repo_name}
"""


def build_agent(repo_name: str) -> AgentExecutor:

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    tools = [
        search_code,
        explain_code,
        suggest_refactor,
        find_bugs,
        generate_docs,
        trace_flow,
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(repo_name=repo_name)),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent    = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
    )
    return executor


def setup_from_url(github_url: str, force_reclone: bool = False, force_reindex: bool = False) -> tuple[AgentExecutor, str]:
    """
    Full pipeline:
      1. Clone the GitHub repo into gitcode/
      2. Build or load FAISS index
      3. Register vectorstore with tools
      4. Return ready AgentExecutor

    Args:
        github_url     : Full GitHub URL
        force_reclone  : Re-clone even if already cloned
        force_reindex  : Rebuild index even if already indexed

    Returns:
        (AgentExecutor, repo_name)
    """
    # Step 1 — Clone
    repo_path = clone_repo(github_url, force=force_reclone)
    repo_name = repo_path.name

    # Step 2 — Index
    vectorstore = build_index(repo_path, force=force_reindex)

    # Step 3 — Register with tools
    set_vectorstore(vectorstore)

    # Step 4 — Build agent
    agent = build_agent(repo_name)

    print(f"\n🤖 Agent ready for repo: {repo_name}")
    return agent, repo_name


def load_existing(repo_name: str) -> tuple[AgentExecutor, str]:

    vectorstore = load_index(repo_name)
    set_vectorstore(vectorstore)
    agent = build_agent(repo_name)
    return agent, repo_name



def run_cli():
    """Interactive CLI chat loop."""
    print("\n" + "="*60)
    print("       🤖 Codebase Q&A Agent — CLI Mode")
    print("="*60)

    # Check for existing repos
    existing = list_indexed_repos()
    if existing:
        print("\n📚 Previously indexed repos:")
        for i, r in enumerate(existing, 1):
            print(f"   {i}. {r}")
        print("\nOptions:")
        print("  [1-N] Load an existing repo")
        print("  [new] Enter a new GitHub URL")
        choice = input("\nYour choice: ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(existing):
            repo_name = existing[int(choice) - 1]
            agent, repo_name = load_existing(repo_name)
        else:
            github_url  = input("\n🔗 Enter GitHub URL: ").strip()
            agent, repo_name = setup_from_url(github_url)
    else:
        github_url  = input("\n🔗 Enter GitHub URL: ").strip()
        agent, repo_name = setup_from_url(github_url)

    print(f"\n✅ Ready! Analyzing: {repo_name}")
    print("💬 Ask anything about the codebase. Type 'exit' to quit.\n")

    # Chat loop
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if question.lower() in {"exit", "quit", "q"}:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        print("\n🤖 Agent thinking...\n")
        try:
            result = agent.invoke({"input": question})
            print(f"\n{'─'*60}")
            print("Agent:", result["output"])
            print(f"{'─'*60}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    run_cli()
