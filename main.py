"""
main.py
-------
Entry point for the Codebase Q&A Agent.

Usage:
  # Full pipeline from GitHub URL
  python main.py --url https://github.com/user/repo

  # Load a previously indexed repo
  python main.py --repo my-repo

  # Launch Streamlit UI
  streamlit run app.py
"""

import argparse
from src.agent import setup_from_url, load_existing, run_cli
from src.vectorstore import list_indexed_repos


def main():
    parser = argparse.ArgumentParser(description="Codebase Q&A Agent")
    parser.add_argument("--url",   type=str, help="GitHub URL to clone and index")
    parser.add_argument("--repo",  type=str, help="Name of already-indexed repo to load")
    parser.add_argument("--force-clone",  action="store_true", help="Force re-clone")
    parser.add_argument("--force-index",  action="store_true", help="Force re-index")
    parser.add_argument("--list",  action="store_true", help="List indexed repos")
    args = parser.parse_args()

    if args.list:
        repos = list_indexed_repos()
        if repos:
            print("\n📚 Indexed repos:")
            for r in repos:
                print(f"  • {r}")
        else:
            print("No repos indexed yet.")
        return

    if args.url:
        agent, repo_name = setup_from_url(
            args.url,
            force_reclone=args.force_clone,
            force_reindex=args.force_index,
        )
    elif args.repo:
        agent, repo_name = load_existing(args.repo)
    else:
        # Interactive mode
        run_cli()
        return

    # One-shot Q&A loop after loading
    print(f"\n✅ Ready! Analyzing: {repo_name}")
    print("💬 Type your questions. Type 'exit' to quit.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue

        result = agent.invoke({"input": q})
        print(f"\nAgent: {result['output']}\n")


if __name__ == "__main__":
    main()
