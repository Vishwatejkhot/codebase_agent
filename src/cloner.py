import os
import shutil
import git
from pathlib import Path


BASE_DIR    = Path(__file__).resolve().parent          
GITCODE_DIR = BASE_DIR / "gitcode"                     


def get_repo_name(github_url: str) -> str:

    name = github_url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name


def clone_repo(github_url: str, force: bool = False) -> Path:
    """
    Clone the GitHub repo into codebase_agent/gitcode/<repo_name>/
    
    Args:
        github_url : Full GitHub URL  e.g. https://github.com/user/repo
        force      : If True, delete existing clone and re-clone
    
    Returns:
        Path to the cloned repo directory
    """
    GITCODE_DIR.mkdir(parents=True, exist_ok=True)

    repo_name   = get_repo_name(github_url)
    target_path = GITCODE_DIR / repo_name

    if target_path.exists():
        if force:
            print(f"🗑️  Removing existing clone at {target_path} ...")
            shutil.rmtree(target_path)
        else:
            print(f"✅ Repo already cloned at: {target_path}")
            print("   Pass force=True to re-clone.")
            return target_path

    print(f"📥 Cloning {github_url}")
    print(f"   → into {target_path}")
    git.Repo.clone_from(github_url, str(target_path))
    print(f"✅ Clone complete: {target_path}")
    return target_path


def list_cloned_repos() -> list[str]:
    """List all repos currently cloned inside gitcode/."""
    if not GITCODE_DIR.exists():
        return []
    return [d.name for d in GITCODE_DIR.iterdir() if d.is_dir()]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python cloner.py <github_url> [--force]")
        sys.exit(1)

    url   = sys.argv[1]
    force = "--force" in sys.argv
    clone_repo(url, force=force)
