from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^]]+]\(([^)]+)\)")
AGENT_ADAPTERS = (
    ROOT / "CLAUDE.md",
    ROOT / "GEMINI.md",
    ROOT / ".github" / "copilot-instructions.md",
    ROOT / ".cursor" / "rules" / "contextgate.mdc",
)


def tracked_markdown() -> list[Path]:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "ls-files", "*.md"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    files = [ROOT / line for line in result.stdout.splitlines() if line]
    for path in ROOT.rglob("*.md"):
        if path.is_file() and not any(
            part.startswith(".") for part in path.relative_to(ROOT).parts
        ):
            files.append(path)
    return sorted(set(files))


def main() -> None:
    failures: list[str] = []
    for document in tracked_markdown():
        text = document.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            target = target.strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                failures.append(f"{document.relative_to(ROOT)} -> {target}")
    for adapter in AGENT_ADAPTERS:
        if not adapter.is_file() or "AGENTS.md" not in adapter.read_text(encoding="utf-8"):
            failures.append(f"{adapter.relative_to(ROOT)} must reference AGENTS.md")
    if failures:
        raise SystemExit("Documentation validation failed:\n" + "\n".join(failures))
    print(f"Validated {len(tracked_markdown())} Markdown files and agent adapters.")


if __name__ == "__main__":
    main()
