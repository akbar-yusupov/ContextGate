from __future__ import annotations

import argparse
import json
from pathlib import Path

RELEASE_CRITICAL = {
    "src/contextgate/adapters/langgraph/runtime.py": 90.0,
    "src/contextgate/adapters/litellm/generation.py": 90.0,
    "src/contextgate/adapters/litellm/providers.py": 90.0,
    "src/contextgate/adapters/local/guardrails.py": 90.0,
    "src/contextgate/adapters/local/ingestion_service.py": 90.0,
    "src/contextgate/adapters/local/loaders.py": 90.0,
    "src/contextgate/adapters/sqlalchemy/ledger.py": 90.0,
    "src/contextgate/application/use_cases.py": 90.0,
    "src/contextgate/apps/api/dependencies.py": 90.0,
    "src/contextgate/domain/evidence.py": 90.0,
    "src/contextgate/domain/gateway.py": 90.0,
    "src/contextgate/domain/risk.py": 90.0,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check release-critical module coverage")
    parser.add_argument("coverage_json", nargs="?", default="coverage.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.coverage_json).read_text(encoding="utf-8"))
    files = {name.replace("\\", "/"): value for name, value in payload["files"].items()}
    failures = []
    for filename, minimum in RELEASE_CRITICAL.items():
        actual = float(files.get(filename, {}).get("summary", {}).get("percent_covered", 0))
        print(f"{filename}: {actual:.2f}% (required {minimum:.0f}%)")
        if actual < minimum:
            failures.append(f"{filename}: {actual:.2f}%")
    if failures:
        raise SystemExit("Release-critical coverage failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
