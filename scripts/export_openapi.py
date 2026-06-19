from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["CONTEXTGATE_ENV_FILE"] = ""
os.environ["CONTEXTGATE_RATE_LIMIT_ENABLED"] = "false"

from contextgate.apps.api.main import app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "openapi.json"


def rendered_schema() -> str:
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export or verify the ContextGate OpenAPI schema")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = rendered_schema()
    if args.check:
        if not TARGET.is_file() or TARGET.read_text(encoding="utf-8") != rendered:
            raise SystemExit("docs/openapi.json is stale; run scripts/export_openapi.py")
        print("docs/openapi.json is current")
        return
    TARGET.write_text(rendered, encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
