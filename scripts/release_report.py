from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from contextgate import __version__

ROOT = Path(__file__).resolve().parents[1]


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, check=True, text=True
    ).stdout.strip()


def optional_json(path: str | None) -> object | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def junit_summary(path: str | None) -> dict[str, int] | None:
    if not path:
        return None
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    total = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    return {
        "total": total,
        "passed": total - failures - errors - skipped,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def coverage_percent(path: str | None) -> float | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(payload["totals"]["percent_covered"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a machine-readable release evidence report"
    )
    parser.add_argument("--output", default="reports/release-report.json")
    parser.add_argument("--image-digest", default="unrecorded")
    parser.add_argument("--junit-xml")
    parser.add_argument("--coverage-json")
    parser.add_argument("--evaluation-json")
    parser.add_argument("--load-json")
    parser.add_argument("--dependency-audit", default="not_recorded")
    parser.add_argument("--container-scan", default="not_recorded")
    parser.add_argument("--sbom", default="not_recorded")
    args = parser.parse_args()
    tests = junit_summary(args.junit_xml)
    payload = {
        "schema_version": "v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "version": __version__,
        "commit": git_value("rev-parse", "HEAD"),
        "dirty": bool(git_value("status", "--porcelain")),
        "image_digest": args.image_digest,
        "alembic_head": "0008_cost_request_id_length",
        "checks": {
            "tests": tests,
            "tests_passed": tests["passed"] if tests else None,
            "coverage_percent": coverage_percent(args.coverage_json),
            "dependency_audit": args.dependency_audit,
            "container_scan": args.container_scan,
            "sbom": args.sbom,
        },
        "evaluation": optional_json(args.evaluation_json),
        "load": optional_json(args.load_json),
        "known_limitations": [
            "single-tenant",
            "local verifier is not formal proof",
            "rule-based injection detection is not a complete firewall",
            "OCR and scanned PDFs are unsupported",
        ],
        "release_status": "candidate",
    }
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
