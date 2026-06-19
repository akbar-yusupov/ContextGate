from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def request_once(
    url: str,
    api_key: str,
    knowledge_base: str,
    policy: str,
) -> tuple[float, int]:
    body = json.dumps(
        {
            "knowledge_base": knowledge_base,
            "query": "How can I cancel an order?",
            "policy": policy,
        }
    ).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/v1/runs/answer",
        data=body,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
            status = response.status
    except Exception:
        status = 0
    return (time.perf_counter() - started) * 1000, status


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded ContextGate answer-path load smoke")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="contextgate-dev-key")
    parser.add_argument("--knowledge-base", default="demo")
    parser.add_argument("--policy", default="fast")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--max-p95-ms", type=float, default=2000)
    parser.add_argument("--max-p99-ms", type=float, default=4000)
    parser.add_argument("--target-rps", type=float, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    results: list[tuple[float, int]] = []
    submission_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for index in range(args.requests):
            if args.target_rps > 0:
                target = submission_started + index / args.target_rps
                time.sleep(max(0.0, target - time.perf_counter()))
            futures.append(
                executor.submit(
                    request_once,
                    args.url,
                    args.api_key,
                    args.knowledge_base,
                    args.policy,
                )
            )
        for future in as_completed(futures):
            results.append(future.result())
    latencies = [latency for latency, _ in results]
    failures = sum(status != 200 for _, status in results)
    summary = {
        "requests": len(results),
        "failures": failures,
        "mean_ms": statistics.mean(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "target_rps": args.target_rps or None,
        "policy": args.policy,
        "elapsed_seconds": time.perf_counter() - submission_started,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    if failures or summary["p95_ms"] > args.max_p95_ms or summary["p99_ms"] > args.max_p99_ms:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
