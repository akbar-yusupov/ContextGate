from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid


class Client:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key}

    def request(
        self, method: str, path: str, payload: dict | None = None
    ) -> tuple[int, bytes, dict[str, str]]:
        body = json.dumps(payload).encode() if payload is not None else None
        headers = dict(self.headers)
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path, data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc.code} {exc.read().decode()}") from exc

    def json(self, method: str, path: str, payload: dict | None = None) -> dict | list:
        status, body, _ = self.request(method, path, payload)
        if status >= 400:
            raise RuntimeError(f"{method} {path} returned {status}")
        return json.loads(body)

    def upload(self, path: str, filename: str, content: bytes) -> dict:
        boundary = f"contextgate-{uuid.uuid4().hex}"
        body = (
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                "Content-Type: application/jsonl\r\n\r\n"
            ).encode()
            + content
            + f"\r\n--{boundary}--\r\n".encode()
        )
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={**self.headers, "Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read())


def wait_job(client: Client, job_id: str, timeout: float = 300) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.json("GET", f"/api/v1/jobs/{job_id}")
        assert isinstance(job, dict)
        if job["status"] in {"succeeded", "succeeded_with_errors"}:
            return job
        if job["status"] in {"failed", "cancelled"}:
            raise RuntimeError(f"job {job_id} ended as {job['status']}: {job.get('error_json')}")
        time.sleep(1)
    raise TimeoutError(f"job {job_id} did not finish")


def main() -> None:
    parser = argparse.ArgumentParser(description="ContextGate clean-Compose acceptance smoke")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="contextgate-dev-key")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    client = Client(args.url, args.api_key)

    readiness = client.json("GET", "/ready")
    models = client.json("GET", "/v1/models")
    assert isinstance(readiness, dict) and readiness["status"] == "ready"
    assert isinstance(models, dict) and any(
        item["id"] == "kb:demo:balanced" for item in models["data"]
    )

    answered = client.json(
        "POST",
        "/api/v1/runs/answer",
        {"knowledge_base": "demo", "query": "How can I cancel an order?", "policy": "balanced"},
    )
    abstained = client.json(
        "POST",
        "/api/v1/runs/answer",
        {"knowledge_base": "demo", "query": "Do you accept cryptocurrency?", "policy": "balanced"},
    )
    assert isinstance(answered, dict) and answered["status"] == "answered"
    assert answered["citations"] and answered["evidence_report"]["passed"]
    assert isinstance(abstained, dict) and abstained["status"] == "abstained"
    assert abstained["answer"] == "" and abstained["abstention_reason"]

    run_id = answered["run_id"]
    trace = client.json("GET", f"/api/v1/runs/{run_id}/trace")
    cost = client.json("GET", f"/api/v1/runs/{run_id}/cost")
    status, events, _ = client.request(
        "GET", f"/api/v1/runs/{run_id}/events?after_sequence=-1&follow=false"
    )
    assert isinstance(trace, dict) and trace["run_id"] == run_id
    assert isinstance(cost, dict) and "estimated_usd" in cost
    assert status == 200 and b"event: final" in events

    evaluation_run_id = None
    if args.evaluate:
        dataset = (
            json.dumps(
                {
                    "id": "e2e-answerable",
                    "query": "How can I cancel an order?",
                    "language": "en",
                    "relevant_chunk_ids": ["cancel-order-en:0"],
                    "expected_facts": [
                        "An order can be cancelled before it is handed to the courier."
                    ],
                    "answerable": True,
                    "tags": ["e2e"],
                }
            )
            + "\n"
            + json.dumps(
                {
                    "id": "e2e-unanswerable",
                    "query": "Do you accept cryptocurrency?",
                    "language": "en",
                    "relevant_chunk_ids": [],
                    "expected_facts": [],
                    "answerable": False,
                    "tags": ["e2e", "unanswerable"],
                }
            )
            + "\n"
        ).encode()
        uploaded = client.upload("/api/v1/evaluations/datasets", "e2e.jsonl", dataset)
        job = client.json(
            "POST",
            "/api/v1/evaluations",
            {
                "knowledge_base": "demo",
                "dataset_path": uploaded["dataset_id"],
                "policies": ["balanced"],
                "evaluate_answers": True,
            },
        )
        assert isinstance(job, dict)
        completed = wait_job(client, job["id"])
        evaluation_run_id = completed["result"]["run_id"]
        results = client.json("GET", f"/api/v1/evaluations/{evaluation_run_id}")
        status, report, _ = client.request("GET", f"/api/v1/evaluations/{evaluation_run_id}/report")
        assert isinstance(results, dict) and results["run_id"] == evaluation_run_id
        assert status == 200 and b"ContextGate" in report

    print(
        json.dumps(
            {
                "status": "passed",
                "answered_run_id": run_id,
                "abstention_reason": abstained["abstention_reason"],
                "evaluation_run_id": evaluation_run_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
