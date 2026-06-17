import json
from collections import Counter
from pathlib import Path


def test_demo_dataset_qrels_are_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = [
        json.loads(line)
        for line in (root / "demo" / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    documents = {
        path.stem: path.read_text(encoding="utf-8")
        for path in (root / "demo" / "documents").glob("*.md")
    }
    combined_text = "\n".join(documents.values()) + "\n".join(
        json.dumps(row, ensure_ascii=False) for row in rows
    )

    assert len(rows) == 150
    assert Counter(row["language"] for row in rows) == {
        "en": 50,
        "ru": 50,
        "uz": 50,
    }
    assert sum(not row["answerable"] for row in rows) == 15
    assert any(row["answerable"] and row["expected_facts"] for row in rows)
    assert any((not row["answerable"]) and "unanswerable" in row["tags"] for row in rows)
    assert all(row["group_id"] for row in rows)
    for row in rows:
        if not row["answerable"]:
            assert row["relevant_chunk_ids"] == []
            continue
        assert len(row["relevant_chunk_ids"]) == 1
        document_id, chunk_index = row["relevant_chunk_ids"][0].rsplit(":", 1)
        assert chunk_index == "0"
        assert document_id in documents
        assert all(fact in documents[document_id] for fact in row["expected_facts"])
    assert not any(marker in combined_text for marker in ("Рћ", "Рџ", "Рњ", "вЂ"))
