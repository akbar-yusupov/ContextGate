from contextgate.domain.documents import TextSection, chunk_sections


def test_chunking_is_stable_and_overlaps() -> None:
    sections = [
        TextSection(
            text=("First sentence. Second sentence. Third sentence. " * 20),
            source="policy.md",
        )
    ]
    chunks = chunk_sections(
        sections,
        document_id="policy",
        language="en",
        target_chars=180,
        overlap_chars=30,
    )

    assert len(chunks) > 2
    assert chunks[0].chunk_id == "policy:0"
    assert chunks[1].chunk_id == "policy:1"
    assert chunks[0].source == "policy.md"
    assert chunks[0].language == "en"
