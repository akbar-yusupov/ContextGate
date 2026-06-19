import pytest
from pypdf import PdfWriter

from contextgate.adapters.local.loaders import (
    iter_document_paths,
    iter_sections,
    load_sections,
    validate_document_file,
)
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


def test_chunking_hard_wraps_long_unpunctuated_content() -> None:
    sections = [TextSection(text="x" * 10_000, source="generated.txt")]

    chunks = chunk_sections(
        sections,
        document_id="generated",
        language="en",
        target_chars=1_200,
        overlap_chars=180,
    )

    assert len(chunks) > 5
    assert max(len(chunk.text) for chunk in chunks) <= 1_200


def test_loader_rejects_extension_spoofing_and_binary_text(tmp_path) -> None:
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"not a pdf")
    binary_text = tmp_path / "binary.txt"
    binary_text.write_bytes(b"hello\x00world")

    with pytest.raises(ValueError, match="signature"):
        validate_document_file(fake_pdf)
    with pytest.raises(ValueError, match="binary"):
        validate_document_file(binary_text)


def test_loader_enforces_pdf_page_and_text_extraction_limits(tmp_path) -> None:
    pdf = tmp_path / "many-pages.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    with pdf.open("wb") as handle:
        writer.write(handle)
    text = tmp_path / "large.md"
    text.write_text("123456", encoding="utf-8")

    with pytest.raises(ValueError, match="page limit"):
        list(iter_sections(pdf, max_pdf_pages=1))
    with pytest.raises(ValueError, match="extraction limit"):
        list(iter_sections(text, max_extracted_chars=5))


def test_loader_extracts_text_and_strips_non_content_html(tmp_path) -> None:
    markdown = tmp_path / "policy.md"
    markdown.write_text("Policy text", encoding="utf-8")
    html = tmp_path / "policy.html"
    html.write_text(
        "<html><nav>menu</nav><style>x</style><script>bad()</script><p>Useful text</p></html>",
        encoding="utf-8",
    )
    ignored = tmp_path / "image.png"
    ignored.write_bytes(b"png")

    assert load_sections(markdown)[0].text == "Policy text"
    html_text = load_sections(html)[0].text
    assert "Useful text" in html_text
    assert "menu" not in html_text and "bad" not in html_text
    assert list(iter_document_paths(tmp_path)) == [html, markdown]
    assert list(iter_document_paths(ignored)) == []


def test_loader_rejects_encrypted_pdf(tmp_path) -> None:
    pdf = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret")
    with pdf.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(ValueError, match="Encrypted"):
        list(iter_sections(pdf))


def test_loader_extracts_pdf_pages_and_rejects_unknown_types(tmp_path, monkeypatch) -> None:
    unsupported = tmp_path / "document.csv"
    unsupported.write_text("value", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        validate_document_file(unsupported)

    pdf = tmp_path / "text.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")

    class Page:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class Reader:
        is_encrypted = False
        pages = [Page(""), Page("Supported statement")]

    monkeypatch.setattr("contextgate.adapters.local.loaders.PdfReader", lambda _: Reader())
    sections = list(iter_sections(pdf))

    assert sections == [
        TextSection(
            text="Supported statement",
            source="text.pdf",
            metadata={"page": 2},
        )
    ]

    with pytest.raises(ValueError, match="extracted text"):
        list(iter_sections(pdf, max_extracted_chars=5))
