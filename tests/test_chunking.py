import pytest

from hybrid_rag.chunking import chunk_by_headers, chunk_documents, chunk_fixed, split_sections
from hybrid_rag.documents import Document

MD = """# Install

Intro text.

## Docker

Run this:

```bash
# this is a shell comment, not a heading
docker compose up
```

## Config

Set `MAX_WORKERS=4`.

# Usage

Call the API.
"""


def doc(text: str) -> Document:
    return Document(doc_id="d1", source="guide.md", text=text)


def test_fixed_windows_cover_text_with_overlap():
    text = "abcdefghij" * 30  # 300 chars
    chunks = chunk_fixed(doc(text), size=100, overlap=20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].text[-20:] == chunks[1].text[:20]  # overlap preserved
    assert chunks[-1].text.endswith(text[-10:])  # tail not dropped


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_fixed_rejects_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        chunk_fixed(doc("x" * 50), size=size, overlap=overlap)


def test_sections_track_heading_path_and_ignore_code_comments():
    sections = split_sections(MD)
    paths = [p for p, _ in sections]
    assert paths == ["Install", "Install > Docker", "Install > Config", "Usage"]
    docker_body = dict(sections)["Install > Docker"]
    assert "# this is a shell comment" in docker_body


def test_small_sections_merge_forward():
    md = "# Parent\n\n## Child\n\n" + "Real content. " * 20 + "\n\n## Tail\n\nshort"
    chunks = chunk_by_headers(doc(md), min_chars=100)
    assert len(chunks) == 1
    assert chunks[0].section == "Parent > Child"
    assert chunks[0].text.startswith("# Parent") and chunks[0].text.endswith("short")


def test_header_chunks_carry_section_and_ids():
    chunks = chunk_by_headers(doc(MD), min_chars=0)
    assert chunks[2].section == "Install > Config"
    assert "MAX_WORKERS" in chunks[2].text
    assert chunks[2].chunk_id == "d1:header:2"
    assert all(c.strategy == "header" for c in chunks)


def test_oversized_section_is_subsplit():
    long_md = "# Big\n\n" + ("word " * 1000)
    chunks = chunk_by_headers(doc(long_md), max_chars=500, overlap=50, min_chars=0)
    assert len(chunks) > 1
    assert all(c.section == "Big" and c.char_count <= 500 for c in chunks)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        chunk_documents([doc(MD)], "semantic")
