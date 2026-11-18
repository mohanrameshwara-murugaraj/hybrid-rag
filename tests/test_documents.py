import pytest

from hybrid_rag.documents import load_corpus


def test_loads_supported_files_sorted_and_skips_others(tmp_path):
    (tmp_path / "b.md").write_text("# B\nbody\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("plain text\r\nwindows line\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "empty.md").write_text("   \n")

    docs = load_corpus(tmp_path)

    assert [d.source for d in docs] == ["b.md", "sub/a.txt"]
    assert "\r" not in docs[1].text
    assert docs[0].metadata["format"] == "md"


def test_doc_ids_are_stable(tmp_path):
    (tmp_path / "x.md").write_text("hello\n")
    assert load_corpus(tmp_path)[0].doc_id == load_corpus(tmp_path)[0].doc_id


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope")


def test_strips_frontmatter_and_admonition_markers_but_keeps_content(tmp_path):
    (tmp_path / "page.md").write_text(
        "---\ntoc_max_heading_level: 2\n---\n\n# Title\n\n"
        ":::warning Be careful\nKeep this sentence.\n:::\n\nBody --- with dashes.\n"
    )
    text = load_corpus(tmp_path)[0].text
    assert text.startswith("# Title")
    assert ":::" not in text and "toc_max_heading_level" not in text
    assert "Keep this sentence." in text and "Body --- with dashes." in text


def test_non_utf8_file_is_skipped(tmp_path):
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\xfa")
    (tmp_path / "ok.txt").write_text("fine\n")
    assert [d.source for d in load_corpus(tmp_path)] == ["ok.txt"]
