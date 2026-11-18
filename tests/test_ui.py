"""UI tests. SIMULATED: the API client is replaced by a fake returning canned responses,
so these check rendering and interaction, not answer quality."""

from pathlib import Path

import pytest

from hybrid_rag.ui_render import (
    answer_html,
    cited_passages,
    hit_html,
    overlap,
    source_label,
    status_badge,
)

RESULT = {
    "question": "q",
    "answer": "A [1]. B [2].",
    "status": "answered",
    "missing": "",
    "abstained": False,
    "claims": [
        {"text": "Use <b>failure: ignore</b>", "citations": [1]},
        {"text": "B", "citations": [2]},
    ],
    "sources": [
        {"n": 1, "chunk_id": "c1", "source": "a.md", "section": "A > B", "score": 0.1, "text": "x"},
        {"n": 2, "chunk_id": "c2", "source": "b.md", "section": "", "score": 0.1, "text": "y"},
        {"n": 3, "chunk_id": "c3", "source": "c.md", "section": "", "score": 0.1, "text": "z"},
    ],
    "verification": {
        "verdicts": [
            {"claim_index": 0, "passage": 1, "supported": True, "judge_missing": False},
            {"claim_index": 1, "passage": 2, "supported": False, "judge_missing": False},
        ]
    },
    "timings": {"total_s": 4.2, "retrieval_s": 0.011},
}


def test_answer_html_escapes_text_and_marks_verdicts_with_icons():
    html = answer_html(RESULT)
    assert "&lt;b&gt;failure: ignore&lt;/b&gt;" in html  # model text is escaped
    assert 'class="chip good"' in html and "1 ✓" in html
    assert 'class="chip bad"' in html and "2 ✗" in html


def test_inline_code_is_rendered_after_escaping():
    from hybrid_rag.ui_render import inline

    assert inline("add `failure: ignore` <x>") == "add <code>failure: ignore</code> &lt;x&gt;"


def test_hit_splits_folder_and_file_and_labels_score_kind():
    hit = {
        "rank": 1,
        "chunk_id": "c",
        "source": "30-admin/10-config/30-agent.md",
        "section": "Agent > `labels`",
        "score": 0.0082,
    }
    html = hit_html(hit, "hybrid", {"c": ["hybrid"]}, {})
    assert '<span class="file">30-agent.md</span>' in html and "30-admin/10-config/" in html
    assert "RRF 0.0082" in html and "<code>labels</code>" in html


def test_unverified_answer_shows_neutral_chips_and_missing_note():
    r = {**RESULT, "verification": None, "status": "partial", "missing": "the default value"}
    html = answer_html(r)
    assert "chip good" not in html and "1 ?" in html
    assert "Not covered by the retrieved docs" in html and "the default value" in html


def test_insufficient_answer_has_no_duplicate_text():
    r = {
        **RESULT,
        "claims": [],
        "status": "insufficient",
        "missing": "LDAP is not covered",
        "answer": "Not covered by the retrieved documentation: LDAP is not covered",
    }
    html = answer_html(r)
    assert html.count("LDAP is not covered") == 1


def test_status_badge_pairs_icon_and_label():
    assert "✓ Answered" in status_badge("answered")
    assert "! Partial answer" in status_badge("partial")
    assert "Not in the docs" in status_badge("insufficient")


def test_source_labels_reflect_citation_verdicts():
    cited = cited_passages(RESULT)
    labels = [source_label(s, cited.get(s["n"])) for s in RESULT["sources"]]
    assert labels[0].endswith("✓ supported") and "a.md › A > B" in labels[0]
    assert labels[1].endswith("✗ unsupported")
    assert labels[2].endswith("not cited")
    mixed = {
        **RESULT,
        "claims": [{"text": "A", "citations": [1]}, {"text": "B", "citations": [1]}],
        "verification": {
            "verdicts": [
                {"claim_index": 0, "passage": 1, "supported": True},
                {"claim_index": 1, "passage": 1, "supported": False},
            ]
        },
    }
    assert source_label(RESULT["sources"][0], cited_passages(mixed)[1]).endswith(
        "mixed (1 ✓ · 1 ✗)"
    )


def test_overlap_and_hit_tags():
    res = {
        "dense": {
            "hits": [{"rank": 1, "chunk_id": "c1", "source": "a.md", "section": "", "score": 0.9}]
        },
        "sparse": {
            "hits": [
                {"rank": 1, "chunk_id": "c1", "source": "a.md", "section": "", "score": 7.0},
                {"rank": 2, "chunk_id": "c2", "source": "<b>.md", "section": "", "score": 3.0},
            ]
        },
    }
    seen = overlap(res)
    assert seen == {"c1": ["dense", "sparse"], "c2": ["sparse"]}
    labels = {"dense": "Dense", "sparse": "BM25"}
    assert "also in Dense" in hit_html(res["sparse"]["hits"][0], "sparse", seen, labels)
    only = hit_html(res["sparse"]["hits"][1], "sparse", seen, labels)
    assert "only here" in only and "&lt;b&gt;.md" in only and "BM25 3" in only


# --- the real Streamlit page, driven by AppTest with a fake API client -------------------
st_testing = pytest.importorskip("streamlit.testing.v1")


class FakeClient:
    calls: list = []

    def __init__(self, base_url, timeout=900):
        self.base_url = base_url

    def health(self):
        return True

    def documents(self):
        return {
            "strategy": "header",
            "n_documents": 2,
            "documents": [{"source": "a.md", "chunks": 3}, {"source": "b.md", "chunks": 1}],
        }

    def ask(self, question, mode, k, verify):
        FakeClient.calls.append(("ask", question, mode, k, verify))
        return {**RESULT, "question": question}

    def search(self, question, modes, k):
        FakeClient.calls.append(("search", question, tuple(modes), k))
        hit = {
            "rank": 1,
            "chunk_id": "c1",
            "source": "a.md",
            "section": "",
            "score": 1.0,
            "text": "t",
        }
        return {
            "question": question,
            "k": k,
            "results": {m: {"ms": 1.0, "hits": [hit]} for m in modes},
        }


@pytest.fixture
def app(monkeypatch):
    import hybrid_rag.ui_client as uc

    monkeypatch.setattr(uc, "ApiClient", FakeClient)
    FakeClient.calls = []
    page = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
    # Generous timeout: the first run imports Streamlit's heavy dependencies (pandas,
    # pyarrow) cold, which took over 30 s on an Intel Mac; later runs take ~1 s.
    at = st_testing.AppTest.from_file(page, default_timeout=180)
    return at.run()


def test_page_renders_header_and_tabs(app):
    assert not app.exception
    html = " ".join(m.value for m in app.markdown)
    assert "API online" in html and "2 documents indexed" in html
    assert [t.label for t in app.tabs] == ["Ask", "Compare retrieval", "Documents"]


def test_example_question_runs_ask_and_shows_cited_answer(app):
    app.button[0].click().run()  # first example question
    assert not app.exception
    assert FakeClient.calls[0][:2] == ("ask", "How do I make a lint step allowed to fail?")
    assert FakeClient.calls[0][2:] == ("hybrid", 10, True)  # sidebar defaults
    html = " ".join(m.value for m in app.markdown)
    assert "✓ Answered" in html and "1 ✓" in html and "2 ✗" in html
    assert [e.label for e in app.expander][0].startswith("[1]  a.md")


def test_compare_example_queries_three_retrievers(app):
    cmp_buttons = [b for b in app.button if b.key and b.key.startswith("cmp-")]
    cmp_buttons[0].click().run()
    assert not app.exception
    assert FakeClient.calls[-1] == (
        "search",
        "WOODPECKER_MAX_WORKFLOWS",
        ("dense", "sparse", "hybrid"),
        5,
    )
    html = " ".join(m.value for m in app.markdown)
    assert "1 of 5 passages found by all three" in html
