import io
import json
import urllib.error

import pytest

from hybrid_rag import llm as llm_mod
from hybrid_rag.chunking import Chunk
from hybrid_rag.eval_generation import run_one, summarize, summarize_by_split
from hybrid_rag.evaluation import GoldenItem
from hybrid_rag.generation import (
    build_user_prompt,
    claims_from_output,
    clean_missing,
    derive_status,
    render_answer,
)
from hybrid_rag.llm import LLMError, LLMResponse, OllamaLLM
from hybrid_rag.pipeline import RAGPipeline
from hybrid_rag.retrieval import Hit
from hybrid_rag.verification import verify


def hit(i, source="a.md", text="passage", score=1.0):
    return Hit(Chunk(f"c{i}", "d", source, text, "Sec", "header", i), score)


# --- structured output ------------------------------------------------------
def ans(*sents, missing=""):
    return {"sentences": [{"text": t, "citations": c} for t, c in sents], "missing": missing}


def test_claims_from_output_keeps_valid_and_flags_invalid_citations():
    claims = claims_from_output(ans(("A", [1]), ("B", [2, 3, 2]), ("C", [])), 2)
    assert [c.cited for c in claims] == [(1,), (2,), ()]
    assert claims[1].invalid == (3,)


@pytest.mark.parametrize(
    "n,missing,expected",
    [(0, "everything", "insufficient"), (2, "the port", "partial"), (2, "", "answered")],
)
def test_status_is_derived_not_chosen_by_model(n, missing, expected):
    assert derive_status(n, missing) == expected


@pytest.mark.parametrize("raw", ["", "None", "N/A.", "nothing", "  "])
def test_clean_missing_treats_placeholders_as_empty(raw):
    assert clean_missing(raw) == ""


def test_render_answer_adds_markers_and_missing_note():
    claims = claims_from_output(ans(("Uses sqlite3.", [1])), 1)
    text = render_answer(claims, "partial", "the alternatives")
    assert text.startswith("Uses sqlite3 [1].")
    assert "Not covered by the retrieved documentation: the alternatives" in text


def test_prompt_numbers_passages_with_source():
    p = build_user_prompt("Q?", [hit(0, "x.md", "alpha"), hit(1, "y.md", "beta")])
    assert "[1] (x.md | Sec)\nalpha" in p and "[2] (y.md | Sec)\nbeta" in p
    assert p.endswith("Question: Q?")


# --- fakes ----------------------------------------------------------------------
class ScriptedLLM:
    """Returns the next scripted JSON reply; records prompts."""

    name = "scripted"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, system, user, schema=None):
        self.calls.append((system, user))
        return LLMResponse(json.dumps(self.replies.pop(0)), 100, 20, 0.1)


class StubRetriever:
    def __init__(self, hits):
        self.hits = hits

    def get_chunk(self, cid):
        return next(h.chunk for h in self.hits if h.chunk.chunk_id == cid)

    def search(self, q, mode, k):
        return self.hits[:k]


# --- verification ---------------------------------------------------------------
def claims_of(*sents, n=2):
    return claims_from_output(ans(*sents), n)


def test_verify_judges_each_pair_and_excludes_missing():
    claims = claims_of(("Fact one", [1]), ("Fact two", [1, 2]), ("Uncited fact", []))
    judge = ScriptedLLM({"supported": True}, {"supported": False}, {"oops": 1})
    v = verify(claims, [hit(0), hit(1)], judge)
    assert len(judge.calls) == 3  # one call per (sentence, passage) pair
    assert [x.supported for x in v.verdicts] == [True, False, None]
    assert v.judge_missing == 1
    assert v.citation_precision == pytest.approx(1 / 2)  # 1 of 2 judged pairs
    assert v.claim_coverage == pytest.approx(1 / 3)  # claims 1,2 judged + uncited claim 3
    assert "Fact one" in judge.calls[0][1] and "passage" in judge.calls[0][1]


def test_judge_error_counts_as_missing_not_unsupported():
    class Broken:
        name = "broken"

        def chat(self, *a, **k):
            raise LLMError("bad json")

    v = verify(claims_of(("A", [1])), [hit(0)], Broken())
    assert v.judge_missing == 1 and v.citation_precision is None


def test_verify_without_citations_makes_no_judge_call():
    judge = ScriptedLLM()
    v = verify(claims_of(("No citations", [])), [hit(0)], judge)
    assert v.citation_precision is None and v.claim_coverage == 0.0 and not judge.calls


# --- pipeline ---------------------------------------------------------------------
def test_pipeline_answers_and_verifies():
    llm = ScriptedLLM(
        ans(("The default is sqlite3", [1])),
        {"supported": True},
    )
    r = RAGPipeline(StubRetriever([hit(0, text="Default: sqlite3")]), llm).answer("db?")
    assert r.status == "answered" and not r.abstained
    assert r.verification.citation_precision == 1.0 and r.answer == "The default is sqlite3 [1]."
    assert r.sources[0]["source"] == "a.md" and "total_s" in r.timings


def test_partial_answer_is_kept_and_verified():
    llm = ScriptedLLM(
        ans(("Six workflows run", [1]), missing="whether files are shared"),
        {"supported": True},
    )
    r = RAGPipeline(StubRetriever([hit(0)]), llm).answer("matrix?")
    assert r.status == "partial" and not r.abstained and r.verification is not None
    assert "whether files are shared" in r.answer


def test_insufficient_abstains_without_judge_call():
    llm = ScriptedLLM(ans(missing="nothing about LDAP"))
    r = RAGPipeline(StubRetriever([hit(0)]), llm).answer("ldap?")
    assert r.abstained and r.verification is None and len(llm.calls) == 1
    assert "nothing about LDAP" in r.answer


def test_no_sentences_means_insufficient_even_without_missing_text():
    r = RAGPipeline(StubRetriever([hit(0)]), ScriptedLLM(ans())).answer("q")
    assert r.status == "insufficient" and r.abstained


def test_retrieval_gate_skips_the_llm():
    llm = ScriptedLLM()
    r = RAGPipeline(StubRetriever([hit(0, score=0.1)]), llm, min_retrieval_score=0.5).answer("q")
    assert r.abstained and r.status == "retrieval_gate" and not llm.calls


# --- eval -----------------------------------------------------------------------
def test_run_one_and_summarize():
    item = GoldenItem("q1", "lookup", "db?", "sqlite3", ("a.md",), ("",), ("sqlite3",))
    llm = ScriptedLLM(
        ans(("sqlite3", [1])),
        {"supported": True},
        {"verdict": "correct"},
    )
    row = run_one(RAGPipeline(StubRetriever([hit(0, text="Default sqlite3")]), llm), item)
    assert row["correctness"] == "correct" and row["retrieved_evidence"] == 1.0

    missed = GoldenItem("q3", "lookup", "port?", "9000", ("a.md",), ("",), ("port 9000",))
    llm3 = ScriptedLLM(ans(missing="port"))
    row3 = run_one(RAGPipeline(StubRetriever([hit(0, text="nothing")]), llm3), missed)
    assert row3["abstained"] and row3["retrieved_evidence"] == 0.0

    no_ans = GoldenItem("q2", "no_answer", "ldap?", "n/a", ())
    llm2 = ScriptedLLM(ans(missing="ldap"))
    row2 = run_one(RAGPipeline(StubRetriever([hit(0)]), llm2), no_ans)
    assert row2["correctness"] is None

    s = summarize([row, row2, row3])
    assert s["no_answer"]["abstention_rate"] == 1.0
    a = s["answerable"]
    assert a["correct"] == 1 and a["citation_precision"] == 1.0
    assert a["abstained_without_full_evidence"] == 1 and a["abstained_with_full_evidence"] == 0
    assert a["status"]["insufficient"] == 1
    by = summarize_by_split([row, row2, row3])
    assert set(by) == {"all", "dev"} and by["all"] == s


# --- Ollama client ------------------------------------------------------------------
class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ollama_sends_deterministic_options_and_parses(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent.update(json.loads(req.data))
        body = {
            "message": {"content": "{}"},
            "prompt_eval_count": 50,
            "eval_count": 5,
            "load_duration": 2_000_000_000,
            "prompt_eval_duration": 1_500_000_000,
        }
        return FakeResp(json.dumps(body).encode())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    r = OllamaLLM().chat("sys", "user", schema={"type": "object"})
    assert sent["options"] == {"temperature": 0, "seed": 0, "num_ctx": 8192}
    assert sent["format"] == {"type": "object"} and sent["stream"] is False
    assert (r.prompt_tokens, r.output_tokens) == (50, 5)
    assert r.breakdown == {"load_s": 2.0, "prompt_s": 1.5}


def test_ollama_detects_context_overflow(monkeypatch):
    body = {"message": {"content": "x"}, "prompt_eval_count": 8192, "eval_count": 1}
    monkeypatch.setattr(
        llm_mod.urllib.request, "urlopen", lambda req, timeout: FakeResp(json.dumps(body).encode())
    )
    with pytest.raises(LLMError, match="truncated"):
        OllamaLLM().chat("s", "u")


def test_ollama_unreachable_gives_clear_error(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", boom)
    with pytest.raises(LLMError, match="Is the Ollama app running"):
        OllamaLLM().chat("s", "u")
