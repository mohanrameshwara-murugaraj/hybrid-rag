"""Pure rendering helpers for the Streamlit UI (HTML strings), kept separate so they are
unit-tested without a browser. All user and document text is HTML-escaped.

Visual rules: status and citation verdicts always pair a color with an icon AND a text
label, never color alone; colors come from a validated status palette."""

from __future__ import annotations

import re
from html import escape

STATUS = {  # status -> (icon, label, css class)
    "answered": ("✓", "Answered", "good"),
    "partial": ("!", "Partial answer", "warn"),
    "insufficient": ("○", "Not in the docs", "neutral"),
    "retrieval_gate": ("○", "Nothing relevant found", "neutral"),
}
SCORE_KIND = {"dense": "cosine", "sparse": "BM25", "hybrid": "RRF", "hybrid_rerank": "rerank"}
_CODE = re.compile(r"`([^`]+)`")


def inline(text: str) -> str:
    """Escape text, then render `inline code` (models and headings use backticks)."""
    return _CODE.sub(r"<code>\1</code>", escape(text))


def split_path(path: str) -> tuple[str, str]:
    folder, _, name = path.rpartition("/")
    return (folder + "/" if folder else ""), name


VERDICT = {
    True: ("✓", "supported", "good"),
    False: ("✗", "unsupported", "bad"),
    None: ("?", "not checked", "neutral"),
}


def status_badge(status: str) -> str:
    icon, label, cls = STATUS.get(status, ("○", status, "neutral"))
    return f'<span class="badge {cls}">{icon} {escape(label)}</span>'


def verdict_map(verification: dict | None) -> dict[tuple[int, int], bool | None]:
    """(claim_index, passage) -> supported (True/False/None)."""
    if not verification:
        return {}
    return {(v["claim_index"], v["passage"]): v["supported"] for v in verification["verdicts"]}


def chip(n: int, supported: bool | None) -> str:
    icon, label, cls = VERDICT[supported]
    return f'<span class="chip {cls}" title="passage {n}: {label}">{n} {icon}</span>'


def answer_html(result: dict) -> str:
    verdicts = verdict_map(result.get("verification"))
    checked = result.get("verification") is not None
    parts = []
    for i, c in enumerate(result.get("claims") or []):
        chips = "".join(
            chip(n, verdicts.get((i, n)) if checked else None) for n in c.get("citations", [])
        )
        parts.append(f"<p>{inline(c['text'].rstrip('.'))}.{chips}</p>")
    missing = (result.get("missing") or "").strip()
    if parts:
        body = "".join(parts)
    else:
        body = "<p>The retrieved documentation does not answer this question.</p>"
    if missing:
        note = f"<b>Not covered by the retrieved docs:</b> {inline(missing)}"
        body += f'<div class="missing">{note}</div>'
    return f'<div class="answer">{body}</div>'


def kpis_html(result: dict) -> str:
    v = result.get("verification") or {}
    verdicts = [x for x in v.get("verdicts", []) if x["supported"] is not None]
    supported = sum(1 for x in verdicts if x["supported"])
    t = result.get("timings", {})
    items = [
        (f"{supported}/{len(verdicts)}" if verdicts else "—", "citations supported"),
        (f"{len(result.get('sources', []))}", "passages read"),
        (f"{t.get('total_s', 0):.1f} s", "total time"),
        (f"{t.get('retrieval_s', 0) * 1000:.0f} ms", "retrieval"),
    ]
    cells = "".join(
        f'<div class="kpi"><div class="v">{a}</div><div class="l">{b}</div></div>' for a, b in items
    )
    return f'<div class="kpis">{cells}</div>'


def cited_passages(result: dict) -> dict[int, list[bool | None]]:
    """passage number -> verdicts of the sentences that cite it."""
    verdicts = verdict_map(result.get("verification"))
    out: dict[int, list[bool | None]] = {}
    for i, c in enumerate(result.get("claims") or []):
        for n in c.get("citations", []):
            out.setdefault(n, []).append(verdicts.get((i, n)))
    return out


def source_label(src: dict, verdicts: list[bool | None] | None) -> str:
    """Plain-text expander label (Streamlit expander labels are not HTML)."""
    where = src["source"] + (f" › {src['section']}" if src.get("section") else "")
    if not verdicts:
        return f"[{src['n']}]  {where}  ·  not cited"
    ok = sum(1 for v in verdicts if v)
    bad = sum(1 for v in verdicts if v is False)
    mark = "✓ supported" if ok and not bad else ("✗ unsupported" if bad and not ok else "")
    if not mark:
        mark = f"mixed ({ok} ✓ · {bad} ✗)"
    if all(v is None for v in verdicts):
        mark = "cited"
    return f"[{src['n']}]  {where}  ·  {mark}"


def passage_html(text: str) -> str:
    return f'<div class="passage">{escape(text)}</div>'


def overlap(results: dict) -> dict[str, list[str]]:
    """chunk_id -> modes that retrieved it, for the comparison view."""
    seen: dict[str, list[str]] = {}
    for mode, r in results.items():
        for h in r["hits"]:
            seen.setdefault(h["chunk_id"], []).append(mode)
    return seen


def hit_html(hit: dict, mode: str, seen: dict[str, list[str]], labels: dict[str, str]) -> str:
    others = [labels.get(m, m) for m in seen.get(hit["chunk_id"], []) if m != mode]
    tags = "".join(f'<span class="tag">also in {escape(o)}</span>' for o in others) or (
        '<span class="tag">only here</span>'
    )
    folder, name = split_path(hit["source"])
    sec = f'<div class="sec">{inline(hit["section"])}</div>' if hit.get("section") else ""
    kind = SCORE_KIND.get(mode, "score")
    return (
        f'<div class="hit"><div><span class="rank">#{hit["rank"]}</span>'
        f'<span class="file">{escape(name)}</span></div>'
        f'<div class="dir">{escape(folder)}</div>{sec}'
        f'<div class="meta">{kind} {hit["score"]:.3g} · {tags}</div></div>'
    )
