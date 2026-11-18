"""Streamlit UI for the hybrid RAG API.

    uvicorn hybrid_rag.api:app --port 8000          # terminal 1: the API
    streamlit run app/streamlit_app.py              # terminal 2: this UI

The UI only talks to the API (RAG_API_URL, default http://localhost:8000).
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from hybrid_rag import ui_client
from hybrid_rag.ui_render import (
    answer_html,
    cited_passages,
    hit_html,
    kpis_html,
    overlap,
    passage_html,
    source_label,
    status_badge,
)

MODE_LABELS = {
    "hybrid": "Hybrid (BM25 + dense, RRF)",
    "dense": "Dense (semantic)",
    "sparse": "BM25 (keyword)",
    "hybrid_rerank": "Hybrid + cross-encoder rerank",
}
ASK_EXAMPLES = [
    "How do I make a lint step allowed to fail?",
    "Which address does a remote agent connect to?",
    "Do pull requests get access to my secrets?",
    "How do I push a commit without starting a pipeline?",
]
COMPARE_EXAMPLES = [
    "WOODPECKER_MAX_WORKFLOWS",
    "how many pipelines can one agent run in parallel",
]

st.set_page_config(page_title="Woodpecker Docs · Hybrid RAG", page_icon="📘", layout="wide")
st.markdown(
    f"<style>{Path(__file__).with_name('style.css').read_text()}</style>", unsafe_allow_html=True
)

# --- sidebar: connection + settings -------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    api_url = st.text_input("API URL", os.environ.get("RAG_API_URL", "http://localhost:8000"))
    client = ui_client.ApiClient(api_url)
    mode = st.selectbox("Retrieval mode", list(MODE_LABELS), format_func=MODE_LABELS.get)
    k = st.slider("Passages given to the model", 1, 20, 10)
    verify = st.toggle("Verify citations", value=True, help="One LLM judge call per citation.")
    st.caption(
        "Answers come from qwen2.5:7b via Ollama. On a CPU-only laptop one answer can take "
        "a few minutes; with a GPU it takes seconds. Retrieval alone is instant."
    )

online = client.health()
docs = None
if online:
    try:
        docs = client.documents()
    except ui_client.ApiError:
        docs = None

# --- header -------------------------------------------------------------------------
dot, state = ("good", "online") if online else ("bad", "offline")
pills = [f'<span class="pill"><span class="dot {dot}"></span>API {state}</span>']
if docs:
    pills.append(f'<span class="pill">{docs["n_documents"]} documents indexed</span>')
    pills.append(f'<span class="pill">{docs["strategy"]} chunks</span>')
st.markdown(
    '<div class="hr-title">Woodpecker CI docs assistant</div>'
    '<div class="hr-sub">Hybrid retrieval · grounded answers · every citation checked</div>'
    f'<div style="margin:.8rem 0 .4rem 0">{"".join(pills)}</div>',
    unsafe_allow_html=True,
)
if not online:
    st.warning(
        f"Can't reach the API at {api_url}. Start it with "
        "`uvicorn hybrid_rag.api:app --port 8000`, or change the URL in the sidebar."
    )

ask_tab, compare_tab, docs_tab = st.tabs(["Ask", "Compare retrieval", "Documents"])

# --- Ask ----------------------------------------------------------------------------
with ask_tab:
    cols = st.columns(len(ASK_EXAMPLES))
    for col, ex in zip(cols, ASK_EXAMPLES, strict=True):
        if col.button(ex, width="stretch", disabled=not online):
            st.session_state["question"] = ex
            st.session_state["run_ask"] = True
    with st.form("ask"):
        question = st.text_area(
            "Question", key="question", height=80, placeholder="Ask about the Woodpecker docs…"
        )
        submitted = st.form_submit_button("Ask", type="primary", disabled=not online)
    if (submitted or st.session_state.pop("run_ask", False)) and question.strip():
        with st.spinner("Searching the docs and writing a cited answer…"):
            try:
                st.session_state["result"] = client.ask(question, mode, k, verify)
            except ui_client.ApiError as e:
                st.session_state.pop("result", None)
                st.error(str(e))

    result = st.session_state.get("result")
    if result:
        st.markdown(
            f'<div class="card">{status_badge(result["status"])}'
            f"{answer_html(result)}{kpis_html(result)}</div>",
            unsafe_allow_html=True,
        )
        cited = cited_passages(result)
        st.markdown("**Passages the model read**")
        for src in result["sources"]:
            with st.expander(source_label(src, cited.get(src["n"])), expanded=False):
                st.markdown(passage_html(src.get("text", "")), unsafe_allow_html=True)

# --- Compare retrieval ----------------------------------------------------------------
with compare_tab:
    st.caption(
        "The same question through three retrievers, no LLM involved. Exact identifiers favour "
        "BM25, paraphrases favour dense search; hybrid fuses both rankings."
    )
    cols = st.columns(len(COMPARE_EXAMPLES) + 2)
    for col, ex in zip(cols, COMPARE_EXAMPLES, strict=False):
        if col.button(ex, key=f"cmp-{ex}", width="stretch", disabled=not online):
            st.session_state["cmp_q"] = ex
            st.session_state["run_cmp"] = True
    with st.form("compare"):
        cq = st.text_input("Query", key="cmp_q", placeholder="e.g. WOODPECKER_MAX_WORKFLOWS")
        c1, c2 = st.columns([3, 1])
        ck = c1.slider("Top k", 3, 10, 5)
        show_text = c2.toggle("Show passage text", value=False)
        go = st.form_submit_button("Compare", type="primary", disabled=not online)
    if (go or st.session_state.pop("run_cmp", False)) and cq.strip():
        try:
            st.session_state["cmp"] = client.search(cq, ["dense", "sparse", "hybrid"], ck)
            st.session_state["cmp_text"] = show_text
        except ui_client.ApiError as e:
            st.session_state.pop("cmp", None)
            st.error(str(e))

    cmp = st.session_state.get("cmp")
    if cmp:
        res = cmp["results"]
        seen = overlap(res)
        in_all = sum(1 for modes in seen.values() if len(modes) == len(res))
        st.markdown(
            f'<span class="pill">{in_all} of {cmp["k"]} passages found by all three</span>'
            f'<span class="pill">{len(seen)} distinct passages overall</span>',
            unsafe_allow_html=True,
        )
        for col, m in zip(st.columns(len(res)), res, strict=True):
            with col:
                st.markdown(
                    f'<div class="col-head">{MODE_LABELS[m]}</div>'
                    f'<div class="col-sub">{res[m]["ms"]:.0f} ms</div>',
                    unsafe_allow_html=True,
                )
                if not res[m]["hits"]:
                    st.caption("No passage matched.")
                for h in res[m]["hits"]:
                    st.markdown(hit_html(h, m, seen, MODE_LABELS), unsafe_allow_html=True)
                    if st.session_state.get("cmp_text"):
                        st.markdown(passage_html(h["text"]), unsafe_allow_html=True)

# --- Documents --------------------------------------------------------------------------
with docs_tab:
    if docs:
        st.dataframe(
            docs["documents"],
            column_config={"source": "Document", "chunks": st.column_config.NumberColumn("Chunks")},
            width="stretch",
            hide_index=True,
        )
    elif online:
        st.info("The index is not built yet: run `python -m hybrid_rag.build_index`.")
    st.markdown("**Add documents**")
    st.caption(
        "Markdown or text files are saved under `ingested/` in the corpus and the whole index is "
        "rebuilt. That takes a few minutes on a CPU."
    )
    files = st.file_uploader("Files", type=["md", "markdown", "txt"], accept_multiple_files=True)
    if st.button("Ingest and rebuild index", disabled=not (online and files)):
        payload = [{"path": f.name, "content": f.getvalue().decode("utf-8")} for f in files]
        with st.spinner("Rebuilding the index…"):
            try:
                out = client.ingest(payload)
                idx = out["index"]
                st.success(
                    f"Added {len(out['written'])} file(s); index now has "
                    f"{idx['n_chunks']} chunks from {idx['n_documents']} documents."
                )
            except ui_client.ApiError as e:
                st.error(str(e))
            except UnicodeDecodeError:
                st.error("Files must be UTF-8 text.")
