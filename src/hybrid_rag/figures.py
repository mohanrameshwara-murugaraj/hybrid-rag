"""Render README figures from the committed results files only.

    pip install -e ".[figures]"
    python -m hybrid_rag.figures            # -> docs/figures/*.png

Inputs (all produced by the eval CLIs, nothing typed in by hand):
    results/retrieval_header.json, results/retrieval_fixed.json
    results/run{1,2,3}/generation_header_hybrid.jsonl

Static PNGs for the GitHub README (no hover layer). Every bar is directly labelled,
and the README repeats the numbers in tables, so no value relies on color alone.
Colors: validated reference palette, light surface (slots 1-3; slot 3 is below 3:1
contrast, which is why direct labels are mandatory here).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
NEUTRAL = "#b8b7b0"  # "abstained": not a category of quality, so no categorical hue
MODES = ["dense", "sparse", "hybrid", "hybrid_rerank"]
OUTCOMES = ["correct", "partial", "incorrect", "abstained"]


def _style(ax, plt) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bars(ax, groups, series, colors, labels, ylabel, title):
    width = 0.8 / len(series)
    for j, (vals, color, name) in enumerate(zip(series, colors, labels, strict=True)):
        xs = [i + (j - (len(series) - 1) / 2) * width for i in range(len(groups))]
        bars = ax.bar(xs, vals, width * 0.92, color=color, label=name, zorder=2)
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.015,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK_2,
            )
    ax.set_xticks(range(len(groups)), groups, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=12)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK_2, fontsize=9)


def retrieval_figures(results: Path, out: Path, plt) -> list[Path]:
    rep = {
        s: json.loads((results / f"retrieval_{s}.json").read_text()) for s in ("header", "fixed")
    }
    n = rep["header"]["n_answerable"]
    written = []

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    _style(ax, plt)
    _bars(
        ax,
        MODES,
        [[rep[s]["modes"][m]["ev_recall@5"] for m in MODES] for s in ("header", "fixed")],
        SLOT[:2],
        ["header-aware chunks", "fixed-size chunks"],
        "Evidence Recall@5",
        f"Answer passage in the top 5, by retrieval mode ({n} answerable questions)",
    )
    p = out / "retrieval_by_mode.png"
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    _style(ax, plt)
    h = rep["header"]["modes"]
    _bars(
        ax,
        MODES,
        [[h[m]["ev_recall@5"] for m in MODES], [h[m]["ev_recall@10"] for m in MODES]],
        [SLOT[0], SLOT[2]],
        ["top 5 passages", "top 10 passages"],
        "Evidence Recall",
        "Passing 10 passages instead of 5 (header-aware chunks)",
    )
    p = out / "recall_at_k.png"
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)
    return written


def _outcomes(path: Path, split: str | None) -> dict[str, int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [
        r for r in rows if r["type"] != "no_answer" and (split is None or r.get("split") == split)
    ]
    counts = dict.fromkeys(OUTCOMES, 0)
    for r in rows:
        counts["abstained" if r["abstained"] else (r.get("correctness") or "incorrect")] += 1
    return counts


def generation_figure(results: Path, out: Path, plt) -> Path:
    gen = "generation_header_hybrid.jsonl"
    rows = [
        ("Run 1 · dev", _outcomes(results / "run1" / gen, None)),
        ("Run 2 · dev", _outcomes(results / "run2" / gen, None)),
        ("Run 3 · dev", _outcomes(results / "run3" / gen, "dev")),
        ("Run 3 · held-out", _outcomes(results / "run3" / gen, "heldout")),
    ]
    colors = dict(zip(OUTCOMES, SLOT + [NEUTRAL], strict=True))
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    _style(ax, plt)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for i, (_, c) in enumerate(reversed(rows)):
        total, left = sum(c.values()), 0.0
        for o in OUTCOMES:
            share = c[o] / total
            if share:
                ax.barh(
                    i,
                    share,
                    left=left,
                    color=colors[o],
                    edgecolor=SURFACE,
                    linewidth=2,
                    label=o if i == 0 else None,
                    zorder=2,
                )
                if share >= 0.06:
                    ax.text(
                        left + share / 2,
                        i,
                        str(c[o]),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=INK,
                    )
            left += share
    ax.set_yticks(
        range(len(rows)), [f"{name} (n={sum(c.values())})" for name, c in reversed(rows)], color=INK
    )
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(
        "Answer outcomes on answerable questions (LLM-judged correctness)",
        loc="left",
        color=INK,
        fontsize=11,
        pad=12,
    )
    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        labelcolor=INK_2,
        fontsize=9,
    )
    p = out / "generation_by_run.png"
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Render README figures from results files")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.matplotlib = matplotlib
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in [*retrieval_figures(results, out, plt), generation_figure(results, out, plt)]:
        print("wrote", p)


if __name__ == "__main__":
    main()
