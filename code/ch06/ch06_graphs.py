"""Generate the Chapter 6 canonical-graph figures.

Emits two figures from a single committed description of the canonical
one-head, one-block decoder graph:

  figures/ch06_forward_graph.svg  -- forward dataflow (used in Section 6.2)
  figures/ch06_backward_graph.svg -- authentic reverse pass (used in 6.18),
                                    including parameter-adjoint side edges

The reverse figure is the manuscript's reverse-mode accounting diagram: every
adjoint is a node, every merge is an explicit accumulation, and every trainable
parameter receives a dashed adjoint edge.

Pure standard library, deterministic output, no external graph dependency.
A graphviz-compatible .dot source is committed alongside each SVG so the figure
can be regenerated with `dot -Tsvg` by a reviewer who prefers it.

Usage:  python code/ch06_graphs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve imports for the migrated layout: a chapter script lives at
# <root>/code/chNN/ and needs its own directory (for sibling imports) and
# <root>/code (where the shared `tgd` package lives).
_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# <root>/code/ch06 -> <root>: the figure directory is a sibling of code/,
# not of this script's directory.
ROOT = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(ROOT, "figures")

# ---------------------------------------------------------------- style ----

FONT = "Arial"
NODE_FILL = "#ffffff"
NODE_STROKE = "#1a1a1a"
TENSOR_FILL = "#eef3f8"
PARAM_FILL = "#f6f0e8"
MERGE_FILL = "#eaf3ea"
SOLID = "#1a1a1a"
DASHED = "#8a6d3b"
MERGE_EDGE = "#1f6f3f"
LABEL_GREY = "#444444"


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def wrap(text: str, width: int = 26) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if len(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------- layout ----

class Node:
    def __init__(self, nid, label, col, row, kind="tensor", w=210, h=44):
        self.id = nid
        self.label = label
        self.col = col
        self.row = row
        self.kind = kind
        self.w = w
        self.h = h

    @property
    def x(self):
        return 40 + self.col * 290

    @property
    def y(self):
        return 40 + self.row * 78


def render(nodes, edges, title, width, height, dot_name, out_svg):
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}pt" height="{height}pt" '
        f'viewBox="0 0 {width} {height}">',
        f"<title>{esc(title)}</title>",
        f"<!-- source: code/ch06_graphs.py  |  graphviz source: figures/{dot_name} -->",
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{SOLID}"/></marker>'
        '<marker id="arrowm" viewBox="0 0 10 10" refX="10" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{MERGE_EDGE}"/></marker>'
        '<marker id="arrowd" viewBox="0 0 10 10" refX="10" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{DASHED}"/></marker></defs>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="24" text-anchor="middle" '
        f'font-family="{FONT}" font-size="13" font-weight="bold" '
        f'fill="{SOLID}">{esc(title)}</text>',
    ]

    by_id = {n.id: n for n in nodes}

    def anchor(n, toward_x, toward_y):
        """Edge attachment point on node boundary toward the target."""
        cx, cy = n.x + n.w / 2, n.y + n.h / 2
        dx, dy = toward_x - cx, toward_y - cy
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return cx, cy
        sx = (n.w / 2) / abs(dx) if dx else math.inf
        sy = (n.h / 2) / abs(dy) if dy else math.inf
        s = min(sx, sy)
        return cx + dx * s, cy + dy * s

    for e in edges:
        a, b = by_id[e["from"]], by_id[e["to"]]
        ax, ay = anchor(a, b.x + b.w / 2, b.y + b.h / 2)
        bx, by = anchor(b, a.x + a.w / 2, a.y + a.h / 2)
        colour = {"merge": MERGE_EDGE, "param": DASHED}.get(e.get("style"), SOLID)
        marker = {"merge": "arrowm", "param": "arrowd"}.get(e.get("style"), "arrow")
        dash = ' stroke-dasharray="5,3"' if e.get("style") == "param" else ""
        parts.append(
            f'<path d="M {ax:.1f} {ay:.1f} L {bx:.1f} {by:.1f}" '
            f'fill="none" stroke="{colour}" stroke-width="1.4"{dash} '
            f'marker-end="url(#{marker})"/>'
        )
        if e.get("label"):
            parts.append(
                f'<text x="{(ax + bx) / 2:.1f}" y="{(ay + by) / 2 - 4:.1f}" '
                f'text-anchor="middle" font-family="{FONT}" font-size="9" '
                f'fill="{LABEL_GREY}">{esc(e["label"])}</text>'
            )

    for n in nodes:
        fill = {
            "param": PARAM_FILL, "merge": MERGE_FILL,
        }.get(n.kind, TENSOR_FILL)
        parts.append(
            f'<rect x="{n.x}" y="{n.y}" width="{n.w}" height="{n.h}" '
            f'rx="5" fill="{fill}" stroke="{NODE_STROKE}" stroke-width="1.2"/>'
        )
        lines = wrap(n.label, 24)
        start = n.y + n.h / 2 - (len(lines) - 1) * 6 + 3
        for i, ln in enumerate(lines):
            style = ' font-style="italic"' if n.kind == "param" else ""
            parts.append(
                f'<text x="{n.x + n.w / 2:.1f}" y="{start + i * 12:.1f}" '
                f'text-anchor="middle" font-family="{FONT}" font-size="9.5"'
                f'{style} fill="{SOLID}">{esc(ln)}</text>'
            )

    parts.append(
        f'<text x="{width / 2:.0f}" y="{height - 12}" text-anchor="middle" '
        f'font-family="{FONT}" font-size="9" fill="{LABEL_GREY}">'
        f'solid arrow = tensor adjoint flows backward &#183; dashed = parameter '
        f'adjoint &#183; green = accumulation merge</text>'
    )
    parts.append("</svg>")

    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")
    return out_svg


# ------------------------------------------------------- forward figure ----

FWD_NODES = [
    Node("X", "Token embeddings, residual X", 0, 0),
    Node("N1", "RMSNorm 1 (Pre-RMSNorm)", 0, 1),
    Node("Q", "Q projection + RoPE", 0, 2),
    Node("K", "K projection + RoPE", 0, 3),
    Node("V", "V projection (unrotated)", 0, 4),
    Node("A", "Scaled scores + causal mask + row softmax", 1, 2),
    Node("O", "O = A V, concat heads, then W_O", 1, 3),
    Node("R", "First residual add: R = X + Y", 1, 4),
    Node("N2", "RMSNorm 2 (Pre-RMSNorm)", 1, 5),
    Node("FFN", "SwiGLU gated feed-forward", 1, 6),
    Node("Xp", "Second residual add: X' = R + F", 1, 7),
    Node("Nf", "Final RMSNorm", 1, 8),
    Node("Z", "Vocabulary projection, logits Z", 1, 9),
    Node("L", "Mean next-token cross-entropy", 1, 10),
]
FWD_EDGES = [
    {"from": "X", "to": "N1"},
    {"from": "N1", "to": "Q"},
    {"from": "N1", "to": "K"},
    {"from": "N1", "to": "V"},
    {"from": "Q", "to": "A", "label": "Q K^T / sqrt(d_h) + M"},
    {"from": "K", "to": "A"},
    {"from": "A", "to": "O"},
    {"from": "V", "to": "O"},
    {"from": "O", "to": "R"},
    {"from": "X", "to": "R", "label": "identity path"},
    {"from": "R", "to": "N2"},
    {"from": "N2", "to": "FFN"},
    {"from": "FFN", "to": "Xp"},
    {"from": "R", "to": "Xp", "label": "identity path"},
    {"from": "Xp", "to": "Nf"},
    {"from": "Nf", "to": "Z"},
    {"from": "Z", "to": "L"},
]

# ------------------------------------------------------ reverse figure ----
# Reverse mode runs loss -> parameters. Column 0/1 carry tensor adjoints;
# column 2 carries the parameter adjoints that each tensor also produces.

BWD_NODES = [
    Node("L", "loss L", 1, 0),
    Node("bZ", "bar Z = (P - Y)/T", 1, 1),
    Node("pWout", "bar W_out", 2, 1, "param", w=150),
    Node("bNf", "bar N_f", 1, 2),
    Node("pgf", "bar g_f  (sum over T)", 2, 2, "param", w=150),
    Node("bXp", "bar X'  = merge", 1, 3, "merge"),
    Node("bF", "bar F", 0, 4),
    Node("pWd", "bar W_d", 2, 4, "param", w=150),
    Node("bH", "bar H  = bar F W_d^T", 0, 5),
    Node("bGU", "bar G, bar U", 0, 6),
    Node("bP", "bar G^pre  (SiLU')", 0, 7),
    Node("pWgu", "bar W_g, bar W_u", 2, 6, "param", w=170),
    Node("bN2", "bar N_2  = merge", 0, 8, "merge"),
    Node("pg2", "bar g_2  (sum over T)", 2, 8, "param", w=150),
    Node("bR", "bar R  = merge", 1, 8, "merge"),
    Node("bY", "bar Y  = bar R", 1, 9),
    Node("pWO", "bar W_O", 2, 9, "param", w=150),
    Node("bO", "bar O  = bar Y W_O^T", 1, 10),
    Node("bAV", "bar A, bar V  = bar O V^T, A^T bar O", 1, 11),
    Node("bs", "bar s  (masked softmax VJP, row sum 0)", 1, 12),
    Node("bQK", "bar Q, bar K  then inverse RoPE", 1, 13),
    Node("bN1", "bar N_1  = merge of Q, K, V", 1, 14, "merge"),
    Node("pWqkv", "bar W_Q, bar W_K, bar W_V", 2, 13, "param", w=180),
    Node("pg1", "bar g_1  (sum over T)", 2, 14, "param", w=150),
    Node("bX", "bar X  = merge", 1, 15, "merge"),
    Node("bE", "bar E  (scatter-add on repeated ids)", 1, 16),
    Node("pE", "bar E  (embedding parameter)", 2, 16, "param", w=190),
]
BWD_EDGES = [
    {"from": "L", "to": "bZ"},
    {"from": "bZ", "to": "pWout", "style": "param", "label": "W̄_out = N_f^T Z̄"},
    {"from": "bZ", "to": "bNf"},
    {"from": "bNf", "to": "pgf", "style": "param", "label": "ḡ_f = Σ_t ȳ_t ⊙ u_t"},
    {"from": "bNf", "to": "bXp"},
    {"from": "bXp", "to": "bF", "label": "transformed path"},
    {"from": "bXp", "to": "pWd", "style": "param", "label": "W̄_d = H^T F̄"},
    {"from": "bF", "to": "bH"},
    {"from": "bH", "to": "bGU"},
    {"from": "bGU", "to": "bP"},
    {"from": "bGU", "to": "pWgu", "style": "param", "label": "W̄_g = N_2^T P̄, W̄_u = N_2^T Ū"},
    {"from": "bP", "to": "bN2"},
    {"from": "bN2", "to": "pg2", "style": "param"},
    {"from": "bN2", "to": "bR", "label": "transformed path", "style": "merge"},
    {"from": "bXp", "to": "bR", "label": "identity path", "style": "merge"},
    {"from": "bR", "to": "bY"},
    {"from": "bY", "to": "pWO", "style": "param", "label": "W̄_O = O^T Ȳ"},
    {"from": "bY", "to": "bO"},
    {"from": "bO", "to": "bAV"},
    {"from": "bAV", "to": "bs"},
    {"from": "bs", "to": "bQK"},
    {"from": "bQK", "to": "pWqkv", "style": "param", "label": "use pre-RoPE adjoints"},
    {"from": "bQK", "to": "bN1", "style": "merge"},
    {"from": "bAV", "to": "bN1", "style": "merge"},
    {"from": "bN1", "to": "pg1", "style": "param"},
    {"from": "bN1", "to": "bX", "style": "merge"},
    {"from": "bY", "to": "bX", "label": "identity path", "style": "merge"},
    {"from": "bX", "to": "bE"},
    {"from": "bE", "to": "pE", "style": "param"},
]

FORWARD_DOT = """// Canonical one-head, one-block forward dataflow.
// Regenerate:  python code/ch06_graphs.py     (or: dot -Tsvg ch06_forward_graph.dot)
digraph ch06_forward {
  rankdir=TB; fontname="Arial"; size="7,10";
  node [shape=box, style="rounded,filled", fillcolor="#eef3f8", fontname="Arial", fontsize=10];
  edge [fontname="Arial", fontsize=9];
  X  [label="Token embeddings, residual X"];
  N1 [label="RMSNorm 1 (Pre-RMSNorm)"];
  Q  [label="Q projection + RoPE"];
  K  [label="K projection + RoPE"];
  V  [label="V projection (unrotated)"];
  A  [label="Scaled scores + causal mask + row softmax"];
  O  [label="O = A V, concat heads, then W_O"];
  R  [label="First residual add: R = X + Y"];
  N2 [label="RMSNorm 2 (Pre-RMSNorm)"];
  FFN[label="SwiGLU gated feed-forward"];
  Xp [label="Second residual add: X' = R + F"];
  Nf [label="Final RMSNorm"];
  Z  [label="Vocabulary projection, logits Z"];
  L  [label="Mean next-token cross-entropy"];
  X -> N1; N1 -> Q; N1 -> K; N1 -> V;
  Q -> A [label="Q K^T / sqrt(d_h) + M"]; K -> A;
  A -> O; V -> O; O -> R;
  X -> R [label="identity path"];
  R -> N2; N2 -> FFN; FFN -> Xp;
  R -> Xp [label="identity path"];
  Xp -> Nf; Nf -> Z; Z -> L;
}
"""

BACKWARD_DOT = """// Authentic reverse pass for the canonical one-head, one-block specialization.
// Dashed edges are parameter adjoints. Green merges are accumulation points.
// Regenerate:  python code/ch06_graphs.py     (or: dot -Tsvg ch06_backward_graph.dot)
digraph ch06_backward {
  rankdir=BT; fontname="Arial";
  node [shape=box, style="rounded,filled", fillcolor="#eef3f8", fontname="Arial", fontsize=10];
  edge [fontname="Arial", fontsize=9];
  L   [label="loss L"];
  bZ  [label="bar Z = (P - Y)/T"];
  bNf [label="bar N_f"];
  bXp [label="bar X'  (merge)", fillcolor="#eaf3ea"];
  bF  [label="bar F"];
  bH  [label="bar H"];
  bGU [label="bar G, bar U"];
  bP  [label="bar G^pre (SiLU')"];
  bN2 [label="bar N_2 (merge)", fillcolor="#eaf3ea"];
  bR  [label="bar R (merge)", fillcolor="#eaf3ea"];
  bY  [label="bar Y"];
  bO  [label="bar O"];
  bAV [label="bar A, bar V"];
  bs  [label="bar s (masked softmax VJP)"];
  bQK [label="bar Q, bar K, then inverse RoPE"];
  bN1 [label="bar N_1 = merge of Q,K,V", fillcolor="#eaf3ea"];
  bX  [label="bar X (merge)", fillcolor="#eaf3ea"];
  bE  [label="bar E (scatter-add)"];
  pWout[label="bar W_out", fillcolor="#f6f0e8", style="rounded,filled,dashed,filled"];
  pgf [label="bar g_f", fillcolor="#f6f0e8"];
  pWd [label="bar W_d", fillcolor="#f6f0e8"];
  pWgu[label="bar W_g, bar W_u", fillcolor="#f6f0e8"];
  pg2 [label="bar g_2", fillcolor="#f6f0e8"];
  pWO [label="bar W_O", fillcolor="#f6f0e8"];
  pWqkv[label="bar W_Q, bar W_K, bar W_V", fillcolor="#f6f0e8"];
  pg1 [label="bar g_1", fillcolor="#f6f0e8"];
  pE  [label="bar E (embedding)", fillcolor="#f6f0e8"];

  L -> bZ;
  bZ -> pWout [style=dashed, label="W_out"];
  bZ -> bNf;
  bNf -> pgf [style=dashed];
  bNf -> bXp;
  bXp -> bF [label="transformed"];
  bXp -> pWd [style=dashed];
  bF -> bH -> bGU -> bP -> bN2;
  bGU -> pWgu [style=dashed];
  bN2 -> pg2 [style=dashed];
  bN2 -> bR [color="#1f6f3f", label="transformed"];
  bXp -> bR [color="#1f6f3f", label="identity"];
  bR -> bY; bY -> pWO [style=dashed]; bY -> bO;
  bO -> bAV -> bs -> bQK;
  bQK -> pWqkv [style=dashed];
  bQK -> bN1 [color="#1f6f3f"];
  bAV -> bN1 [color="#1f6f3f"];
  bN1 -> pg1 [style=dashed];
  bN1 -> bX [color="#1f6f3f"];
  bY  -> bX [color="#1f6f3f", label="identity"];
  bX -> bE -> pE [style=dashed];
}
"""


def main() -> None:
    os.makedirs(FIG, exist_ok=True)

    render(
        FWD_NODES, FWD_EDGES,
        "Canonical decoder block, one head, one block: forward dataflow",
        620, 900, "ch06_forward_graph.dot",
        os.path.join(FIG, "ch06_forward_graph.svg"),
    )
    render(
        BWD_NODES, BWD_EDGES,
        "Reverse pass: adjoint flow, accumulation merges, and parameter gradients",
        900, 1430, "ch06_backward_graph.dot",
        os.path.join(FIG, "ch06_backward_graph.svg"),
    )
    for name, src in (
        ("ch06_forward_graph.dot", FORWARD_DOT),
        ("ch06_backward_graph.dot", BACKWARD_DOT),
    ):
        with open(os.path.join(FIG, name), "w", encoding="utf-8") as f:
            f.write(src)

    print("wrote figures/ch06_forward_graph.svg  (forward dataflow)")
    print("wrote figures/ch06_backward_graph.svg  (reverse pass + parameter adjoints)")
    print("wrote figures/ch06_forward_graph.dot")
    print("wrote figures/ch06_backward_graph.dot")


if __name__ == "__main__":
    main()
