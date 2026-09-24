"""Look and feel of the Studio: theme, CSS and small HTML pieces. Spec 2026-09-23-studio section 8.
Chip text always says what it means, so colour is never the only signal; every pair meets 4.5:1 contrast."""
from __future__ import annotations

import html

import gradio as gr

THEME = gr.themes.Base(primary_hue="indigo", secondary_hue="amber", neutral_hue="slate",
                       radius_size=gr.themes.sizes.radius_md, spacing_size=gr.themes.sizes.spacing_md).set(
    body_background_fill="*neutral_50", body_background_fill_dark="*neutral_950",
    block_border_width="1px", block_shadow="none", block_radius="*radius_lg",
    button_primary_background_fill="*primary_600", button_primary_background_fill_hover="*primary_700",
)
TONES = {"ok": ("#065F46", "#D1FAE5"), "check": ("#92400E", "#FEF3C7"), "ai": ("#5B21B6", "#EDE9FE"),
         "stop": ("#991B1B", "#FEE2E2"), "info": ("#334155", "#E2E8F0")}
TRUSTED = frozenset({"user_written", "measured", "user_edited"})
SOURCE_TEXT = {"user_written": "written", "measured": "measured", "user_edited": "you set it",
               "scaled": "scaled · check", "inferred": "AI · check", "estimated": "estimated · check",
               "default": "default · check"}
FACE_BADGES = {"observed": ("observed", "ok"), "mirrored": ("mirrored", "ok"),
               "qwen-image": ("drawn by Qwen-Image", "ai"), "triposr": ("predicted by TripoSR", "ai"),
               "assumed": ("assumed rectangle", "check")}
CSS = """
.studio-header {display:flex; flex-wrap:wrap; align-items:center; gap:12px; justify-content:space-between}
.studio-header h1 {margin:0; font-size:1.6rem}
.studio-header p {margin:2px 0 0; opacity:.8}
.chip {display:inline-block; padding:2px 10px; margin:2px 4px 2px 0; border-radius:999px; font-size:.82rem;
       font-weight:600; white-space:nowrap}
.card {border-radius:12px; padding:12px 16px; border:1px solid; margin:4px 0}
.card h4 {margin:0 0 4px}
.card h4, .card div, .card ul, .card li {color:inherit !important}
.stats {display:grid; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); gap:8px}
.stat {border:1px solid var(--border-color-primary); border-radius:12px; padding:8px 12px}
.stat b {display:block; font-size:1.15rem}
.size.required textarea, .size.required input {border-color:#DC2626 !important}
.size.required label span {color:#991B1B}
.dark .size.required label span {color:#FCA5A5}
@media (max-width: 640px) {
    label span {white-space:normal; overflow-wrap:anywhere}
}
"""


def chip(text: str, tone: str = "info") -> str:
    fg, bg = TONES[tone]
    return f'<span class="chip" style="color:{fg};background:{bg}">{html.escape(text)}</span>'


def source_chip(provenance: str) -> str:
    tone = "ok" if provenance in TRUSTED else "ai" if provenance == "inferred" else "check"
    return chip(SOURCE_TEXT.get(provenance, provenance), tone)


def card(title: str, body: str, tone: str = "info") -> str:
    fg, bg = TONES[tone]
    return (f'<div class="card" style="border-color:{fg};background:{bg};color:{fg}">'
            f"<h4>{html.escape(title)}</h4><div>{html.escape(body)}</div></div>")


def stats_html(items: list[tuple[str, str]]) -> str:
    cells = "".join(f'<div class="stat">{html.escape(k)}<b>{html.escape(v)}</b></div>' for k, v in items)
    return f'<div class="stats">{cells}</div>'


def bullet_html(title: str, lines: list[str], tone: str) -> str:
    if not lines:
        return ""
    items = "".join(f"<li>{html.escape(line)}</li>" for line in lines)
    fg, bg = TONES[tone]
    return (f'<div class="card" style="border-color:{fg};background:{bg};color:{fg}">'
            f"<h4>{title}</h4><ul>{items}</ul></div>")
