"""Presentation primitives for the standalone HTML reports: CSS, SVG charts, shell.

Knows nothing about any pipeline — it takes numbers and returns HTML strings, so
the DAD report and (later) the SDF report can share one look.

The CSS and the ``esc``/``hbar``/``histogram``/``stat``/``table`` primitives are
ported from ``evals/report_sdf.py`` on the unmerged branch
``origin/aidan/sdf-500-run-and-report``; that file is deliberately left untouched
so the branch still merges cleanly, and rewiring it onto this module is a
follow-up once it lands. Everything else here (grouped/stacked bars, scatter,
side-by-side, highlight) is new — the DAD report compares two arms, which the SDF
report never had to.

Output is ONE self-contained file: no external CSS, JS, fonts, or images. An
artifact host's CSP blocks every external origin, and the file has to survive
being downloaded and opened offline. Charts are therefore inline <svg> generated
here rather than a charting library, and the only JS is a tooltip handler.

stdlib only, and no repo imports: the report generator must run anywhere,
including where the pipeline's own dependencies are not installed.
"""

import re

# Series colors are CSS custom properties, never literal hexes: each has a light
# and a dark value declared in CSS, so a chart themes itself. (The viewer's
# AUDIT_ARM_COLORS hexes are tuned for Streamlit's single theme and would fail
# dark mode here.)
PAL = [f"var(--series-{i})" for i in range(1, 9)]

# The two arms, everywhere. Plain = warm/orange family, pipeline = green family,
# matching the viewer's terracotta/green pairing without inheriting its hexes.
PLAIN = "var(--series-2)"
PIPELINE = "var(--series-3)"
ARM_COLORS = {"plain": PLAIN, "plain Claude": PLAIN, "pipeline": PIPELINE}


class Raw(str):
    """HTML that is already built and must not be escaped again.

    ``table()`` escapes every cell by default — the SDF generator escapes at each
    call site instead, and one missed call there is a mangled cell or an
    injection. Wrap pre-built markup in ``Raw`` to opt out.
    """


def esc(s):
    if isinstance(s, Raw):
        return str(s)
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


_MD_CODE = re.compile(r"`([^`]+)`")
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITAL = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def inline_md(text):
    """Escape, then apply a bold/italic/code/link subset of markdown.

    Used on prose only — editorial copy and LLM-written judge notes, which
    contain ``**bold**``. NEVER used on corpus text, which must render verbatim.
    """
    out = esc(text)
    out = _MD_CODE.sub(r"<code>\1</code>", out)
    out = _MD_BOLD.sub(r"<b>\1</b>", out)
    out = _MD_ITAL.sub(r"<i>\1</i>", out)
    out = _MD_LINK.sub(r"<a href='\2'>\1</a>", out)
    return out


def paragraphs(text):
    """Blank-line-separated prose to <p>/<ul> blocks, with inline markdown.

    A block whose lines all start with "- " becomes a list; everything else is a
    paragraph. That is the whole of the markdown this report needs.
    """
    blocks = []
    for block in re.split(r"\n\s*\n", (text or "").strip()):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        if all(ln.startswith("- ") for ln in lines):
            items = "".join(f"<li>{inline_md(ln[2:])}</li>" for ln in lines)
            blocks.append(f"<ul>{items}</ul>")
        elif lines[0].startswith("### "):
            head = inline_md(lines[0][4:])
            rest = " ".join(lines[1:])
            blocks.append(f"<h3>{head}</h3>" + (f"<p>{inline_md(rest)}</p>" if rest else ""))
        else:
            blocks.append(f"<p>{inline_md(' '.join(lines))}</p>")
    return "".join(blocks)


def chip(text, tone=""):
    return f"<span class='chip {tone}'>{esc(text)}</span>"


def note(text, tone="warn"):
    """A called-out caveat line. The report's candour depends on these being
    visually unmissable rather than gray small print."""
    return f"<p class='{tone}-note'>{inline_md(text)}</p>"


def details(summary, body, open_=False):
    """A layered-detail block. <details> needs no JS and prints expanded."""
    return (f"<details{' open' if open_ else ''}><summary>{esc(summary)}</summary>"
            f"<div class='det-body'>{body}</div></details>")


def stat(value, label, sub="", tone=""):
    cls = f"tile {tone}".strip()
    return (f"<div class='{cls}'><div class='tile-v'>{esc(value)}</div>"
            f"<div class='tile-l'>{esc(label)}</div>"
            + (f"<div class='tile-s'>{esc(sub)}</div>" if sub else "") + "</div>")


def tiles(items):
    kept = [t for t in items if t]
    return f"<div class='tiles'>{''.join(kept)}</div>" if kept else ""


def table(headers, rows, cls=""):
    """Cells are escaped; wrap pre-built markup in Raw() to pass it through."""
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (f"<div class='scroll'><table class='{cls}'><thead><tr>{th}</tr></thead>"
            f"<tbody>{trs}</tbody></table></div>")


def _no_data(msg="not measured on this run"):
    return f"<p class='muted'>{esc(msg)}</p>"


def hbar(pairs, *, unit="", width=760, row=26, color=None, maxval=None, fmt="{:g}"):
    """Horizontal bars: magnitude by identity. Labels outside, value at bar end."""
    if not pairs:
        return _no_data()
    label_w, pad = 300, 60
    mx = maxval or max((v for _, v in pairs), default=0) or 1
    bar_w = width - label_w - pad
    h = row * len(pairs) + 8
    out = [f"<svg viewBox='0 0 {width} {h}' role='img' class='chart'>"]
    for i, (lab, val) in enumerate(pairs):
        y = i * row + 4
        w = max(2, bar_w * val / mx)
        fill = color or PAL[i % 8]
        shown = fmt.format(val) + unit
        out.append(
            f"<text x='{label_w - 8}' y='{y + 14}' class='lab' text-anchor='end'>"
            f"{esc(str(lab)[:46])}</text>"
            f"<rect x='{label_w}' y='{y + 3}' width='{w:.1f}' height='14' rx='4' fill='{fill}'"
            f" data-tip='{esc(lab)}: {esc(shown)}'/>"
            f"<text x='{label_w + w + 6}' y='{y + 14}' class='val'>{esc(shown)}</text>")
    out.append("</svg>")
    return "".join(out)


def grouped_hbar(rows, *, series, width=760, group_gap=12, bar_h=13, percent=False,
                 rule=None, rule_label="", label_w=250, fmt="{:g}"):
    """One group of bars per category, one bar per series — the plain-vs-pipeline
    workhorse.

    rows: [{"label": str, <series name>: value, ...}]
    series: [(name, color)] in draw order.
    rule: optional dashed threshold line (a share, when percent=True).
    """
    rows = [r for r in rows if any(r.get(s) is not None for s, _ in series)]
    if not rows:
        return _no_data()
    pad = 64
    bar_w = width - label_w - pad
    mx = 1.0 if percent else (max(
        (r.get(s) or 0) for r in rows for s, _ in series) or 1)
    grp_h = bar_h * len(series) + group_gap
    h = grp_h * len(rows) + 20
    out = [f"<svg viewBox='0 0 {width} {h}' role='img' class='chart'>"]
    for i, r in enumerate(rows):
        top = i * grp_h + 6
        out.append(f"<text x='{label_w - 8}' y='{top + grp_h / 2 - 2:.0f}' class='lab' "
                   f"text-anchor='end'>{esc(str(r['label'])[:40])}</text>")
        for j, (name, color) in enumerate(series):
            val = r.get(name)
            if val is None:
                continue
            y = top + j * bar_h
            w = max(1.5, bar_w * val / mx)
            shown = f"{val:.0%}" if percent else fmt.format(val)
            out.append(
                f"<rect x='{label_w}' y='{y}' width='{w:.1f}' height='{bar_h - 3}' rx='3' "
                f"fill='{color}' data-tip='{esc(r['label'])} — {esc(name)}: {esc(shown)}'/>"
                f"<text x='{label_w + w + 5}' y='{y + bar_h - 5}' class='val'>{esc(shown)}</text>")
    if rule is not None:
        x = label_w + bar_w * rule / mx
        out.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='0' y2='{h - 18}' class='rule'/>"
                   f"<text x='{x + 5:.1f}' y='{h - 6}' class='muted-svg'>{esc(rule_label)}</text>")
    out.append("</svg>")
    return "".join(out) + _legend(series)


def _legend(series):
    keys = "".join(f"<span class='key'><i style='background:{c}'></i>{esc(n)}</span>"
                   for n, c in series)
    return f"<div class='legend'>{keys}</div>"


def stacked_bar(rows, *, categories, width=760, height=260, xlabel="", ylabel=""):
    """One stacked column per record — the kept/weakened/dropped/added view.

    rows: [{"label": str, "segments": {category: count}, "tips": {category: str}}]
    categories: [(name, color)] bottom-to-top.
    """
    rows = [r for r in rows if r.get("segments")]
    if not rows:
        return _no_data()
    left, bottom, top = 40, 34, 10
    totals = [sum((r["segments"].get(c) or 0) for c, _ in categories) for r in rows]
    mx = max(totals) or 1
    plot_h = height - bottom - top
    bw = (width - left - 12) / len(rows)
    out = [f"<svg viewBox='0 0 {width} {height}' role='img' class='chart'>"]
    for gy in (0, 0.5, 1.0):
        y = top + plot_h * (1 - gy)
        out.append(f"<line x1='{left}' x2='{width - 12}' y1='{y:.1f}' y2='{y:.1f}' class='grid'/>"
                   f"<text x='{left - 6}' y='{y + 4:.1f}' class='val' text-anchor='end'>"
                   f"{int(mx * gy)}</text>")
    for i, r in enumerate(rows):
        x = left + i * bw + bw * 0.14
        w = bw * 0.72
        y_cursor = top + plot_h
        for cat, color in categories:
            val = r["segments"].get(cat) or 0
            if not val:
                continue
            seg_h = plot_h * val / mx
            y_cursor -= seg_h
            tip = (r.get("tips") or {}).get(cat) or f"{r['label']} — {cat}: {val}"
            out.append(f"<rect x='{x:.1f}' y='{y_cursor:.1f}' width='{w:.1f}' "
                       f"height='{seg_h:.1f}' fill='{color}' data-tip='{esc(tip)}'/>")
        if len(rows) <= 24:
            out.append(f"<text x='{x + w / 2:.1f}' y='{height - 20}' class='val' "
                       f"text-anchor='middle' transform='rotate(-40 {x + w / 2:.1f} "
                       f"{height - 20})'>{esc(str(r['label'])[-6:])}</text>")
    if xlabel:
        out.append(f"<text x='{width / 2:.0f}' y='{height - 2}' class='muted-svg' "
                   f"text-anchor='middle'>{esc(xlabel)}</text>")
    if ylabel:
        out.append(f"<text x='4' y='{top - 1}' class='muted-svg'>{esc(ylabel)}</text>")
    out.append("</svg>")
    return "".join(out) + _legend(categories)


def scatter(points, *, xlabel="", ylabel="", xdomain=None, ydomain=None,
            marks=(), width=760, height=340):
    """points/marks: [{"x","y","color","tip","shape"?}]. marks draw larger and
    outlined (the per-arm means the dots scatter around)."""
    pts = [p for p in points if p.get("x") is not None and p.get("y") is not None]
    if not pts:
        return _no_data()
    left, right, top, bottom = 46, 16, 14, 40
    xs = [p["x"] for p in pts] + [m["x"] for m in marks]
    ys = [p["y"] for p in pts] + [m["y"] for m in marks]
    x0, x1 = xdomain or (min(xs), max(xs))
    y0, y1 = ydomain or (0, max(ys) * 1.12 or 1)
    x1 = x1 if x1 > x0 else x0 + 1
    y1 = y1 if y1 > y0 else y0 + 1
    pw, ph = width - left - right, height - top - bottom

    def px(x):
        return left + pw * (x - x0) / (x1 - x0)

    def py(y):
        return top + ph * (1 - (y - y0) / (y1 - y0))

    out = [f"<svg viewBox='0 0 {width} {height}' role='img' class='chart'>"]
    for k in range(5):
        gy = y0 + (y1 - y0) * k / 4
        out.append(f"<line x1='{left}' x2='{width - right}' y1='{py(gy):.1f}' "
                   f"y2='{py(gy):.1f}' class='grid'/>"
                   f"<text x='{left - 6}' y='{py(gy) + 4:.1f}' class='val' "
                   f"text-anchor='end'>{gy:.0f}</text>")
    for k in range(6):
        gx = x0 + (x1 - x0) * k / 5
        out.append(f"<text x='{px(gx):.1f}' y='{height - 22}' class='val' "
                   f"text-anchor='middle'>{gx:.0f}</text>")
    for p in pts:
        out.append(f"<circle cx='{px(p['x']):.1f}' cy='{py(p['y']):.1f}' r='5' "
                   f"fill='{p.get('color', PAL[0])}' opacity='.62' "
                   f"data-tip='{esc(p.get('tip', ''))}'/>")
    for m in marks:
        out.append(f"<rect x='{px(m['x']) - 7:.1f}' y='{py(m['y']) - 7:.1f}' width='14' "
                   f"height='14' transform='rotate(45 {px(m['x']):.1f} {py(m['y']):.1f})' "
                   f"fill='{m.get('color', PAL[0])}' stroke='var(--surface-0)' "
                   f"stroke-width='2' data-tip='{esc(m.get('tip', ''))}'/>")
    if xlabel:
        out.append(f"<text x='{left + pw / 2:.0f}' y='{height - 4}' class='muted-svg' "
                   f"text-anchor='middle'>{esc(xlabel)}</text>")
    if ylabel:
        out.append(f"<text x='4' y='{top - 2}' class='muted-svg'>{esc(ylabel)}</text>")
    out.append("</svg>")
    return "".join(out)


def highlight(text, spans):
    """Escaped text with each verbatim span wrapped in <mark>.

    Fail-open, matching the viewer: spans were substring-validated at audit time,
    so a span that no longer locates (or straddles our escaping) renders
    unhighlighted rather than corrupting the text.
    """
    out = esc(text)
    for span in spans or []:
        if not span:
            continue
        marked = esc(span)
        if marked in out:
            out = out.replace(marked, f"<mark>{marked}</mark>", 1)
    return f"<div class='resp'>{out}</div>"


def sidebyside(left_title, left_html, right_title, right_html):
    return (f"<div class='pair'>"
            f"<div class='pane'><h4>{esc(left_title)}</h4>{left_html}</div>"
            f"<div class='pane'><h4>{esc(right_title)}</h4>{right_html}</div></div>")


def quote(text):
    return f"<blockquote>{esc(text)}</blockquote>"


CSS = """
:root{color-scheme:light;--surface-0:#ffffff;--surface-1:#fcfcfb;--surface-2:#f4f3ef;
--border:#e2e0d8;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#7a7973;
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;
--series-5:#e87ba4;--series-6:#008300;--series-7:#4a3aa7;--series-8:#e34948;
--good:#1baf7a;--warn:#eda100;--bad:#e34948;--grid:#e8e6de;--mark:rgba(255,212,90,.55)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){color-scheme:dark;
--surface-0:#121211;--surface-1:#1a1a19;--surface-2:#242422;--border:#34342f;
--text-primary:#ffffff;--text-secondary:#c3c2b7;--text-muted:#96958c;
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;
--series-5:#d55181;--series-6:#008300;--series-7:#9085e9;--series-8:#e66767;
--good:#199e70;--warn:#c98500;--bad:#e66767;--grid:#2e2e2a;--mark:rgba(201,133,0,.42)}}
:root[data-theme=dark]{color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;
--surface-2:#242422;--border:#34342f;--text-primary:#ffffff;--text-secondary:#c3c2b7;
--text-muted:#96958c;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;
--series-4:#c98500;--series-5:#d55181;--series-6:#008300;--series-7:#9085e9;
--series-8:#e66767;--good:#199e70;--warn:#c98500;--bad:#e66767;--grid:#2e2e2a;
--mark:rgba(201,133,0,.42)}
*{box-sizing:border-box}
html{--serif:ui-serif,Charter,"Bitstream Charter","Iowan Old Style","Source Serif Pro",Georgia,serif;
--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
font:17px/1.65 var(--serif);-webkit-text-size-adjust:100%;font-variant-numeric:tabular-nums}
.wrap{max-width:940px;margin:0 auto;padding:40px 24px 90px}
header.top{border-bottom:2px solid var(--text-primary);padding-bottom:20px;margin-bottom:8px}
h1{font-size:2.05rem;line-height:1.18;margin:0 0 10px;letter-spacing:-.012em;text-wrap:balance}
h2{font-family:var(--sans);font-size:1.16rem;font-weight:650;letter-spacing:-.005em;
margin:56px 0 6px;padding-top:22px;border-top:1px solid var(--border);text-wrap:balance}
section#summary h2{border-top:0;padding-top:0;margin-top:28px}
h3{font-family:var(--sans);font-size:1rem;font-weight:650;margin:34px 0 4px;text-wrap:balance}
h4{font-family:var(--sans);font-size:.82rem;font-weight:650;margin:22px 0 6px;
color:var(--text-secondary);text-transform:uppercase;letter-spacing:.055em}
p{margin:11px 0;color:var(--text-secondary);max-width:66ch}
ul{color:var(--text-secondary);max-width:66ch;padding-left:20px}li{margin:5px 0}
.sub{font-family:var(--sans);color:var(--text-muted);font-size:.88rem;margin:0;
max-width:74ch;line-height:1.5}
.eyebrow{display:block;font-family:var(--sans);font-size:.66rem;font-weight:650;
text-transform:uppercase;letter-spacing:.09em;color:var(--text-muted);margin-bottom:3px}
.muted{color:var(--text-muted);font-size:.84rem;font-family:var(--sans)}
.mono{font-family:var(--mono);font-size:.82em}
nav.toc{font-family:var(--sans);font-size:.83rem;display:flex;flex-wrap:wrap;gap:4px 18px;
padding:12px 0 0;margin-bottom:4px}
nav.toc a{color:var(--text-secondary);text-decoration:none;border-bottom:1px solid transparent}
nav.toc a:hover,nav.toc a:focus-visible{color:var(--text-primary);border-bottom-color:var(--series-1)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(196px,1fr));gap:1px;
margin:18px 0;background:var(--border);border:1px solid var(--border)}
.tile{background:var(--surface-1);padding:15px 17px}
.tile-v{font-family:var(--sans);font-size:1.72rem;font-weight:660;letter-spacing:-.025em;
line-height:1.08}
.tile-l{font-family:var(--sans);font-size:.84rem;color:var(--text-secondary);margin-top:5px;
line-height:1.35}
.tile-s{font-family:var(--sans);font-size:.74rem;color:var(--text-muted);margin-top:5px;
line-height:1.4}
.tile.good .tile-v{color:var(--good)}.tile.warn .tile-v{color:var(--warn)}
.tile.bad .tile-v{color:var(--bad)}
.chart{width:100%;height:auto;overflow:visible;display:block;margin:8px 0}
.lab,.val,.muted-svg{font-family:var(--sans)}
.lab{font-size:11px;fill:var(--text-secondary)}
.val{font-size:11px;fill:var(--text-muted)}
.muted-svg{font-size:11px;fill:var(--text-muted)}
.grid{stroke:var(--grid);stroke-width:1}
.rule{stroke:var(--warn);stroke-width:1.5;stroke-dasharray:5 3}
.legend{font-family:var(--sans);font-size:.76rem;color:var(--text-secondary);display:flex;
gap:16px;flex-wrap:wrap;margin:2px 0 14px}
.legend .key{display:inline-flex;align-items:center;gap:6px}
.legend i{width:11px;height:11px;border-radius:2px;display:inline-block}
.scroll{overflow-x:auto;margin:12px 0;border:1px solid var(--border)}
table{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:.82rem;
font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 11px;border-bottom:1px solid var(--border);vertical-align:top}
tr:last-child td{border-bottom:0}
th{color:var(--text-muted);font-weight:650;font-size:.7rem;text-transform:uppercase;
letter-spacing:.07em;background:var(--surface-1)}
code{font-family:var(--mono);background:var(--surface-2);padding:1px 5px;font-size:.82em}
pre{font-family:var(--mono);background:var(--surface-2);border:1px solid var(--border);
padding:13px 15px;overflow-x:auto;font-size:.78rem;line-height:1.6;color:var(--text-primary)}
.chip{font-family:var(--sans);font-size:.66rem;font-weight:700;text-transform:uppercase;
letter-spacing:.085em;padding:3px 7px;background:var(--surface-2);color:var(--text-secondary)}
.chip.good{background:var(--good);color:var(--surface-0)}
.chip.bad{background:var(--bad);color:var(--surface-0)}
.chip.warn{background:var(--warn);color:var(--surface-0)}
blockquote{margin:12px 0;white-space:pre-wrap;font-size:1.02rem;line-height:1.5;
color:var(--text-primary);padding-left:16px;border-left:2px solid var(--series-1)}
.warn-note{color:var(--text-primary);border-left:3px solid var(--warn);padding:2px 0 2px 14px;
font-size:.92rem;background:var(--surface-1);max-width:none}
.bad-note{color:var(--text-primary);border-left:3px solid var(--bad);padding:2px 0 2px 14px;
font-size:.92rem;background:var(--surface-1);max-width:none}
details{border-top:1px solid var(--border);margin:14px 0;padding-top:10px}
summary{font-family:var(--sans);font-size:.85rem;font-weight:650;cursor:pointer;
color:var(--text-secondary)}
summary:hover{color:var(--text-primary)}
.det-body{padding-top:6px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:14px 0}
.pane{min-width:0}
.resp{white-space:pre-wrap;font-size:.94rem;line-height:1.6;color:var(--text-primary);
border-left:2px solid var(--border);padding-left:14px}
mark{background:var(--mark);color:inherit;padding:0 2px;border-radius:2px}
a{color:var(--series-1)}
a:focus-visible,[tabindex]:focus-visible,summary:focus-visible{outline:2px solid var(--series-1);
outline-offset:2px}
#tip{position:fixed;pointer-events:none;opacity:0;background:var(--text-primary);
color:var(--surface-0);font-family:var(--sans);font-size:12px;padding:5px 8px;
transition:opacity .1s;z-index:9;max-width:340px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
@media (max-width:700px){.pair{grid-template-columns:1fr}}
@media (max-width:600px){body{font-size:16px}.wrap{padding:28px 16px 60px}h1{font-size:1.6rem}}
@media print{details{display:block}details>div{display:block!important}nav.toc{display:none}}
"""

JS = """
(function(){var t=document.getElementById('tip');
document.addEventListener('mouseover',function(e){var el=e.target.closest('[data-tip]');
if(!el){t.style.opacity=0;return;}t.textContent=el.getAttribute('data-tip');t.style.opacity=1;});
document.addEventListener('mousemove',function(e){if(t.style.opacity=='1'){
t.style.left=Math.min(e.clientX+12,window.innerWidth-t.offsetWidth-8)+'px';
t.style.top=(e.clientY-32)+'px';}});})();
"""


def document(*, title, subtitle, meta_line, toc, body, eyebrow="Synthetic training data"):
    """The shell. `meta_line` is pre-built HTML; everything else is escaped."""
    nav = "".join(f"<a href='#{i}'>{esc(l)}</a>" for i, l in toc)
    return (f"<!DOCTYPE html>\n<html lang='en'>\n<meta charset='utf-8'>\n"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>\n"
            f"<title>{esc(title)}</title>\n<style>{CSS}</style>\n"
            f"<div class='wrap'>\n<header class='top'>\n"
            f"<span class='eyebrow'>{esc(eyebrow)}</span>\n"
            f"<h1>{esc(title)}</h1>\n"
            f"<p class='sub'>{inline_md(subtitle)}</p>\n"
            f"<p class='sub' style='margin-top:8px'>{meta_line}</p>\n</header>\n"
            f"<nav class='toc' aria-label='Sections'>{nav}</nav>\n{body}\n</div>\n"
            f"<div id='tip'></div>\n<script>{JS}</script>\n</html>\n")
