"""Presentation primitives for the standalone HTML reports: CSS, SVG charts, shell.

Knows nothing about any pipeline — it takes numbers and returns HTML strings, so the
hub, the DAD report and (later) the SDF report share one look.

ONE THEME, LIGHT, deliberately. This is a document that gets handed to an external
reader, printed, and screenshotted into slides. A viewer's OS preference is not a
signal about how a published artefact should look, and the automatic dark flip of a
page whose whole visual argument is warm paper and ink produced a page its author had
never reviewed. ``color-scheme:only light`` (not bare ``light``) is what opts the page
out of Chrome-Android and Samsung Internet's auto-darkening, which is a separate
mechanism from ``prefers-color-scheme``.

Output is ONE self-contained file: no external CSS, JS, fonts, or images. An artifact
host's CSP blocks every external origin, and the file has to survive being downloaded
and opened offline. Charts are therefore inline <svg> generated here rather than a
charting library, and the only JS is a tooltip and a scroll-spy.

stdlib only, and no repo imports: the report generator must run anywhere, including
where the pipeline's own dependencies are not installed.
"""

import re

# Series colors stay CSS custom properties rather than literal hexes. With one theme
# the original reason (a light and a dark value per slot) is gone; four live ones are
# not: there are ~40 fill sites, so retuning a hue for the paper surface is one line
# instead of forty; naming the roles is what makes "a series hue must never mean
# good" testable; --surface-0 is used INSIDE the svg for segment gaps and mark rings,
# so it has to track the surface; and @media print neutralizes every tinted surface in
# one block.
PAL = [f"var(--series-{i})" for i in range(1, 9)]

# The two arms, everywhere. Plain = warm/terracotta, pipeline = green.
PLAIN = "var(--series-2)"
PIPELINE = "var(--series-3)"
ARM_COLORS = {"plain": PLAIN, "plain Claude": PLAIN, "pipeline": PIPELINE}
# Pass this as hbar(color=...) for any (control, pipeline) chart. Without it hbar
# falls back to PAL[i], which colors bars by ROW ORDER — so the headline chart used
# to paint the pipeline in the control's own color.
ARM_PAIR = (PLAIN, PIPELINE)


class Raw(str):
    """HTML that is already built and must not be escaped again.

    ``table()`` escapes every cell by default — wrap pre-built markup in ``Raw`` to
    opt out.
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

    Used on prose only — editorial copy and LLM-written judge notes, which contain
    ``**bold**``. NEVER used on corpus text, which must render verbatim.
    """
    out = esc(text)
    out = _MD_CODE.sub(r"<code>\1</code>", out)
    out = _MD_BOLD.sub(r"<b>\1</b>", out)
    out = _MD_ITAL.sub(r"<i>\1</i>", out)
    out = _MD_LINK.sub(r"<a href='\2'>\1</a>", out)
    return out


def paragraphs(text):
    """Blank-line-separated prose to <p>/<ul>/<h3>/dek blocks, with inline markdown.

    Conventions, all of which the prose files use: a block whose lines all start with
    ``- `` is a list; a block opening ``### `` is a subhead; a block opening ``> `` is
    a dek — the one-line finding that sits under a heading.
    """
    blocks = []
    for block in re.split(r"\n\s*\n", (text or "").strip()):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        if all(ln.startswith("- ") for ln in lines):
            items = "".join(f"<li>{inline_md(ln[2:])}</li>" for ln in lines)
            blocks.append(f"<ul>{items}</ul>")
        elif lines[0].startswith("> "):
            blocks.append(dek(" ".join(ln.lstrip("> ") for ln in lines)))
        elif lines[0].startswith("### "):
            head = inline_md(lines[0][4:])
            rest = " ".join(lines[1:])
            blocks.append(f"<h3>{head}</h3>" + (f"<p>{inline_md(rest)}</p>" if rest else ""))
        else:
            blocks.append(f"<p>{inline_md(' '.join(lines))}</p>")
    return "".join(blocks)


def dek(text):
    """The one line under a heading. It states the finding, not the topic."""
    return f"<p class='dek'>{inline_md(text)}</p>"


def chip(text, tone=""):
    return f"<span class='chip{' ' + tone if tone else ''}'>{esc(text)}</span>"


def note(text, tone="warn"):
    """A called-out caveat. The report's candour depends on these being visually
    unmissable rather than gray small print."""
    return f"<p class='{tone}-note'>{inline_md(text)}</p>"


def details(summary, body, meta="", open_=False):
    """A drawer. ``meta`` names the payload's size, so collapsing costs nothing:
    "Full pipeline answer · 1,010 words". <details> needs no JS and prints open."""
    label = esc(summary) + (f" <span class='sum-m'>{esc(meta)}</span>" if meta else "")
    return (f"<details{' open' if open_ else ''}><summary>{label}</summary>"
            f"<div class='det-body'>{body}</div></details>")


def stat(value, label, sub="", flag="", tone=""):
    """A number. Direction is carried by a labelled chip, never by coloring the
    numeral: a status color must not travel alone."""
    cls = "tile hero" if tone == "hero" else "tile"
    return (f"<div class='{cls}'><div class='tile-v'>{esc(value)}</div>"
            f"<div class='tile-l'>{esc(label)}</div>"
            + (f"<div class='tile-s'>{esc(sub)}</div>" if sub else "")
            + (f"<div class='tile-f'>{chip(flag, tone if tone != 'hero' else '')}</div>"
               if flag else "") + "</div>")


def tiles(items):
    kept = [t for t in items if t]
    if not kept:
        return ""
    if sum(1 for t in kept if "tile hero" in t) > 1:
        raise ValueError("a tile row may have at most one hero tile")
    return f"<div class='tiles'>{''.join(kept)}</div>"


def table(headers, rows, cls="", align=""):
    """Cells are escaped; wrap pre-built markup in Raw() to pass it through.

    ``align`` is one character per column — l/r/c. Numeric columns should be r, so
    magnitudes line up and a delta reads down a single column.
    """
    def cell(tag, i, v):
        a = align[i] if i < len(align) else "l"
        klass = f" class='{ {'r': 'num', 'c': 'ctr'}[a] }'" if a in "rc" else ""
        return f"<{tag}{klass}>{esc(v)}</{tag}>"

    th = "".join(cell("th", i, h) for i, h in enumerate(headers))
    trs = "".join("<tr>" + "".join(cell("td", i, c) for i, c in enumerate(r)) + "</tr>"
                  for r in rows)
    return (f"<div class='scroll'><table class='{cls}'><thead><tr>{th}</tr></thead>"
            f"<tbody>{trs}</tbody></table></div>")


_SVG_OPEN = re.compile(r"(<svg\b[^>]*>)")


def figure(*, title, chart, caption="", note_="", table_html=None, table_label="Show the numbers"):
    """A chart with its title, caption and optional table view, as one unit.

    The title is a <figcaption>, not a heading: chart titles were polluting the
    document outline and the section list. The caption states the FINDING; axis
    descriptions go in ``note_``, above the chart, where they are read before it
    rather than after. The table view is the relief a chart with a sub-3:1 series
    needs, and the only way a touch user reaches the tooltip's numbers.
    """
    named = _SVG_OPEN.sub(lambda m: m.group(1) + f"<title>{esc(title)}</title>", chart, count=1)
    return ("<figure>"
            f"<figcaption class='fig-t'>{esc(title)}</figcaption>"
            + (f"<p class='fig-n'>{inline_md(note_)}</p>" if note_ else "")
            + named
            + (f"<figcaption class='fig-c'>{inline_md(caption)}</figcaption>" if caption else "")
            + (details(table_label, table_html) if table_html else "")
            + "</figure>")


def _no_data(msg="not measured on this run"):
    return f"<p class='muted'>{esc(msg)}</p>"


W = 800  # every chart is drawn at the figure track's own width, so an 11px label is
         # 11px in every figure instead of scaling with the column it lands in.


def _bar(x, y, w, h, fill, tip, r=3):
    """A bar rounded on the value end only. Rounding the baseline end too made every
    bar look like a lozenge floating free of its axis."""
    r = max(0.0, min(r, w / 2, h / 2))
    d = (f"M{x:.1f},{y:.1f} H{x + w - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + w:.1f},{y + r:.1f} "
         f"V{y + h - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + w - r:.1f},{y + h:.1f} "
         f"H{x:.1f} Z")
    return f"<path d='{d}' fill='{fill}' data-tip='{esc(tip)}'/>"


def hbar(pairs, *, unit="", width=W, row=28, color=None, maxval=None, fmt="{:g}", label_w=240):
    """Horizontal bars: magnitude by identity. Labels outside, value at the bar end.

    ``color`` takes a single color for every bar OR a sequence indexed by row — pass
    ARM_PAIR for a (control, pipeline) chart so the color follows the arm.
    """
    if not pairs:
        return _no_data()
    pad = 72
    mx = maxval or max((v for _, v in pairs), default=0) or 1
    bar_w = width - label_w - pad
    h = row * len(pairs) + 6
    out = [f"<svg viewBox='0 0 {width} {h}' role='img' class='chart'>"]
    for i, (lab, val) in enumerate(pairs):
        y = i * row + 4
        w = max(2, bar_w * val / mx)
        fill = (color[i % len(color)] if isinstance(color, (list, tuple)) else color) or PAL[0]
        shown = fmt.format(val) + unit
        out.append(
            f"<text x='{label_w - 10}' y='{y + 14}' class='lab' text-anchor='end'>"
            f"{esc(str(lab)[:46])}</text>"
            + _bar(label_w, y + 3, w, 15, fill, f"{lab}: {shown}")
            + f"<text x='{label_w + w + 7:.1f}' y='{y + 15}' class='val strong'>"
              f"{esc(shown)}</text>")
    out.append("</svg>")
    return "".join(out)


def grouped_hbar(rows, *, series, width=W, group_gap=13, bar_h=13, percent=False,
                 rule=None, rule_label="", label_w=250, fmt="{:g}", direct_labels=True,
                 glossary=None):
    """One group of bars per category, one bar per series — the control-vs-pipeline
    workhorse.

    rows: [{"label": str, <series name>: value, ...}]
    series: [(name, color)] in draw order.
    ``direct_labels`` names the series at the end of the first group's bars, so the
    color mapping is learned inside the figure instead of below it.
    ``glossary`` is {label: definition} folded into the tooltip, which is how a chart
    of named jargon avoids needing a data-dictionary table under it.
    """
    rows = [r for r in rows if any(r.get(s) is not None for s, _ in series)]
    if not rows:
        return _no_data()
    pad = 96
    bar_w = width - label_w - pad
    mx = 1.0 if percent else (max((r.get(s) or 0) for r in rows for s, _ in series) or 1)
    grp_h = bar_h * len(series) + group_gap
    h = grp_h * len(rows) + (22 if rule is not None else 8)
    out = [f"<svg viewBox='0 0 {width} {h}' role='img' class='chart'>"]
    for i, r in enumerate(rows):
        top = i * grp_h + 6
        out.append(f"<text x='{label_w - 10}' y='{top + grp_h / 2 - 2:.0f}' class='lab' "
                   f"text-anchor='end'>{esc(str(r['label'])[:44])}</text>")
        for j, (name, color) in enumerate(series):
            val = r.get(name)
            if val is None:
                continue
            y = top + j * bar_h
            w = max(1.5, bar_w * val / mx)
            shown = f"{val:.0%}" if percent else fmt.format(val)
            tip = f"{r['label']} — {name}: {shown}"
            if glossary and glossary.get(r["label"]):
                tip += f" · {glossary[r['label']]}"
            out.append(_bar(label_w, y, w, bar_h - 3, color, tip)
                       + f"<text x='{label_w + w + 6:.1f}' y='{y + bar_h - 4}' class='val'>"
                         f"{esc(shown)}</text>")
            if direct_labels and i == 0:
                out.append(f"<text x='{label_w + w + 34:.1f}' y='{y + bar_h - 4}' "
                           f"class='val key-in'>{esc(name)}</text>")
    if rule is not None:
        x = label_w + bar_w * rule / mx
        out.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='2' y2='{h - 20}' class='rule'/>"
                   f"<text x='{x + 5:.1f}' y='{h - 7}' class='muted-svg'>{esc(rule_label)}</text>")
    out.append("</svg>")
    return "".join(out) + _legend(series)


def _legend(series):
    keys = "".join(f"<span class='key'><i style='background:{c}'></i>{esc(n)}</span>"
                   for n, c in series)
    return f"<div class='legend'>{keys}</div>"


def segbar(segments, *, width=W, height=30):
    """One bar split into proportional labelled segments — the whole-corpus view of
    kept/weakened/dropped/added, which as 39 unlabelled columns was unreadable.

    segments: [(name, value, color)]
    """
    segments = [(n, v, c) for n, v, c in segments if v]
    total = sum(v for _, v, _ in segments)
    if not total:
        return _no_data()
    out = [f"<svg viewBox='0 0 {width} {height + 4}' role='img' class='chart'>"]
    x = 0.0
    for name, val, color in segments:
        w = width * val / total
        out.append(f"<rect x='{x:.1f}' y='0' width='{max(w - 2, 1):.1f}' height='{height}' "
                   f"fill='{color}' data-tip='{esc(name)}: {val} ({val / total:.0%})'/>")
        if w > 68:
            out.append(f"<text x='{x + 9:.1f}' y='{height / 2 + 4:.0f}' class='seg-l'>"
                       f"{esc(name)} {val}</text>")
        x += w
    out.append("</svg>")
    return "".join(out) + _legend([(f"{n} · {v}", c) for n, v, c in segments])


def stacked_bar(rows, *, categories, width=W, height=270, xlabel="", ylabel=""):
    """One stacked column per record. rows: [{"label", "segments", "tips"}]."""
    rows = [r for r in rows if r.get("segments")]
    if not rows:
        return _no_data()
    left, bottom, top = 44, 34, 10
    totals = [sum((r["segments"].get(c) or 0) for c, _ in categories) for r in rows]
    mx = max(totals) or 1
    plot_h = height - bottom - top
    bw = (width - left - 12) / len(rows)
    out = [f"<svg viewBox='0 0 {width} {height}' role='img' class='chart'>"]
    for gy in (0, 0.5, 1.0):
        y = top + plot_h * (1 - gy)
        out.append(f"<line x1='{left}' x2='{width - 12}' y1='{y:.1f}' y2='{y:.1f}' class='grid'/>"
                   f"<text x='{left - 7}' y='{y + 4:.1f}' class='val' text-anchor='end'>"
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
            out.append(f"<rect x='{x:.1f}' y='{y_cursor + 1:.1f}' width='{w:.1f}' "
                       f"height='{max(seg_h - 1, 0.5):.1f}' fill='{color}' "
                       f"data-tip='{esc(tip)}'/>")
        if len(rows) <= 24:
            out.append(f"<text x='{x + w / 2:.1f}' y='{height - 20}' class='val' "
                       f"text-anchor='middle' transform='rotate(-40 {x + w / 2:.1f} "
                       f"{height - 20})'>{esc(str(r['label'])[-6:])}</text>")
    if xlabel:
        out.append(f"<text x='{width / 2:.0f}' y='{height - 2}' class='muted-svg' "
                   f"text-anchor='middle'>{esc(xlabel)}</text>")
    if ylabel:
        out.append(f"<text x='2' y='{top - 1}' class='muted-svg'>{esc(ylabel)}</text>")
    out.append("</svg>")
    return "".join(out) + _legend(categories)


def histogram(counts, *, width=W, height=170, color=None, xlabel=""):
    """Distribution of a score or length. counts: [(bucket_label, n)]."""
    counts = list(counts)
    if not counts:
        return _no_data()
    left, bottom, top = 40, 30, 8
    mx = max(n for _, n in counts) or 1
    plot_h = height - bottom - top
    bw = (width - left - 10) / len(counts)
    out = [f"<svg viewBox='0 0 {width} {height}' role='img' class='chart'>"]
    for gy in (0, 0.5, 1.0):
        y = top + plot_h * (1 - gy)
        out.append(f"<line x1='{left}' x2='{width - 10}' y1='{y:.1f}' y2='{y:.1f}' class='grid'/>"
                   f"<text x='{left - 7}' y='{y + 4:.1f}' class='val' text-anchor='end'>"
                   f"{int(mx * gy)}</text>")
    for i, (lab, n) in enumerate(counts):
        bh = plot_h * n / mx
        x = left + i * bw + bw * 0.12
        out.append(f"<rect x='{x:.1f}' y='{top + plot_h - bh:.1f}' width='{bw * 0.76:.1f}' "
                   f"height='{bh:.1f}' fill='{color or PAL[0]}' data-tip='{esc(lab)}: {n}'/>"
                   f"<text x='{x + bw * 0.38:.1f}' y='{height - 16}' class='val' "
                   f"text-anchor='middle'>{esc(lab)}</text>")
    if xlabel:
        out.append(f"<text x='{width / 2:.0f}' y='{height - 2}' class='muted-svg' "
                   f"text-anchor='middle'>{esc(xlabel)}</text>")
    out.append("</svg>")
    return "".join(out)


def scatter(points, *, xdomain=None, ydomain=None, marks=(), width=W, height=330):
    """points/marks: [{"x","y","color","tip"}]. marks draw larger and ringed (the
    per-arm means the dots scatter around). Axis names belong in figure()'s note."""
    pts = [p for p in points if p.get("x") is not None and p.get("y") is not None]
    if not pts:
        return _no_data()
    left, right, top, bottom = 44, 14, 12, 26
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
                   f"<text x='{left - 7}' y='{py(gy) + 4:.1f}' class='val' "
                   f"text-anchor='end'>{gy:.0f}</text>")
    out.append(f"<line x1='{left}' x2='{width - right}' y1='{py(y0):.1f}' y2='{py(y0):.1f}' "
               f"class='axis'/>")
    for k in range(6):
        gx = x0 + (x1 - x0) * k / 5
        out.append(f"<text x='{px(gx):.1f}' y='{height - 10}' class='val' "
                   f"text-anchor='middle'>{gx:.0f}</text>")
    for p in pts:
        out.append(f"<circle cx='{px(p['x']):.1f}' cy='{py(p['y']):.1f}' r='4.5' "
                   f"fill='{p.get('color', PAL[0])}' stroke='var(--surface-0)' "
                   f"stroke-width='1.2' opacity='.82' data-tip='{esc(p.get('tip', ''))}'/>")
    for m in marks:
        out.append(f"<rect x='{px(m['x']) - 7:.1f}' y='{py(m['y']) - 7:.1f}' width='14' "
                   f"height='14' transform='rotate(45 {px(m['x']):.1f} {py(m['y']):.1f})' "
                   f"fill='{m.get('color', PAL[0])}' stroke='var(--surface-0)' "
                   f"stroke-width='2' data-tip='{esc(m.get('tip', ''))}'/>")
    out.append("</svg>")
    return "".join(out)


def highlight(text, spans):
    """Escaped text with each verbatim span wrapped in <mark>.

    Fail-open, matching the viewer: spans were substring-validated at audit time, so
    a span that no longer locates renders unhighlighted rather than corrupting text.
    """
    out = esc(text)
    for span in spans or []:
        if not span:
            continue
        marked = esc(span)
        if marked in out:
            out = out.replace(marked, f"<mark>{marked}</mark>", 1)
    return f"<div class='resp'>{out}</div>"


def sidebyside(left_title, left_html, right_title, right_html, left_tone="", right_tone=""):
    return (f"<div class='pair'>"
            f"<div class='pane {left_tone}'><h4 class='pane-h'>{esc(left_title)}</h4>{left_html}</div>"
            f"<div class='pane {right_tone}'><h4 class='pane-h'>{esc(right_title)}</h4>"
            f"{right_html}</div></div>")


def quote(text):
    return f"<blockquote>{esc(text)}</blockquote>"


CSS = """
:root{color-scheme:only light;
--surface-0:#ffffff;--surface-1:#faf9f6;--surface-2:#f2f1ec;
--border:#dcd9cf;--hairline:#ebe9e1;--grid:#e6e4dc;--axis:#c9c6ba;
--text-primary:#12110f;--text-secondary:#4a4844;--text-muted:#6e6c62;
--link:#1c5cab;--link-rule:#b7d3f6;
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;
--series-5:#e87ba4;--series-6:#008300;--series-7:#4a3aa7;--series-8:#e34948;
--good:#0ca30c;--warn:#fab219;--bad:#d03b3b;
--good-ink:#0a6b12;--warn-ink:#7a4d00;--bad-ink:#a52222;
--good-wash:#eaf6ea;--warn-wash:#fdf3dc;--bad-wash:#fbeceb;
--mark:#fdf0bf}
*{box-sizing:border-box}
html{--serif:ui-serif,Charter,"Bitstream Charter","Iowan Old Style","Source Serif 4","Charis SIL",Georgia,serif;
--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
font:1.0625rem/1.62 var(--serif);-webkit-text-size-adjust:100%}
.skip{position:absolute;left:-9999px}
.skip:focus{left:12px;top:12px;z-index:20;background:var(--surface-0);padding:8px 12px;
border:1px solid var(--border);font-family:var(--sans);font-size:.85rem}

/* Shell: a sticky rail, a prose measure, and a figure track that bleeds past it. */
.shell{display:grid;grid-template-columns:12.5rem minmax(0,50rem);column-gap:3.5rem;
max-width:1120px;margin:0 auto;padding:44px 28px 110px}
.rail{grid-column:1;position:sticky;top:32px;align-self:start;font:500 .78rem/1.45 var(--sans)}
main{grid-column:2;min-width:0}
section{display:grid;grid-template-columns:[text-start] minmax(0,38rem) [text-end] 1fr [full-end]}
section>*{grid-column:text-start/text-end}
section>figure,section>.tiles,section>.scroll,section>.pair,section>details{
grid-column:text-start/full-end}
section+section{margin-top:5rem}
header.top{margin-bottom:3.2rem}

/* Type: the serif argues, the sans measures. */
h1{font:700 2.6rem/1.07 var(--serif);letter-spacing:-.02em;margin:0 0 .5rem;
text-wrap:balance;font-variant-numeric:proportional-nums}
h2{font:600 1.55rem/1.2 var(--serif);letter-spacing:-.011em;margin:0 0 .4rem;text-wrap:balance}
h3{font:600 1.1rem/1.3 var(--serif);margin:2.3rem 0 .3rem;text-wrap:balance}
h4{font:650 .82rem/1.35 var(--sans);margin:1.5rem 0 .4rem;color:var(--text-primary)}
p{margin:0 0 1.05em;color:var(--text-secondary);text-wrap:pretty}
ul{color:var(--text-secondary);padding-left:20px;margin:0 0 1.05em}li{margin:.3em 0}
.lede{font:1.22rem/1.5 var(--serif);color:var(--text-primary);margin:0 0 1.1rem;max-width:40rem}
.dek{font:.9rem/1.5 var(--sans);color:var(--text-muted);margin:0 0 1.4rem;max-width:44rem}
.eyebrow{display:block;font:650 .68rem/1 var(--sans);text-transform:uppercase;
letter-spacing:.1em;color:var(--text-muted);margin-bottom:.7rem}
.meta{font:.8rem/1.55 var(--sans);color:var(--text-muted);margin:1.2rem 0 0;
padding-top:1rem;border-top:1px solid var(--border);max-width:46rem}
.muted{color:var(--text-muted);font:.84rem/1.5 var(--sans)}
.mono{font-family:var(--mono);font-size:.86em}

/* Rail: section numbers live here, so headings stay plain English. */
.rail ol{list-style:none;margin:0;padding:0;counter-reset:sec}
.rail li{display:grid;grid-template-columns:1.3em 1fr;gap:7px;margin:0 0 .62rem}
.rail li::before{counter-increment:sec;content:counter(sec);color:var(--text-muted);
font-variant-numeric:tabular-nums;text-align:right;font-size:.72rem;line-height:1.6}
.rail li.nonum::before{content:"";counter-increment:none}
.rail a{color:var(--text-secondary);text-decoration:none}
.rail a:hover{color:var(--text-primary)}
.rail a[aria-current=true]{color:var(--text-primary);font-weight:650}
.rail .away{margin-top:1.3rem;padding-top:.9rem;border-top:1px solid var(--hairline)}
.rail .away a{color:var(--link)}

/* Numbers. Direction is a labelled chip, never a colored numeral. */
.tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0 2rem;
border-top:1px solid var(--border);padding-top:1rem;margin:1.4rem 0 1.6rem}
.tile-v{font:650 1.9rem/1.04 var(--sans);letter-spacing:-.022em;
font-variant-numeric:proportional-nums}
.tile-l{font:.82rem/1.35 var(--sans);color:var(--text-secondary);margin-top:.4rem}
.tile-s{font:.74rem/1.4 var(--sans);color:var(--text-muted);margin-top:.35rem}
.tile-f{margin-top:.5rem}
.tile.hero{grid-column:span 1}.tile.hero .tile-v{font-size:2.9rem}

/* Figures. The title is a caption, not a heading; the caption states the finding. */
figure{margin:1.6rem 0 1.9rem}
.fig-t{font:650 .84rem/1.35 var(--sans);color:var(--text-primary);margin-bottom:.15rem}
.fig-n{font:.78rem/1.5 var(--sans);color:var(--text-muted);margin:0 0 .5rem;max-width:52ch}
.fig-c{font:.8rem/1.55 var(--sans);color:var(--text-secondary);margin-top:.5rem;max-width:58ch}
.chart{width:100%;max-width:800px;height:auto;overflow:visible;display:block;margin:.2rem 0}
.lab,.val,.muted-svg,.seg-l{font-family:var(--sans)}
.lab{font-size:11.5px;fill:var(--text-secondary)}
.val{font-size:11px;fill:var(--text-muted);font-variant-numeric:tabular-nums}
.val.strong{fill:var(--text-primary);font-weight:650}
.key-in{font-style:italic}
.seg-l{font-size:11px;fill:var(--surface-0);font-weight:650}
.muted-svg{font-size:11px;fill:var(--text-muted)}
.grid{stroke:var(--grid);stroke-width:1;shape-rendering:crispEdges}
.axis{stroke:var(--axis);stroke-width:1;shape-rendering:crispEdges}
.rule{stroke:var(--text-muted);stroke-width:1;stroke-dasharray:4 3}
.legend{font:.76rem/1.4 var(--sans);color:var(--text-secondary);display:flex;gap:1rem;
flex-wrap:wrap;margin:.1rem 0 .2rem}
.legend .key{display:inline-flex;align-items:center;gap:6px}
.legend i{width:8px;height:8px;border-radius:50%;display:inline-block;flex:0 0 auto}

/* Tables: hairlines and alignment, no fills. */
.scroll{overflow-x:auto;margin:1rem 0 1.2rem}
table{border-collapse:collapse;width:100%;font:.83rem/1.5 var(--sans);
font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:.5rem .7rem;border-bottom:1px solid var(--hairline);
vertical-align:top}
th:first-child,td:first-child{padding-left:0}
th{color:var(--text-muted);font-weight:600;font-size:.76rem;
border-bottom:1.5px solid var(--border)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.ctr,th.ctr{text-align:center}
tbody tr:last-child td{border-bottom:0}
td b{color:var(--text-primary)}

code{font-family:var(--mono);font-size:.88em;color:var(--text-primary);word-break:break-word}
td code,th code,.meta code{background:var(--surface-2);padding:1px 4px}
pre{font-family:var(--mono);background:var(--surface-1);border:0;
border-left:2px solid var(--border);padding:.9rem 1.1rem;overflow-x:auto;
font-size:.79rem;line-height:1.65;color:var(--text-primary)}
.chip{font:700 .66rem/1.5 var(--sans);text-transform:uppercase;letter-spacing:.07em;
padding:.15rem .4rem;background:var(--surface-2);color:var(--text-secondary);
white-space:nowrap}
.chip.good{background:var(--good-wash);color:var(--good-ink)}
.chip.warn{background:var(--warn-wash);color:var(--warn-ink)}
.chip.bad{background:var(--bad-wash);color:var(--bad-ink)}
blockquote{margin:1rem 0 1.3rem;white-space:pre-wrap;font-size:1.02rem;line-height:1.55;
color:var(--text-primary);padding-left:1.15rem;border-left:2px solid var(--border)}
.warn-note{color:var(--text-primary);border-left:3px solid var(--warn);
background:var(--warn-wash);padding:.55rem .8rem;font-size:.92rem;margin:1rem 0}
.bad-note{color:var(--text-primary);border-left:3px solid var(--bad);
background:var(--bad-wash);padding:.55rem .8rem;font-size:.92rem;margin:1rem 0}
details{margin:1rem 0;border-top:1px solid var(--hairline);padding-top:.6rem}
summary{font:600 .84rem/1.5 var(--sans);cursor:pointer;color:var(--text-secondary)}
summary:hover{color:var(--text-primary)}
summary .sum-m{color:var(--text-muted);font-weight:400}
.det-body{padding-top:.5rem}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:2rem;margin:1.1rem 0}
.pane{min-width:0}
.pane-h{margin-top:0;color:var(--text-muted)}
.resp{white-space:pre-wrap;font-size:.94rem;line-height:1.6;color:var(--text-primary);
border-left:2px solid var(--hairline);padding-left:.9rem}
.pane.pipeline .resp{border-left-color:var(--series-3)}
.pane.plain .resp{border-left-color:var(--series-2)}
mark{background:var(--mark);color:inherit;padding:0 .1em}
a{color:var(--link);text-decoration:underline;text-decoration-thickness:1px;
text-underline-offset:2px;text-decoration-color:var(--link-rule)}
a:hover{text-decoration-color:var(--link)}
a:focus-visible,[tabindex]:focus-visible,summary:focus-visible{outline:2px solid var(--link);
outline-offset:2px}
footer.foot{margin-top:5rem;padding-top:1rem;border-top:1px solid var(--border);
font:.78rem/1.6 var(--sans);color:var(--text-muted)}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:1.6rem;margin:1.6rem 0 0}
.card{border-top:2px solid var(--text-primary);padding-top:.9rem}
.card.soon{border-top-color:var(--border)}
.card h3{margin:0 0 .3rem;font-size:1.22rem}
.card .card-k{font:650 .68rem/1 var(--sans);text-transform:uppercase;letter-spacing:.1em;
color:var(--text-muted);margin-bottom:.5rem}
.card p{font-size:.95rem;margin-bottom:.7rem}
.card .card-n{font:.8rem/1.6 var(--sans);color:var(--text-secondary);
border-top:1px solid var(--hairline);padding-top:.6rem;margin:0 0 .7rem}
.card .card-go{font:650 .86rem/1.4 var(--sans)}
#tip{position:fixed;pointer-events:none;opacity:0;background:var(--text-primary);
color:var(--surface-0);font:12px/1.4 var(--sans);padding:5px 8px;transition:opacity .1s;
z-index:9;max-width:320px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
@media (max-width:1080px){.shell{grid-template-columns:minmax(0,1fr);max-width:52rem;
row-gap:0}.rail{position:static;grid-column:1;margin-bottom:2.4rem}
.rail ol{display:flex;flex-wrap:wrap;gap:.3rem 1.3rem}.rail li{margin:0}
.rail .away{width:100%;margin-top:.8rem}main{grid-column:1}}
@media (max-width:760px){section{grid-template-columns:minmax(0,1fr)}
section>*{grid-column:1}.pair{grid-template-columns:1fr}
.cards{grid-template-columns:1fr}}
@media (max-width:620px){body{font-size:1rem}.shell{padding:26px 16px 70px}
h1{font-size:1.9rem}h2{font-size:1.3rem}.lede{font-size:1.1rem}
.tiles{grid-template-columns:repeat(2,minmax(0,1fr));gap:1.2rem}}
@media print{
@page{margin:16mm 14mm}
:root{--surface-1:#fff;--surface-2:#fff;--hairline:#d8d6cd}
body{font-size:10.5pt;line-height:1.5}
.rail,#tip,.skip{display:none}
.shell,section{display:block;max-width:none}
p,ul,.dek,.fig-c,.fig-n,.lede{max-width:none}
h1,h2,h3,h4,.fig-t,.dek{break-after:avoid-page}
figure,.tiles,table,.pair,blockquote,.card{break-inside:avoid-page}
tr,li{break-inside:avoid}
thead{display:table-header-group}
details{display:block}details>div{display:block!important}summary{list-style:none}
.tiles{display:grid;grid-template-columns:repeat(3,1fr)}
main a[href^="http"]::after{content:" (" attr(href) ")";font-size:.85em;
color:#555;word-break:break-all}
.chip,rect,circle,path,line{-webkit-print-color-adjust:exact;print-color-adjust:exact}
a{color:var(--text-primary)}}
"""

JS = """
(function(){var t=document.getElementById('tip');
document.addEventListener('mouseover',function(e){var el=e.target.closest('[data-tip]');
if(!el){t.style.opacity=0;return;}t.textContent=el.getAttribute('data-tip');t.style.opacity=1;});
document.addEventListener('mousemove',function(e){if(t.style.opacity=='1'){
t.style.left=Math.min(e.clientX+12,window.innerWidth-t.offsetWidth-8)+'px';
t.style.top=(e.clientY-32)+'px';}});
var links={},secs=[];
[].forEach.call(document.querySelectorAll('.rail a[href^="#"]'),function(a){
var s=document.getElementById(a.getAttribute('href').slice(1));
if(s){links[s.id]=a;secs.push(s);}});
if(!secs.length||!window.IntersectionObserver)return;
var io=new IntersectionObserver(function(es){es.forEach(function(en){
if(en.isIntersecting){for(var k in links)links[k].removeAttribute('aria-current');
links[en.target.id].setAttribute('aria-current','true');}});},
{rootMargin:'-15% 0px -70% 0px'});
secs.forEach(function(s){io.observe(s);});})();
"""


def document(*, title, toc, body, eyebrow="", heading=None, lede="", hero="", meta_line="",
             sibling=None, footer=""):
    """The shell. One file, one theme, no external anything.

    ``sibling`` is (href, label) for the companion report — one link in the rail, not
    a tab bar: a tab that 404s when the file travels alone is a lie about the artefact.
    """
    items = []
    for i, l in toc:
        cls = " class='nonum'" if i in ("summary", "appendix") else ""
        items.append(f"<li{cls}><a href='#{i}'>{esc(l)}</a></li>")
    away = (f"<div class='away'><a href='{esc(sibling[0])}'>{esc(sibling[1])} &rarr;</a></div>"
            if sibling else "")
    return (f"<!DOCTYPE html>\n<html lang='en'>\n<meta charset='utf-8'>\n"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>\n"
            f"<meta name='color-scheme' content='only light'>\n"
            f"<title>{esc(title)}</title>\n<style>{CSS}</style>\n"
            f"<a class='skip' href='#main'>Skip to content</a>\n"
            f"<div class='shell'>\n"
            f"<nav class='rail' aria-label='Sections'><ol>{''.join(items)}</ol>{away}</nav>\n"
            f"<main id='main'>\n<header class='top'>\n"
            + (f"<span class='eyebrow'>{esc(eyebrow)}</span>\n" if eyebrow else "")
            + f"<h1>{esc(heading if heading is not None else title)}</h1>\n"
            + (f"<p class='lede'>{inline_md(lede)}</p>\n" if lede else "")
            + hero
            + (f"<p class='meta'>{meta_line}</p>\n" if meta_line else "")
            + f"</header>\n{body}\n"
            + (f"<footer class='foot'>{footer}</footer>\n" if footer else "")
            + f"</main>\n</div>\n"
            f"<div id='tip'></div>\n<script>{JS}</script>\n</html>\n")
