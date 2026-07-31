"""Presentation primitives for the standalone HTML reports: CSS, SVG charts, shell.

Knows nothing about any pipeline — it takes numbers and returns HTML strings, so both
datasets' reports on the handoff page share one look.

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
charting library, and the only JS is a tooltip and the chooser.

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
    out = _MD_LINK.sub(_link, out)
    return out


# Leaving the page means leaving it in a NEW TAB: this is a long read whose chooser
# state lives in the URL, and a reader who follows a link out and comes back with the
# back button lands on a page that has closed itself again.
NEW_TAB = " target='_blank' rel='noopener noreferrer'"

# The outbound mark, drawn rather than typed. As a glyph (U+2197) it is a hairline in
# most faces and a different shape in every one; this page is printed and screenshotted,
# so the mark has to be the same weight as the type it sits beside, everywhere.
EXT_ARROW = ("<svg class='ext' viewBox='0 0 12 12' width='9' height='9' aria-hidden='true' "
             "fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' "
             "stroke-linejoin='round'><path d='M3.1 8.9 8.9 3.1'/>"
             "<path d='M4.6 3.1h4.3v4.3'/></svg>")


def _link(m):
    """A link. One that leaves the page says so, with the arrow that means exactly that.

    On a page that is a single file, "does this take me somewhere else" is the only
    distinction between links that matters, so it is marked rather than left to the
    reader to guess from the href.
    """
    label, href = m.group(1), m.group(2)
    if not href.startswith("http"):
        return f"<a href='{href}'>{label}</a>"
    return f"<a href='{href}'{NEW_TAB}>{label}{EXT_ARROW}</a>"


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
    """One bar split into proportional segments — the whole-corpus view of
    kept/weakened/dropped/added, which as 39 unlabelled columns was unreadable.

    segments: [(name, value, color)]

    Names and counts live in the legend below the bar, not inside it. Segment labels
    drawn on the fill were surface-coloured text at 2.5:1 on the green and 2.8:1 on the
    terracotta — a fail on cream and already a fail on white.
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


def sub(anchor, text):
    """A subheading that is also a deep-link target.

    The page is one document with two report sections in it, so every beat inside a
    section needs its own id: a reader arriving from the dataset card lands on #dad,
    and anyone quoting a finding wants #dad-weak.
    """
    return f"<h3 id='{esc(anchor)}'>{esc(text)}</h3>"


def illustration(data_uri="", alt="", label="Illustration"):
    """The hero's illustration.

    ``data_uri`` must be a ``data:`` URI — the whole page is one file, so a reference
    to anything outside it (even a relative path) breaks the artefact the moment it
    travels. Without one the slot renders empty at the right proportions, so the hero
    keeps the shape the finished page will have.
    """
    if not data_uri:
        return ("\n<!-- TODO: hero illustration. Drop a PNG into report/assets/hero.png; "
                "build_report.py inlines it as a data URI. No external asset may be "
                "referenced: this page has to open offline from the filesystem. -->\n"
                f"<div class='illo'><span>{esc(label)}</span></div>\n")
    if not data_uri.startswith("data:"):
        raise ValueError("the hero illustration must be a data: URI — the page is one file")
    return (f"\n<div class='illo art'><img src='{esc(data_uri)}' alt='{esc(alt)}'></div>\n")




def hero(title, art="", intro=""):
    """The opening: the illustration, the title, and the lines that follow from it.

    The intro is part of the hero rather than a section of its own — it is the second
    half of the title's sentence, and a heading over it ("Intro") only told a reader
    what they could already see. It carries the ``#intro`` id so the skip link has
    somewhere to land.
    """
    return (f"<header class='hero'>{art}<h1>{esc(title)}</h1>"
            + (f"<div class='hero-intro' id='intro'>{intro}</div>" if intro else "")
            + "</header>\n")


# Monochrome marks, drawn inline and inheriting currentColor: the page is one file, and a
# colour logo would fight a palette built out of ink on paper. The GitHub silhouette is
# the published mark; the Hugging Face one is a simplified face, which is what survives
# being drawn at 15px in a single colour.
# Marks drawn inline: the page is one file, so a logo is path data or it is nothing.
# GitHub's is the published silhouette and inherits currentColor. Hugging Face's is
# their actual logo, fetched from huggingface.co/front/assets, keeping its own fills —
# it IS a smiley face, but theirs, hands and all, rather than a circle I drew.
# (name -> viewBox, width, height, paths)
ICONS = {
    "github": ("0 0 16 16", 15, 15,
               "<path d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 "
               "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-"
               ".15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-"
               ".87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02."
               "08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-"
               ".82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 "
               "3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 "
               "0 0 16 8c0-4.42-3.58-8-8-8z'/>"),
    "hf": ("0 0 95 88", 16, 15, '<path fill="#FFD21E" d="M47.21 76.5a34.75 34.75 0 1 0 0-69.5 34.75 34.75 0 0 0 0 69.5Z"/><path fill="#FF9D0B" d="M81.96 41.75a34.75 34.75 0 1 0-69.5 0 34.75 34.75 0 0 0 69.5 0Zm-73.5 0a38.75 38.75 0 1 1 77.5 0 38.75 38.75 0 0 1-77.5 0Z"/><path fill="#3A3B45" d="M58.5 32.3c1.28.44 1.78 3.06 3.07 2.38a5 5 0 1 0-6.76-2.07c.61 1.15 2.55-.72 3.7-.32ZM34.95 32.3c-1.28.44-1.79 3.06-3.07 2.38a5 5 0 1 1 6.76-2.07c-.61 1.15-2.56-.72-3.7-.32Z"/><path fill="#FF323D" d="M46.96 56.29c9.83 0 13-8.76 13-13.26 0-2.34-1.57-1.6-4.09-.36-2.33 1.15-5.46 2.74-8.9 2.74-7.19 0-13-6.88-13-2.38s3.16 13.26 13 13.26Z"/><path fill="#3A3B45" d="M39.43 54a8.7 8.7 0 0 1 5.3-4.49c.4-.12.81.57 1.24 1.28.4.68.82 1.37 1.24 1.37.45 0 .9-.68 1.33-1.35.45-.7.89-1.38 1.32-1.25a8.61 8.61 0 0 1 5 4.17c3.73-2.94 5.1-7.74 5.1-10.7 0-2.34-1.57-1.6-4.09-.36l-.14.07c-2.31 1.15-5.39 2.67-8.77 2.67s-6.45-1.52-8.77-2.67c-2.6-1.29-4.23-2.1-4.23.29 0 3.05 1.46 8.06 5.47 10.97Z"/><path fill="#FF9D0B" d="M70.71 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM24.21 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM17.52 48c-1.62 0-3.06.66-4.07 1.87a5.97 5.97 0 0 0-1.33 3.76 7.1 7.1 0 0 0-1.94-.3c-1.55 0-2.95.59-3.94 1.66a5.8 5.8 0 0 0-.8 7 5.3 5.3 0 0 0-1.79 2.82c-.24.9-.48 2.8.8 4.74a5.22 5.22 0 0 0-.37 5.02c1.02 2.32 3.57 4.14 8.52 6.1 3.07 1.22 5.89 2 5.91 2.01a44.33 44.33 0 0 0 10.93 1.6c5.86 0 10.05-1.8 12.46-5.34 3.88-5.69 3.33-10.9-1.7-15.92-2.77-2.78-4.62-6.87-5-7.77-.78-2.66-2.84-5.62-6.25-5.62a5.7 5.7 0 0 0-4.6 2.46c-1-1.26-1.98-2.25-2.86-2.82A7.4 7.4 0 0 0 17.52 48Zm0 4c.51 0 1.14.22 1.82.65 2.14 1.36 6.25 8.43 7.76 11.18.5.92 1.37 1.31 2.14 1.31 1.55 0 2.75-1.53.15-3.48-3.92-2.93-2.55-7.72-.68-8.01.08-.02.17-.02.24-.02 1.7 0 2.45 2.93 2.45 2.93s2.2 5.52 5.98 9.3c3.77 3.77 3.97 6.8 1.22 10.83-1.88 2.75-5.47 3.58-9.16 3.58-3.81 0-7.73-.9-9.92-1.46-.11-.03-13.45-3.8-11.76-7 .28-.54.75-.76 1.34-.76 2.38 0 6.7 3.54 8.57 3.54.41 0 .7-.17.83-.6.79-2.85-12.06-4.05-10.98-8.17.2-.73.71-1.02 1.44-1.02 3.14 0 10.2 5.53 11.68 5.53.11 0 .2-.03.24-.1.74-1.2.33-2.04-4.9-5.2-5.21-3.16-8.88-5.06-6.8-7.33.24-.26.58-.38 1-.38 3.17 0 10.66 6.82 10.66 6.82s2.02 2.1 3.25 2.1c.28 0 .52-.1.68-.38.86-1.46-8.06-8.22-8.56-11.01-.34-1.9.24-2.85 1.31-2.85Z"/><path fill="#FFD21E" d="M38.6 76.69c2.75-4.04 2.55-7.07-1.22-10.84-3.78-3.77-5.98-9.3-5.98-9.3s-.82-3.2-2.69-2.9c-1.87.3-3.24 5.08.68 8.01 3.91 2.93-.78 4.92-2.29 2.17-1.5-2.75-5.62-9.82-7.76-11.18-2.13-1.35-3.63-.6-3.13 2.2.5 2.79 9.43 9.55 8.56 11-.87 1.47-3.93-1.71-3.93-1.71s-9.57-8.71-11.66-6.44c-2.08 2.27 1.59 4.17 6.8 7.33 5.23 3.16 5.64 4 4.9 5.2-.75 1.2-12.28-8.53-13.36-4.4-1.08 4.11 11.77 5.3 10.98 8.15-.8 2.85-9.06-5.38-10.74-2.18-1.7 3.21 11.65 6.98 11.76 7.01 4.3 1.12 15.25 3.49 19.08-2.12Z"/><path fill="#FF9D0B" d="M77.4 48c1.62 0 3.07.66 4.07 1.87a5.97 5.97 0 0 1 1.33 3.76 7.1 7.1 0 0 1 1.95-.3c1.55 0 2.95.59 3.94 1.66a5.8 5.8 0 0 1 .8 7 5.3 5.3 0 0 1 1.78 2.82c.24.9.48 2.8-.8 4.74a5.22 5.22 0 0 1 .37 5.02c-1.02 2.32-3.57 4.14-8.51 6.1-3.08 1.22-5.9 2-5.92 2.01a44.33 44.33 0 0 1-10.93 1.6c-5.86 0-10.05-1.8-12.46-5.34-3.88-5.69-3.33-10.9 1.7-15.92 2.78-2.78 4.63-6.87 5.01-7.77.78-2.66 2.83-5.62 6.24-5.62a5.7 5.7 0 0 1 4.6 2.46c1-1.26 1.98-2.25 2.87-2.82A7.4 7.4 0 0 1 77.4 48Zm0 4c-.51 0-1.13.22-1.82.65-2.13 1.36-6.25 8.43-7.76 11.18a2.43 2.43 0 0 1-2.14 1.31c-1.54 0-2.75-1.53-.14-3.48 3.91-2.93 2.54-7.72.67-8.01a1.54 1.54 0 0 0-.24-.02c-1.7 0-2.45 2.93-2.45 2.93s-2.2 5.52-5.97 9.3c-3.78 3.77-3.98 6.8-1.22 10.83 1.87 2.75 5.47 3.58 9.15 3.58 3.82 0 7.73-.9 9.93-1.46.1-.03 13.45-3.8 11.76-7-.29-.54-.75-.76-1.34-.76-2.38 0-6.71 3.54-8.57 3.54-.42 0-.71-.17-.83-.6-.8-2.85 12.05-4.05 10.97-8.17-.19-.73-.7-1.02-1.44-1.02-3.14 0-10.2 5.53-11.68 5.53-.1 0-.19-.03-.23-.1-.74-1.2-.34-2.04 4.88-5.2 5.23-3.16 8.9-5.06 6.8-7.33-.23-.26-.57-.38-.98-.38-3.18 0-10.67 6.82-10.67 6.82s-2.02 2.1-3.24 2.1a.74.74 0 0 1-.68-.38c-.87-1.46 8.05-8.22 8.55-11.01.34-1.9-.24-2.85-1.31-2.85Z"/><path fill="#FFD21E" d="M56.33 76.69c-2.75-4.04-2.56-7.07 1.22-10.84 3.77-3.77 5.97-9.3 5.97-9.3s.82-3.2 2.7-2.9c1.86.3 3.23 5.08-.68 8.01-3.92 2.93.78 4.92 2.28 2.17 1.51-2.75 5.63-9.82 7.76-11.18 2.13-1.35 3.64-.6 3.13 2.2-.5 2.79-9.42 9.55-8.55 11 .86 1.47 3.92-1.71 3.92-1.71s9.58-8.71 11.66-6.44c2.08 2.27-1.58 4.17-6.8 7.33-5.23 3.16-5.63 4-4.9 5.2.75 1.2 12.28-8.53 13.36-4.4 1.08 4.11-11.76 5.3-10.97 8.15.8 2.85 9.05-5.38 10.74-2.18 1.69 3.21-11.65 6.98-11.76 7.01-4.31 1.12-15.26 3.49-19.08-2.12Z"/>'),
}


def icon(name):
    if name not in ICONS:
        return ""
    viewbox, w, h, paths = ICONS[name]
    return (f"<svg class='ico' viewBox='{viewbox}' width='{w}' height='{h}' "
            f"aria-hidden='true' fill='currentColor'>{paths}</svg>")


def linkbutton(href, label, name="", meta=""):
    """An outbound link as a button: mark, label, and the arrow that says it leaves.

    Every link off this page is one of two destinations, so both get the same treatment
    and a reader can tell at a glance which is which.
    """
    return (f"<a class='lbtn' href='{esc(href)}'{NEW_TAB}>{icon(name)}"
            f"<span>{esc(label)}</span>"
            + (f"<span class='lbtn-m'>{esc(meta)}</span>" if meta else "")
            + EXT_ARROW + "</a>")


def compare(columns, rows, actions=()):
    """The two datasets, side by side, with their names as the masthead.

    columns: [(name, description)]. rows: [(label, cell, cell)]. actions: [(cell, cell)],
    one row of buttons at the foot of each column.

    A comparison is a table — the whole point is that "records" lines up with "records" —
    but the names carry the section instead of a heading above it, so the header cells do
    the work a masthead would.
    """
    heads = "".join(f"<th><span class='cmp-name'>{esc(name)}</span>"
                    + (f"<span class='cmp-d'>{inline_md(desc)}</span>" if desc else "")
                    + "</th>" for name, desc in columns)
    body = "".join("<tr><th class='cmp-k' scope='row'>" + esc(label) + "</th>"
                   + "".join(f"<td>{esc(c)}</td>" for c in cells) + "</tr>"
                   for label, *cells in rows)
    foot = "".join("<tr><td></td>" + "".join(f"<td class='cmp-a'>{esc(c)}</td>" for c in cells)
                   + "</tr>" for cells in actions)
    return (f"<div class='cmp-wrap'><table class='cmp'>"
            f"<thead><tr><td class='cmp-corner'></td>{heads}</tr></thead>"
            f"<tbody>{body}</tbody>"
            + (f"<tfoot>{foot}</tfoot>" if foot else "")
            + "</table></div>")


def iconlink(href, label, name=""):
    """A link with its provider's mark and the outbound arrow. Not a button: in the
    footer there is nothing to press, only two places to go."""
    return (f"<a class='ilink' href='{esc(href)}'{NEW_TAB}>{icon(name)}"
            f"<span>{esc(label)}</span>{EXT_ARROW}</a>")


def chooser(options, prompt=""):
    """The two datasets as a choice, then the chosen one below.

    options: [(panel_id, label)]. Just the names: the description and the figures are
    both on the page already, a few inches up, and a reader choosing between two names
    needs neither repeated. Nothing is open on load — the choice is the point — and a
    ``#panel_id`` in the URL opens that panel, so a deep link from the dataset card
    still lands where it says it will.

    The buttons come wrapped in ``.choicebar``, which is what sticks to the top of the
    screen: the bar needs a full-column box to carry the page's own background, and
    ``.choices`` is 40rem centred, so a report would scroll up either side of it.
    """
    buttons = []
    for pid, label in options:
        buttons.append(
            f"<button class='choice' type='button' role='tab' aria-selected='false' "
            f"aria-controls='{esc(pid)}' data-panel='{esc(pid)}' id='choose-{esc(pid)}'>"
            f"{esc(label)}<span class='choice-a' aria-hidden='true'>&darr;</span></button>")
    return ((f"<p class='choose-q'>{inline_md(prompt)}</p>" if prompt else "")
            + f"<div class='choicebar'><div class='choices' role='tablist'>"
              f"{''.join(buttons)}</div></div>")


def tabs(panes):
    """Several records behind one set of buttons. panes: [(id, label, body, open_)].

    The same mechanism as the chooser, not a second one: buttons carry ``data-pane`` and
    the page's own inline JS toggles ``hidden``. The pane marked open renders WITHOUT
    ``hidden``, so with JS off this degrades to one visible record rather than to none,
    and the print rule expands the rest.
    """
    kept = [p for p in panes if p]
    if not kept:
        return ""
    btns = "".join(
        f"<button class='tab' type='button' role='tab' data-pane='{esc(pid)}' "
        f"aria-selected='{'true' if open_ else 'false'}' "
        f"aria-controls='{esc(pid)}'>{esc(label)}</button>"
        for pid, label, _, open_ in kept)
    bodies = "".join(
        f"<div class='pane-x' id='{esc(pid)}' role='tabpanel'"
        f"{'' if open_ else ' hidden'}>{body}</div>"
        for pid, _, body, open_ in kept)
    return (f"<div class='carousel'><div class='tabs' role='tablist'>{btns}</div>"
            f"{bodies}</div>")


def panel(pid, body, cta=None):
    """One chooser panel: closed until its button is pressed.

    ``cta`` is (other_panel_id, label) and renders at the bottom as the way across, so
    the dataset a reader did not choose is still offered to them once they finish.
    """
    tail = ""
    if cta:
        other, label = cta
        tail = (f"<p class='panel-cta'><button class='cta' type='button' "
                f"data-panel='{esc(other)}'>{esc(label)}"
                f"<span aria-hidden='true'> &rarr;</span></button></p>")
    return (f"<section id='{esc(pid)}' class='panel' role='tabpanel' "
            f"aria-labelledby='choose-{esc(pid)}' hidden>{body}{tail}</section>")


def explore_body(bar, panels):
    """The bar and both reports as one block, because that block is the bar's travel.

    ``position:sticky`` moves only inside its containing block, and the containing block
    of a grid item is its own grid area — one row, as tall as the buttons — so a sticky
    bar left in ``#explore``'s grid would have nowhere to go. This plain div holds the
    bar and both panels, so the bar pins for exactly as long as a report is being read
    and is released at the end of it, before the footer.
    """
    return f"<div class='explore-body'>{bar}{panels}</div>"


CSS = """
/* Aged paper, one theme. Every text-on-surface pair below clears WCAG AA (4.5:1) and
   tests/test_report_common.py::test_text_contrast_meets_wcag_aa recomputes them from
   these tokens, so darkening a surface without darkening its ink fails the suite. On
   cream the rules and the chip washes both need to be markedly stronger than they were
   on white, where a 1.1:1 wash still read as a chip. */
:root{color-scheme:only light;
--surface-0:#f7f4ea;--surface-1:#f1ebdd;--surface-2:#e9e1cd;
--border:#cec3a6;--hairline:#ded5be;--grid:#e1d9c4;--axis:#bcaf90;
--text-primary:#1a1712;--text-secondary:#4a443c;--text-muted:#675f54;
--accent:#3b2fa0;--accent-wash:#eae7f7;--accent-edge:#c9c3ea;
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;
--series-5:#e87ba4;--series-6:#008300;--series-7:#4a3aa7;--series-8:#e34948;
--good:#0ca30c;--warn:#fab219;--bad:#d03b3b;
--good-ink:#0a6b12;--warn-ink:#7a4d00;--bad-ink:#a52222;
--good-wash:#dcecd0;--warn-wash:#f4e4c2;--bad-wash:#f4dbd5;
--good-edge:#b6d3a4;--warn-edge:#dcc48c;--bad-edge:#e0b3aa;
--mark:#f2e39c}
*{box-sizing:border-box}
html{--serif:ui-serif,Charter,"Bitstream Charter","Iowan Old Style","Source Serif 4","Charis SIL",Georgia,serif;
--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
font:1.0625rem/1.62 var(--serif);-webkit-text-size-adjust:100%}
.skip{position:absolute;left:-9999px}
.skip:focus{left:12px;top:12px;z-index:20;background:var(--surface-0);padding:8px 12px;
border:1px solid var(--border);font-family:var(--sans);font-size:.85rem}

/* Shell: one centred column, with a figure track that bleeds past the prose measure. */
html{scroll-behavior:smooth}
.shell{max-width:53rem;margin:0 auto;padding:0 28px 110px}
main{min-width:0}
/* minmax(0,1fr), never a bare 1fr: a bare fr track takes its automatic minimum from
   the item's min-content size, so a child with a definite width wider than the column —
   the comparison's 64rem wrapper — GROWS the track past the page, and every percentage
   resolved against that grid area (left:50%, margin-left:50%) then points somewhere to
   the right of the page centre. Measured: the wrapper's centre landed 116px right. */
section{display:grid;
grid-template-columns:[text-start] minmax(0,38rem) [text-end] minmax(0,1fr) [full-end];
scroll-margin-top:2.5rem}
section>*{grid-column:text-start/text-end}
section>figure,section>.tiles,section>.scroll,section>.pair,section>details,
section>.explore-body,section>.lbtns,section>.cmp-wrap,
section>.carousel{grid-column:text-start/full-end}
section+section{margin-top:5rem}
/* The panel is a section, so its own display:grid would beat the browser's default
   [hidden] rule. It has to be said out loud. */
.panel[hidden]{display:none}

/* The hero: the image, the title and the lines that follow from it, centred, with
   enough air to separate them from the page and no more. The two datasets are two
   things, so they are two things here as well as in the table below. */
.hero{display:flex;flex-direction:column;align-items:center;
padding:2.6rem 28px 5rem;text-align:center}
.hero h1{max-width:22ch;margin:6rem 0 0;font-size:3rem}
.hero .illo{margin:0;width:100%}
/* The artwork is 1536x1024 but its ink occupies only a 1318x425 band centred at 48.5%
   of the height — a third of the file is transparent above it and a third below. Left
   uncropped it spends ~340px of the hero on nothing, and every gap measured against it
   is a gap the reader cannot see. Cropped here rather than in the asset, which stays
   exactly as it was supplied. */
.hero .illo.art img{max-width:36rem;margin:6rem auto 0;aspect-ratio:1318/425;
object-fit:cover;object-position:50% 48.5%}
.hero-intro{max-width:60ch;margin:3rem auto 0}
.hero-intro p{margin:0;color:var(--text-secondary);font-size:1.1rem;line-height:1.6}
.hero-intro p+p{margin-top:1.05rem}
.hero-intro ul+p{margin-top:1.9rem}
.hero-intro ul{list-style:none;padding:0;margin:2.1rem 0 0;display:grid;
grid-template-columns:1fr 1fr;gap:1.6rem;text-align:left}
.hero-intro li{margin:0;padding-top:.6rem;border-top:2px solid var(--text-primary);
font:.92rem/1.55 var(--sans);color:var(--text-secondary)}
.hero-intro li b{display:block;margin-bottom:.15rem;color:var(--text-primary);
font:650 1.02rem/1.3 var(--serif)}
/* Type: the serif argues, the sans measures. */
h1{font:700 2.6rem/1.07 var(--serif);letter-spacing:-.02em;margin:0 0 .5rem;
text-wrap:balance;font-variant-numeric:proportional-nums}
h2{font:600 1.55rem/1.2 var(--serif);letter-spacing:-.011em;margin:0 0 .4rem;text-wrap:balance}
h3{font:600 1.1rem/1.3 var(--serif);margin:2.3rem 0 .3rem;text-wrap:balance}
/* Every beat inside a report is its own deep-link target, so it needs the same
   headroom a section gets — and inside a report the chooser is pinned to the top of the
   screen, so the headroom has to clear the bar as well. The bar is 5.21rem:

     .choicebar padding  .8 + .8                = 1.600rem
     .choice padding     1 + 1                  = 2.000rem
     .choice line box    1.14rem x 1.3          = 1.482rem
     .choice border      2 x 1px                = 0.125rem

   7rem is that plus air. Stated in CSS rather than measured in the script because a
   native fragment jump reads it too, and test_a_deep_linked_beat_lands_clear_of_the_bar
   recomputes the sum from these same declarations. */
h3[id]{scroll-margin-top:7rem}
h4{font:650 .82rem/1.35 var(--sans);margin:1.5rem 0 .4rem;color:var(--text-primary)}
p{margin:0 0 1.05em;color:var(--text-secondary);text-wrap:pretty}
ul{color:var(--text-secondary);padding-left:20px;margin:0 0 1.05em}li{margin:.3em 0}
.lede{font:1.22rem/1.5 var(--serif);color:var(--text-primary);margin:0 0 1.1rem;max-width:40rem}
.dek{font:.9rem/1.5 var(--sans);color:var(--text-muted);margin:0 0 1.4rem;max-width:44rem}
.meta{font:.8rem/1.55 var(--sans);color:var(--text-muted);margin:1.2rem 0 0;
padding-top:1rem;border-top:1px solid var(--border);max-width:46rem}
.muted{color:var(--text-muted);font:.84rem/1.5 var(--sans)}
.mono{font-family:var(--mono);font-size:.86em}

/* The choice, and the two ways out of the page. */
.choose-q{font:1.22rem/1.5 var(--serif);color:var(--text-primary);margin:0 0 1.4rem}
/* The choice lines up with the thing being chosen: 40rem centred on the page is exactly
   the two dataset columns above (2 x 20rem), so each button sits under its own column.
   Its heading centres over them for the same reason.

   The child combinator is load-bearing: both reports live INSIDE #explore now (that is
   what gives the sticky bar its travel), and each opens with its own <h2>, so a
   descendant selector here centres and stretches both report titles too. */
#explore>h2{grid-column:text-start/full-end;text-align:center;margin-bottom:1.2rem}
/* The bar's travel. min-width:0 for the reason main has it: it is a grid item now, and a
   grid track takes its automatic minimum from the item's min-content size. */
.explore-body{min-width:0}
/* Pinned to the top of the screen for as long as a report is being read, carrying the
   page's own background so the report scrolls under it and out of sight. Full column
   width, not the buttons' 40rem, or a figure would scroll up either side of it.
   z-index:5 sits under #tip (9) and .skip:focus (20) and over everything else. */
.choicebar{position:sticky;top:0;z-index:5;background:var(--surface-0);padding:.8rem 0}
.choices{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;
width:min(100%,40rem);margin:0 auto}
/* Two names and an arrow, in the accent. The cream fill with a border was doing duty as
   a button, a card, a chip and a code block at once, and had stopped meaning anything;
   an outline in the accent that fills when you choose says "this is a control". */
.choice{display:flex;align-items:center;justify-content:space-between;gap:1rem;
padding:1rem 1.25rem;background:none;border:1px solid var(--accent-edge);
border-radius:4px;cursor:pointer;font:650 1.14rem/1.3 var(--serif);color:var(--accent);
text-align:left}
.choice:hover{background:var(--accent-wash)}
.choice[aria-selected=true]{background:var(--accent);border-color:var(--accent);
color:var(--surface-0)}
.choice-a{font:400 1.1rem/1 var(--sans);opacity:.8}
.panel{margin-top:3.2rem;scroll-margin-top:7rem}
.panel-cta{margin:3rem 0 0;padding-top:1.2rem;border-top:1px solid var(--border)}
/* The primary button: the same shape as the outline ones, filled. It is the one thing
   the page asks a reader to do at the end of a report. */
.cta{display:inline-flex;align-items:center;gap:.45rem;padding:.6rem 1rem;cursor:pointer;
background:var(--accent);border:1px solid var(--accent);border-radius:4px;
font:600 .82rem/1.3 var(--mono);letter-spacing:-.01em;color:var(--surface-0)}
.cta:hover{background:var(--text-primary);border-color:var(--text-primary)}
/* The comparison: a table, but the names are its masthead rather than a heading over
   the top of it, and the last row is what a reader does next.

   WHAT IS CENTRED IS THE PAIR, NOT THE TABLE. The two dataset columns straddle the page
   centre and the field labels hang off their left, in the margin — so the thing being
   compared sits in the middle and the labels read as an index down the side.

   Two steps, both stated as arithmetic rather than left to a layout mode to work out:

     1. .cmp-wrap is centred on the PAGE. left:50% resolves against its grid area, so
        the wrapper must be in the full-bleed track above (the full main column, which is
        centred in the viewport). In the default 38rem prose track it centres on the text
        column instead and the whole block lands ~5.75rem left of the hero.
     2. .cmp is pushed right by exactly `half the wrapper − one column − the labels`, so
        the pair's midpoint lands on the wrapper's midpoint. A percentage margin resolves
        against the wrapper's width, so this is one subtraction and needs no auto margins,
        no flex free space, and no negative margins.

   The three widths are custom properties: change one and the offset follows. */
.cmp-wrap{--cmp-label:10.5rem;--cmp-col:20rem;
position:relative;left:50%;transform:translateX(-50%);
width:min(100vw - 2.5rem,64rem);overflow-x:auto;margin:.4rem 0 0}
.cmp{border-collapse:collapse;table-layout:fixed;
width:calc(var(--cmp-label) + 2*var(--cmp-col));
margin-left:calc(50% - var(--cmp-col) - var(--cmp-label));
font:.86rem/1.55 var(--sans)}
.cmp th,.cmp td{text-align:left;vertical-align:top;padding:.62rem .9rem;
border-bottom:1px solid var(--hairline)}
.cmp thead th{border-bottom:1px solid var(--border);padding:0 .9rem 1.35rem;
width:var(--cmp-col)}
/* table-layout:fixed takes every column width from the FIRST row, so the corner cell
   has to carry the label width — the .cmp-k rule below is in the body rows, where fixed
   layout never looks. */
.cmp .cmp-corner{border:0;width:var(--cmp-label)}
.cmp-name{display:block;font:600 1.28rem/1.2 var(--serif);letter-spacing:-.012em;
color:var(--text-primary)}
.cmp-d{display:block;margin-top:.75rem;font:.85rem/1.55 var(--sans);color:var(--text-muted)}
/* Flush right, hard against the pair, and never wrapped: the labels are an index down
   the side of the comparison, and an index that breaks over two lines stops reading as
   one. --cmp-label is wide enough for the longest of them. */
.cmp-k{font:650 .68rem/1.9 var(--sans);text-transform:uppercase;letter-spacing:.08em;
color:var(--text-muted);width:var(--cmp-label);white-space:nowrap}
/* Everything `.cmp th` already sets — the rule, the alignment, the padding — needs a
   rule that OUT-SPECIFIES it, not merely one that follows it. */
.cmp th.cmp-k{border-bottom:0;text-align:right;padding:.62rem 1.1rem .62rem 0;
vertical-align:middle}
.cmp tbody td{color:var(--text-secondary)}
.cmp tfoot td{border-bottom:0;padding-top:1rem}
.cmp-a{display:table-cell}
/* Each way in sits in the row of the figure it belongs to — the prompts against how
   many there are, the sample records against how many were published — with the figure
   at the column's left edge and the button at its right. */
.cmp-fig{display:flex;align-items:center;justify-content:space-between;gap:1rem}
.lbtns{display:flex;flex-wrap:wrap;gap:.7rem;margin:1.1rem 0}
.lbtn{display:inline-flex;align-items:center;gap:.45rem;padding:.45rem .8rem;
border:1px solid var(--accent-edge);border-radius:4px;background:none;text-decoration:none;
font:600 .8rem/1.3 var(--mono);letter-spacing:-.01em;color:var(--accent)}
.lbtn:hover{background:var(--accent-wash)}
.lbtn .ico{flex:0 0 auto}
.lbtn-m{font-weight:400;color:var(--text-muted)}
.lbtn:hover .lbtn-m{color:var(--text-secondary)}
svg.ext{margin-left:.22em;vertical-align:-.05em;flex:0 0 auto}

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
.lab,.val,.muted-svg{font-family:var(--sans)}
.lab{font-size:11.5px;fill:var(--text-secondary)}
.val{font-size:11px;fill:var(--text-muted);font-variant-numeric:tabular-nums}
.val.strong{fill:var(--text-primary);font-weight:650}
.key-in{font-style:italic}
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
/* The hairline is load-bearing on cream: the washes only reach ~1.15:1 against the
   page, so without an edge a chip stops reading as a chip. */
.chip{font:700 .66rem/1.5 var(--sans);text-transform:uppercase;letter-spacing:.07em;
padding:.1rem .38rem;background:var(--surface-2);color:var(--text-secondary);
border:1px solid var(--border);white-space:nowrap}
.chip.good{background:var(--good-wash);color:var(--good-ink);border-color:var(--good-edge)}
.chip.warn{background:var(--warn-wash);color:var(--warn-ink);border-color:var(--warn-edge)}
.chip.bad{background:var(--bad-wash);color:var(--bad-ink);border-color:var(--bad-edge)}
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
/* The example carousel. Same outline-button family as .choice, one size down: a record
   id is a label, not a title, so it takes the mono face the ids use everywhere else. */
.carousel{margin:1.1rem 0}
.tabs{display:flex;flex-wrap:wrap;gap:.6rem;margin-bottom:1.1rem}
.tab{padding:.4rem .75rem;background:none;border:1px solid var(--accent-edge);
border-radius:4px;cursor:pointer;font:600 .88rem/1.3 var(--mono);color:var(--accent)}
.tab:hover{background:var(--accent-wash)}
.tab[aria-selected=true]{background:var(--accent);border-color:var(--accent);
color:var(--surface-0)}
.pane-x>h4:first-child{margin-top:0}
.resp{white-space:pre-wrap;font-size:.94rem;line-height:1.6;color:var(--text-primary);
border-left:2px solid var(--hairline);padding-left:.9rem}
.pane.pipeline .resp{border-left-color:var(--series-3)}
.pane.plain .resp{border-left-color:var(--series-2)}
mark{background:var(--mark);color:inherit;padding:0 .1em}
/* Selection is the page's one piece of interaction colour, so it is the accent at full
   strength rather than the browser's blue. */
::selection{background:var(--accent);color:var(--surface-0)}
/* A link is a typographic object, not a coloured word: mono against the serif, bold
   enough to hold the accent, and underlined in the accent rather than in a tint of it.
   Buttons are unaffected — .lbtn, .choice, .cta and .skip each set their own font
   shorthand, which beats a bare element selector. */
a{font-family:var(--mono);font-weight:600;font-size:.92em;letter-spacing:-.01em;
color:var(--accent);text-decoration:underline;text-decoration-thickness:2px;
text-underline-offset:3px;text-decoration-color:var(--accent)}
a:hover{background:var(--accent-wash)}
a:focus-visible,[tabindex]:focus-visible,summary:focus-visible{outline:2px solid var(--accent);
outline-offset:2px}
/* One line: who made it on the left, where to go on the right. */
footer.foot{margin-top:5rem;padding-top:1.1rem;border-top:1px solid var(--border);
font:.85rem/1.6 var(--sans);color:var(--text-muted);
display:flex;justify-content:space-between;align-items:baseline;gap:1.5rem;flex-wrap:wrap}
footer.foot p{margin:0;color:inherit}
.foot-links{display:flex;gap:1.6rem}
/* A supplied mark rather than a drawn one — inlined as a data URI like the hero, so
   the page stays one file. */
.ico-img{width:15px;height:15px;border-radius:3px;vertical-align:-.17em;
margin-right:.5rem;flex:0 0 auto}
.ilink{display:inline-flex;align-items:center;gap:.45rem;font:600 .82rem/1.3 var(--mono);
color:var(--accent);text-decoration:underline;text-decoration-thickness:2px;
text-underline-offset:3px;text-decoration-color:var(--accent)}
.ilink:hover{background:var(--accent-wash)}
/* The hero's illustration. Dashed while empty, so an unfilled slot reads as deliberate
   rather than as a broken asset; once filled it is line art on the paper, with no frame
   of its own. */
.illo{aspect-ratio:16/6;margin:2.6rem 0 0;border:1px dashed var(--accent-edge);
display:flex;align-items:center;justify-content:center}
.illo span{font:650 .7rem/1 var(--sans);text-transform:uppercase;letter-spacing:.12em;
color:var(--accent)}
.illo.art{aspect-ratio:auto;border:0;background:none;display:block;margin:4rem 0 .4rem}
.illo.art img{display:block;width:100%;height:auto;max-width:46rem}
#tip{position:fixed;pointer-events:none;opacity:0;background:var(--text-primary);
color:var(--surface-0);font:12px/1.4 var(--sans);padding:5px 8px;transition:opacity .1s;
z-index:9;max-width:320px}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}
*{transition:none!important;animation:none!important}}
/* Below the width the offset needs, the comparison goes back to being an ordinary
   full-width table: a centred pair that runs off the left of the screen is worse than
   an uncentred one. */
/* Below the width the offset needs, the pair cannot straddle the centre without the
   labels running off the left of the screen, so the comparison goes back to being an
   ordinary full-width table. */
@media (max-width:1000px){.cmp-wrap{position:static;left:auto;transform:none;width:100%}
.cmp{table-layout:auto;width:100%;margin-left:0}
.cmp-k{width:8rem;white-space:normal;text-align:left}.cmp thead th{width:auto}}
@media (max-width:760px){section{grid-template-columns:minmax(0,1fr)}
section>*{grid-column:1}.pair{grid-template-columns:1fr}
/* The bar stays pinned on a phone, so it has to stay ONE ROW: stacking the two buttons
   is ~10rem of permanent chrome, a quarter of a small screen. Two columns and tighter
   type instead. A bare font-size keeps the .choice shorthand's 1.3 line-height. */
.choicebar{padding:.6rem 0}.choices{gap:.7rem}
.choice{padding:.7rem .8rem;font-size:1rem}}
@media (max-width:620px){body{font-size:1rem}.shell{padding:0 16px 70px}
/* Tighter again, and the arrow goes: at this width both labels wrap to two lines and
   space-between drops the arrow beside a ragged edge. It is decorative — the pair still
   reads as a control. */
.choicebar{padding:.5rem 0}.choices{gap:.6rem}
.choice{padding:.6rem .7rem;font-size:.95rem}.choice-a{display:none}
h1{font-size:1.9rem}h2{font-size:1.3rem}.lede{font-size:1.1rem}
.hero{padding:1.8rem 16px 3.4rem}.hero h1{margin-top:1.6rem;font-size:2.2rem}
.hero-intro{margin-top:1.2rem}.hero-intro p{font-size:1.05rem}
.hero-intro ul{grid-template-columns:1fr;gap:1.1rem}
.tiles{grid-template-columns:repeat(2,minmax(0,1fr));gap:1.2rem}
.illo{aspect-ratio:16/9}}
@media print{
@page{margin:16mm 14mm}
:root{--surface-1:#fff;--surface-2:#fff;--hairline:#d8d6cd}
body{font-size:10.5pt;line-height:1.5}
#tip,.skip,.choicebar,.panel-cta{display:none}
.shell,section{display:block;max-width:none}
.hero{padding:0 0 1.5rem;display:block;text-align:left}
/* A sheet of paper is narrower than the bleed the centred pair needs, and the labels
   would print off the left edge. */
.cmp-wrap{position:static;left:auto;transform:none;width:100%;overflow:visible}
.cmp{table-layout:auto;width:100%;margin-left:0}
/* A printed page is not a page anyone can click, so both reports print, whichever
   one is open on screen — and every example in the carousel prints, not just the tab
   that happened to be showing. */
.panel[hidden],.pane-x[hidden]{display:block!important}
.tabs{display:none}
p,ul,.dek,.fig-c,.fig-n,.lede{max-width:none}
h1,h2,h3,h4,.fig-t,.dek{break-after:avoid-page}
figure,.tiles,table,.pair,blockquote{break-inside:avoid-page}
.panel{break-before:page}
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
/* The example carousel. Its own block, and before the chooser's early return, because a
   page can carry examples without carrying a chooser. One pane is already visible in the
   markup, so with this script absent the carousel still shows a record. */
[].forEach.call(document.querySelectorAll('.carousel'),function(c){
var tabs=[].slice.call(c.querySelectorAll('.tab'));
tabs.forEach(function(b){b.addEventListener('click',function(){
tabs.forEach(function(o){var on=o===b;o.setAttribute('aria-selected',on?'true':'false');
var p=document.getElementById(o.getAttribute('data-pane'));
if(p){if(on){p.removeAttribute('hidden');}else{p.setAttribute('hidden','');}}});});});});
var choices=[].slice.call(document.querySelectorAll('.choice'));
if(!choices.length)return;
/* Where the bar SITS, not where it is painted. Once sticky takes hold, the bar's own
   getBoundingClientRect() and offsetTop both report the shifted position, so measuring
   from it would scroll to wherever the reader already was. .explore-body never moves and
   the bar is its first child, so its top IS the bar's flow top — which is also the
   sticky threshold, so nothing jumps as the bar pins. */
var flow=document.querySelector('.explore-body');
function open(id,to){
choices.forEach(function(b){var on=b.getAttribute('data-panel')===id;
b.setAttribute('aria-selected',on?'true':'false');
var p=document.getElementById(b.getAttribute('data-panel'));
if(p){if(on){p.removeAttribute('hidden');}else{p.setAttribute('hidden','');}}});
if(!to)return;
var target=document.getElementById(to===true?id:to);
if(!target)return;
/* Choosing a report puts the bar at the top of the screen; a deep link to a beat INSIDE
   a report goes to the beat, clear of the pinned bar. Both are scrollIntoView plus
   scroll-margin-top rather than arithmetic: the headroom the bar needs is stated once,
   in the CSS, where a native fragment jump reads it too — and the smoothness stays
   html{scroll-behavior}, so prefers-reduced-motion can still turn it off. */
(target.classList.contains('panel')&&flow?flow:target).scrollIntoView();}
function mark(id){if(history.replaceState)history.replaceState(null,'','#'+id);
else location.hash=id;}
/* A hash may name a panel (#dad) or anything inside one (#dad-weak, from a quoted
   finding). Either way the panel it lives in is the one to open. */
function fromHash(){var id=(location.hash||'').slice(1);if(!id)return false;
var el=document.getElementById(id);var p=el&&el.closest?el.closest('.panel'):null;
if(!p)return false;open(p.id,id);return true;}
/* One handler for both: a tab and the end-of-report button open a report and put the bar
   back at the top of the screen. */
[].forEach.call(document.querySelectorAll('.choice,.cta'),function(b){
b.addEventListener('click',function(){var id=b.getAttribute('data-panel');
open(id,true);mark(id);});});
window.addEventListener('hashchange',fromHash);
/* Wait for load, not parse: the hero image is a data URI several megabytes long, and
   scrolling to a deep-linked beat before it has laid out puts the reader thousands of
   pixels away from it once the image finally takes up its space. */
if(document.readyState==='complete')fromHash();
else window.addEventListener('load',fromHash);})();
"""


def document(*, title, masthead, body, footer=""):
    """The shell. One file, one theme, no external anything.

    There is no contents rail: the page is a hero, three short sections and a choice, and
    a list of five links beside that is furniture. Everything a reader navigates to is
    either on the first screen or one button away.
    """
    return (f"<!DOCTYPE html>\n<html lang='en'>\n<meta charset='utf-8'>\n"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>\n"
            f"<meta name='color-scheme' content='only light'>\n"
            f"<title>{esc(title)}</title>\n<style>{CSS}</style>\n"
            f"<a class='skip' href='#intro'>Skip to content</a>\n"
            f"{masthead}"
            f"<div class='shell'>\n<main id='main'>\n{body}\n"
            + (f"<footer class='foot'>{footer}</footer>\n" if footer else "")
            + f"</main>\n</div>\n"
            f"<div id='tip'></div>\n<script>{JS}</script>\n</html>\n")
