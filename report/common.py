"""Loading, prose and CLI plumbing shared by the per-pipeline report modules.

Same contract as render.py: stdlib only, no repo imports, no pipeline knowledge — the
report generators have to build in an environment where the pipeline's own
dependencies are not installed, which is also what makes them portable.

Everything here is used by report/page.py and report/dad.py today. Anything that only
one pipeline needs stays in that pipeline's module: in particular the weaknesses floor
splits in two, because ``evals/audit_dad.py`` records its verdicts into
``sections[].rows[]`` and ``evals/audit_sdf.py`` only prints them. So
``audit_verdict_warnings()`` returns nothing for an SDF audit, and an SDF page will
have to compute its own thresholds.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from report import render as R


# ------------------------------------------------------------------ loading

# A run snapshots the prompts it ran with into inputs/prompts/, which is the honest
# source for "how many prompts is this pipeline" — the question a reader who wants to
# run it against their own model is actually asking. Counted: the stage templates only.
# Not the variables matrix (a weighted table, not a prompt), not the reasoning library,
# not archive/, and not *_score.txt, which is an eval rather than a generation stage.
def prompt_count(run_dir, glob):
    """How many stage templates the run was generated with, or None if it kept no
    snapshot of them."""
    snapshot = Path(run_dir) / "inputs" / "prompts"
    if not snapshot.is_dir():
        return None
    n = sum(1 for p in snapshot.glob(glob) if not p.name.endswith("_score.txt"))
    return n or None


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ------------------------------------------------------------------ prose

def parse_content(text, ids):
    """A prose file -> {section_id: markdown}, delimited by ``<!-- id: name -->``.

    ``ids`` is the owning module's tuple. An unknown or missing id raises, so a typo
    can never silently drop a section from a page.
    """
    parts = re.split(r"<!--\s*id:\s*([a-z0-9_]+)\s*-->", text)
    if len(parts) < 3:
        raise ValueError("content file has no '<!-- id: ... -->' section markers")
    found = {}
    for i in range(1, len(parts), 2):
        found[parts[i]] = parts[i + 1].strip()
    unknown = sorted(set(found) - set(ids))
    if unknown:
        raise ValueError(f"content file has unknown section id(s): {', '.join(unknown)}")
    missing = sorted(set(ids) - set(found))
    if missing:
        raise ValueError(f"content file is missing section id(s): {', '.join(missing)}")
    return found


def load_content(paths, ids):
    """Merge one or more prose files into one id namespace.

    Two files may not both own a section, and the union must be exactly ``ids`` — so
    moving a block from a per-pipeline file into the shared one is a rename, never a
    silent duplicate.
    """
    merged, seen = {}, {}
    texts = [(Path(p), Path(p).read_text(encoding="utf-8")) for p in paths]
    all_found = {}
    for path, text in texts:
        parts = re.split(r"<!--\s*id:\s*([a-z0-9_]+)\s*-->", text)
        if len(parts) < 3:
            raise ValueError(f"{path} has no '<!-- id: ... -->' section markers")
        for i in range(1, len(parts), 2):
            sid = parts[i]
            if sid in seen:
                raise ValueError(f"section id '{sid}' is defined in both {seen[sid]} and {path}")
            seen[sid] = path
            all_found[sid] = parts[i + 1].strip()
    unknown = sorted(set(all_found) - set(ids))
    if unknown:
        raise ValueError(f"unknown section id(s) across {', '.join(str(p) for p, _ in texts)}: "
                         f"{', '.join(unknown)}")
    missing = sorted(set(ids) - set(all_found))
    if missing:
        raise ValueError(f"missing section id(s): {', '.join(missing)}")
    merged.update(all_found)
    return merged


_PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+)\}\}")


def fill(text, f):
    """Resolve {{placeholders}} from the facts dict. Unknown key -> build error.

    This is the enforcement half of "no number is ever typed into the prose": a figure
    can only reach the page by being computed from the run's own output, and a prose
    file that references a fact the run does not have fails the build rather than
    shipping a stale sentence.
    """
    def sub(m):
        key = m.group(1)
        if key not in f:
            raise KeyError(f"prose references unknown fact '{{{{{key}}}}}' "
                           f"(available: {', '.join(sorted(f))})")
        return str(f[key])
    return _PLACEHOLDER.sub(sub, text or "")


def prose(content, key, f):
    return R.paragraphs(fill(content.get(key, ""), f))


def section(sid, heading, *blocks):
    """A section. A falsy heading omits the <h2> — for a section whose own content is
    its title, like the comparison, whose two column mastheads say what it is."""
    body = "".join(b for b in blocks if b)
    head = f"<h2>{R.esc(heading)}</h2>" if heading else ""
    return f"<section id='{sid}'>{head}{body}</section>"


_STRIP_BLOCKS = re.compile(
    r"<(script|style|svg|nav)\b.*?</\1>|<blockquote\b.*?</blockquote>"
    r"|<div class='resp'>.*?</div>|<table\b.*?</table>|<!--.*?-->", re.S)


def editorial_words(html):
    """How many words of authored prose a built page carries.

    Corpus text, chart internals, every table — including the derived warnings, whose
    wording comes from the audit — and every ``<nav>`` are excluded, so what is counted is
    the part a person wrote. A rail's labels are the document's own headings, already
    counted where they are written; counting them twice would spend the ceiling on
    navigation and let real prose in under it.

    Printed at build time: the page's whole brief is that a reader with forty seconds gets
    what they need, and prose is the thing that grows back.
    """
    text = _STRIP_BLOCKS.sub(" ", html or "")
    return len(re.findall(r"[A-Za-z][A-Za-z'’-]*", re.sub(r"<[^>]+>", " ", text)))


# ------------------------------------------------------------------ cost

def costs_by_stage(costs):
    agg = {}
    for rec in costs or []:
        stage = rec.get("stage") or "(untagged)"
        entry = agg.setdefault(stage, {"calls": 0, "cost": 0.0, "models": set()})
        entry["calls"] += 1
        entry["cost"] += rec.get("cost_usd") or 0.0
        if rec.get("model"):
            entry["models"].add(rec["model"])
    return agg


def stage_cost_table(costs, labels):
    """(tag, display name) pairs in pipeline order -> the per-stage cost table.

    Stages the labels don't name are appended rather than dropped, so a new stage tag
    shows up as itself instead of vanishing.
    """
    agg = costs_by_stage(costs)
    if not agg:
        return ""
    rows = []
    for tag, label in labels:
        entry = agg.get(tag)
        if entry:
            rows.append((label, ", ".join(sorted(entry["models"])) or "—",
                         entry["calls"], f"${entry['cost']:,.2f}"))
    for tag in sorted(set(agg) - {t for t, _ in labels}):
        rows.append((tag, ", ".join(sorted(agg[tag]["models"])) or "—",
                     agg[tag]["calls"], f"${agg[tag]['cost']:,.2f}"))
    return R.table(["stage", "model", "calls", "cost"], rows, align="llrr")


# ------------------------------------------------------------------ candour floor

def provenance_warnings(manifest, *, n=None, small_n=100):
    """The warnings that are true of any run of any pipeline.

    Severity, not prose, is the contract: these are appended to whatever the audit
    itself flagged, and nothing is ever filtered back out.
    """
    out = []
    cfg = (manifest or {}).get("config") or {}
    backend = cfg.get("backend")
    if backend and backend != "api":
        out.append(("BAD" if backend == "claude_code" else "OK",
                    f"Generated on the `{backend}` backend rather than `api`. `api` is the "
                    "documented faithful mode, and the one a reader reproducing this would "
                    "use. Read these numbers as representative, not exact."))
    if (manifest or {}).get("git_dirty"):
        out.append(("OK", "The working tree had uncommitted changes when this run was generated, "
                          "so the recorded commit does not fully describe the code that ran."))
    if n and n < small_n:
        out.append(("OK", f"n = {n}, from one run on one seed. Every percentage here is "
                          f"indicative."))
    return out


def audit_verdict_warnings(audit):
    """Every BAD or OK row the audit itself recorded.

    Returns [] for an audit with no verdict rows, which is what an SDF audit looks
    like today — that pipeline's page has to derive its own thresholds instead.
    """
    out = []
    for sec in (audit or {}).get("sections") or []:
        for row in sec.get("rows") or []:
            if row.get("verdict") in ("BAD", "OK"):
                out.append((row["verdict"], f"{sec.get('title', '?')} — "
                                            f"{row.get('label', '')}: {row.get('value', '')}"
                                            + (f" {row.get('note')}" if row.get("note") else "")))
    return out


def warnings_table(warnings, *, inline=3, drawer_label="more findings at this level"):
    """BADs first, then the most severe OKs; the rest in a counted drawer.

    The drawer exists so the page is skimmable, and it is COUNTED so that collapsing
    is visibly a view and not a filter. The list itself is never trimmed.
    """
    if not warnings:
        return ""
    ordered = sorted(warnings, key=lambda w: 0 if w[0] == "BAD" else 1)
    bads = [w for w in ordered if w[0] == "BAD"]
    rest = [w for w in ordered if w[0] != "BAD"]
    head, tail = bads + rest[:inline], rest[inline:]

    def build(ws):
        return R.table(["severity", "what the data says"],
                       [(R.Raw(R.chip(sev, "bad" if sev == "BAD" else "warn")),
                         R.Raw(R.inline_md(text))) for sev, text in ws])

    out = build(head)
    if tail:
        out += R.details(f"{len(tail)} {drawer_label}", build(tail))
    return out


# ------------------------------------------------------------------ shell bits

def meta_line(*, run_id, manifest, pairs=()):
    """The provenance line. ``pairs`` is [(label, value_html)] appended in order."""
    m = manifest or {}
    cfg = m.get("config") or {}
    bits = [f"run <span class='mono'>{R.esc(run_id or m.get('run_id', '?'))}</span>",
            f"git <span class='mono'>{R.esc(str(m.get('git_commit', '?'))[:8])}</span>"
            + (" <span class='mono'>+ uncommitted changes</span>" if m.get("git_dirty") else ""),
            f"backend <code>{R.esc(cfg.get('backend', '?'))}</code>"]
    bits += [f"{R.esc(k)} {v}" for k, v in pairs]
    return " · ".join(bits)


# ------------------------------------------------------------------ CLI

def write(path, html, *, label=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"wrote {path} ({len(html):,} bytes){' — ' + label if label else ''}")
    return path


def cli_parser(doc):
    p = argparse.ArgumentParser(description=(doc or "").strip().split("\n")[0])
    p.add_argument("--dad-run", "--run", dest="dad_run", default=None,
                   help="DAD run directory (required)")
    p.add_argument("--sdf-run", dest="sdf_run", default=None,
                   help="SDF run directory. Optional: without it the document corpus's "
                        "column and section say so instead of showing figures")
    p.add_argument("--out-dir", default=None, help="output directory (default report/)")
    p.add_argument("--content", action="append", default=None,
                   help="prose file, repeatable; overrides the page's default prose file(s)")
    p.add_argument("--example", default=None, help="prompt_id to feature as the worked example")
    return p


def die(msg):
    sys.exit(msg)
