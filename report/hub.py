#!/usr/bin/env python3
"""The hub page: what these two corpora are, and why there are two of them.

Everything true of both pipelines is met once, here, so neither report has to carry it:
why this data does not already exist, the *Teaching Claude Why* grounding, the two
routes and the choice between them, the shared measurement philosophy, the shared
limits, and how to read a provenance line. The DAD page used to gesture at the second
route as an unnamed "reviewed sister pipeline" that a reader could not resolve; that
comparison belongs on a page that owns both.

The SDF card renders whether or not its report exists yet. Without a run it says so and
carries no link — a dead link is worse than an honest absence.
"""

from report import common as C
from report import render as R

CONTENT_IDS = (
    "title", "lede", "dad_card", "sdf_card", "why", "corpora", "limits",
)

# A landing page carries no contents list: its job is the hero and the two links out of
# it, and a rail for four short sections is furniture.
TOC = []


def load_inputs(content_paths, dad_run=None, sdf_run=None):
    from pathlib import Path
    out = {"content": C.load_content(content_paths, CONTENT_IDS)}
    for key, run in (("dad", dad_run), ("sdf", sdf_run)):
        if not run:
            continue
        run = Path(run)
        out[f"{key}_audit"] = C.read_json(run / "audit" / "audit_report.json")
        out[f"{key}_diversity"] = C.read_json(run / "audit" / "diversity_report.json")
        out[f"{key}_manifest"] = C.read_json(run / "run_manifest.json")
        out[f"{key}_costs"] = C.read_jsonl(run / "cost_log.jsonl")
        out[f"{key}_run_id"] = run.name
    return out


def _card(*, kicker, prose_html, numbers, href=None, go="Read the report"):
    body = [f"<div class='card-k'>{R.esc(kicker)}</div>", prose_html]
    if numbers:
        body.append(f"<p class='card-n'>{numbers}</p>")
    body.append(f"<p class='card-go'><a href='{R.esc(href)}'>{R.esc(go)} &rarr;</a></p>"
                if href else f"<p class='card-n' style='border:0'>{R.esc(go)}</p>")
    return f"<div class='card{'' if href else ' soon'}'>{''.join(body)}</div>"


def _dad_numbers(audit, costs, manifest, diversity):
    """The DAD card's figures, computed from the same facts() the DAD page uses — so
    the hub and the report can never disagree about a headline."""
    if not audit:
        return ""
    from report import dad
    f = dad.facts(audit, manifest, diversity, costs)
    bits = [f"{f.get('n_measured', '?')} examples measured"]
    if f.get("lift_pct"):
        bits.append(f"+{f['lift_pct']} welfare considerations per answer")
    if f.get("delivery_pipeline"):
        bits.append(f"delivery {f['delivery_pipeline']} against the control's "
                    f"{f['delivery_plain']}")
    if f.get("cost_per_example") != "not logged":
        bits.append(f"{f['cost_per_example']} an example")
    return " · ".join(R.esc(b) for b in bits)


def _sdf_numbers(audit, diversity):
    """The SDF card's figures. Its audit has a different shape and no verdict rows, so
    this reads only the handful of fields that are stable across its runs."""
    if not audit:
        return ""
    bits = []
    n = audit.get("n_docs")
    if n:
        bits.append(f"{n} documents")
    comp = audit.get("composition") or {}
    if comp.get("languages"):
        bits.append(f"{len(comp['languages'])} languages")
    if comp.get("types"):
        bits.append(f"{len(comp['types'])} document types")
    nd = (audit.get("near_dups") or {}).get("0.9")
    if nd is not None:
        bits.append(f"{nd:.0%} near-duplicates")
    if (diversity or {}).get("vendi"):
        bits.append(f"{diversity['vendi'].get('score', 0):.0f} effectively distinct")
    return " · ".join(R.esc(b) for b in bits)


def _split_around_cards(html):
    """Split rendered prose into what goes above the cards and what goes below.

    The dek plus the first paragraph introduce the two corpora; everything after that is
    commentary on them, so the cards belong in between. Falls back to putting all of it
    above, which is how it renders if someone writes a single block.
    """
    parts = html.split("</p>")
    above = 0
    for i, part in enumerate(parts):
        above = i + 1
        if "class='dek'" not in part:
            break
    if above >= len(parts) - 1:
        return html, ""
    return "</p>".join(parts[:above]) + "</p>", "</p>".join(parts[above:])


def body(*, content, dad_audit=None, dad_costs=None, dad_manifest=None, dad_diversity=None,
         dad_run_id="", sdf_audit=None, sdf_diversity=None, sdf_run_id="", dad_href="dad.html",
         sdf_href=None):
    f = {}  # the landing page interpolates nothing: it has no run of its own to be stale
    dad_live = bool(dad_audit)
    # The hero is the two destinations. An unbuilt report keeps the row's shape without
    # being clickable; the card below it says the same thing in words.
    hero = R.buttons([
        ("The dilemma corpus", dad_href if dad_live else None,
         "" if dad_live else "no run built yet"),
        ("The document corpus", sdf_href, "" if sdf_href else "report in preparation"),
    ])
    cards = ("<div class='cards'>"
             + _card(kicker="Chat SFT · one dilemma, one answer",
                     prose_html=C.prose(content, "dad_card", f),
                     numbers=_dad_numbers(dad_audit, dad_costs, dad_manifest, dad_diversity),
                     href=dad_href if dad_live else None,
                     go="Read the report" if dad_live else "No run built yet")
             + _card(kicker="Pretraining documents · a world, depicted",
                     prose_html=C.prose(content, "sdf_card", f),
                     numbers=_sdf_numbers(sdf_audit, sdf_diversity),
                     href=sdf_href,
                     go="Read the report" if sdf_href else "Report in preparation")
             + "</div>")
    # The corpora section's prose is split around the cards: whatever precedes the first
    # blank line after the dek introduces them, and the rest (the third route we turned
    # down) reads better once the reader has seen what the two are.
    lead, rest = _split_around_cards(C.prose(content, "corpora", f))
    sections = [
        C.section("why", "Why this data is missing", C.prose(content, "why", f)),
        C.section("corpora", "The two corpora", lead, cards, rest),
        C.section("limits", "What this does not show", C.prose(content, "limits", f)),
    ]
    head = {
        "title": content["title"].strip(),
        "lede": content["lede"].strip(),
        "hero": hero,
        "meta_line": "",
        "footer": "Both pages are generated from their run's own audit output. No figure on "
                  "either is typed in by hand, and each page's weaknesses section is derived "
                  "from the audit's verdicts rather than written.",
    }
    return "".join(sections), TOC, head


def build(**kwargs):
    body_html, toc, head = body(**kwargs)
    return R.document(toc=toc, body=body_html, layout="landing", **head)
