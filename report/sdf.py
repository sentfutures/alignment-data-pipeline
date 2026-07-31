#!/usr/bin/env python3
"""The document corpus's section of the handoff page: the ``#sdf`` beats.

Deliberately small. The written report for this corpus is still being produced, so what
ships today is the anchor, the figures the comparison table needs, and links out. When
the full section lands it takes report/dad.py's skeleton — ``R.sub()`` beats named
``sdf-what``, ``sdf-built``, ``sdf-example``, ``sdf-weak``, ``sdf-appendix`` — and
everything below stays as the loader and the facts. Note the order: how it is built comes
before the example that walks through it, and there is no "what we measured" beat, because
neither report is a results report. Charts belong in the appendix.

One thing to know before writing it: ``evals/audit_sdf.py`` prints its verdicts and
does not record them into ``sections[].rows[]`` the way ``evals/audit_dad.py`` does, so
``common.audit_verdict_warnings()`` returns nothing here and this section will have to
derive its own thresholds. The ones the eval already uses, for reference: top document
type share (GOOD ≤ 0.15, OK ≤ 0.30), truncated fraction (GOOD 0, OK ≤ 0.02),
near-duplicates over 0.90 (GOOD ≤ 0.02, OK ≤ 0.08), formulaic openings (GOOD ≤ 0.15,
OK ≤ 0.35), and a scanned pattern flagged red when it is a defect above 0.30 prevalence.

stdlib only, and no imports from viewer/ or shared/.
"""

from report import common as C
from report import render as R

CONTENT_IDS = ("sdf_what", "sdf_soon")

SECTION_ID = "sdf"
SECTION_TITLE = "Synthetic documents"

_STAGE_KNOBS = ("plan_model", "draft_model", "rewrite_model", "score_model")


def load_inputs(run_dir):
    """All filesystem access, in one place. Returns the section's kwargs.

    A missing audit is not fatal here: the section and the comparison table's column
    both degrade to saying so.
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    return {
        "audit": C.read_json(run_dir / "audit" / "audit_report.json"),
        "diversity": C.read_json(run_dir / "audit" / "diversity_report.json"),
        "manifest": C.read_json(run_dir / "run_manifest.json"),
        "costs": C.read_jsonl(run_dir / "cost_log.jsonl"),
        "n_prompt_templates": C.prompt_count(run_dir, "layer*.txt"),
        "run_id": run_dir.name,
    }


def models(manifest):
    """Every model this run actually generated with, deduplicated."""
    cfg = (manifest or {}).get("config") or {}
    sdf = cfg.get("sdf") or {}
    glob = cfg.get("model")
    return sorted({(sdf.get(k) or glob) for k in _STAGE_KNOBS if (sdf.get(k) or glob)})


def facts(audit=None, diversity=None, manifest=None):
    """The handful of figures the comparison table and the placeholder section need.

    Reads the field names a real SDF audit actually writes: ``composition.language``
    and ``composition.n_types`` (an earlier version of this code read ``languages`` and
    ``types``, which no audit has ever produced, so both cells rendered empty).
    """
    comp = (audit or {}).get("composition") or {}
    f = {"n_docs": (audit or {}).get("n_docs")}
    if comp.get("language"):
        f["n_languages"] = len(comp["language"])
    if comp.get("n_types"):
        f["n_types"] = comp["n_types"]
    if comp.get("top_type_share") is not None:
        f["top_type_share"] = f"{comp['top_type_share']:.0%}"
    if (diversity or {}).get("vendi"):
        f["vendi"] = f"{diversity['vendi'].get('score', 0):.0f}"
    f["models"] = ", ".join(models(manifest))
    return f


def spread(f):
    """The comparison table's spread cell."""
    bits = []
    if f.get("n_languages"):
        bits.append(f"{f['n_languages']} languages")
    if f.get("n_types"):
        bits.append(f"{f['n_types']} document types")
    if f.get("top_type_share"):
        bits.append(f"largest type {f['top_type_share']}")
    return " · ".join(bits)


def _verdict(value, good, ok):
    """``evals/audit_sdf.py``'s own thresholds, mirrored. Lower is better for all of them."""
    return "GOOD" if value <= good else "OK" if value <= ok else "BAD"


def derived_warnings(audit, manifest, f):
    """The weaknesses floor for this dataset, computed rather than written.

    ``evals/audit_sdf.py`` prints its verdicts and does not record them, so
    ``common.audit_verdict_warnings()`` finds nothing in an SDF audit and the thresholds
    are re-applied here, matching the ones the eval itself uses. Only non-GOOD rows are
    emitted, and provenance is appended exactly as it is on the other section.
    """
    out = []
    length = (audit or {}).get("length") or {}
    frac = length.get("truncated_frac")
    if frac:
        out.append((_verdict(frac, 0.0, 0.02),
                    f"{frac:.0%} of documents are truncated ({length.get('truncated')} of "
                    f"{f.get('n_docs', '?')}), so those documents stop mid-thought."))
    checks = (
        ((audit or {}).get("composition") or {}).get("top_type_share"), 0.15, 0.30,
        "The largest document type is {v:.0%} of the dataset.",
    ), (
        ((audit or {}).get("near_dups") or {}).get("0.9"), 0.02, 0.08,
        "{v:.0%} of documents are near-duplicates of another, above 0.90 similarity.",
    ), (
        ((audit or {}).get("openings") or {}).get("formulaic_frac"), 0.15, 0.35,
        "{v:.0%} of documents open with a formulaic pattern.",
    )
    for value, good, ok, text in checks:
        if value is None:
            continue
        verdict = _verdict(value, good, ok)
        if verdict != "GOOD":
            out.append((verdict, text.format(v=value)))
    for pattern in (audit or {}).get("patterns") or []:
        if pattern.get("flagged"):
            out.append(("BAD", f"Templating scan: **{pattern.get('pattern')}** appears in "
                               f"{pattern.get('prevalence', 0):.0%} of documents and is judged "
                               f"a generator defect."))
    out += C.provenance_warnings(manifest, n=f.get("n_docs"))
    return sorted(out, key=lambda w: 0 if w[0] == "BAD" else 1)


def blocks(*, content, f, run_id="", audit=None, diversity=None, manifest=None,
           hf_href="", repo_href=""):
    """The placeholder report: what the dataset is, its measured scale, and links out.

    No worked example and no measurement beats yet — those wait for the full report.
    What is here is computed from the run like everything else on the page, so the
    report cannot drift from the dataset it describes.
    """
    out = [R.sub("sdf-what", "What it is"), C.prose(content, "sdf_what", f)]
    tiles = []
    if f.get("n_docs"):
        tiles.append(R.stat(f"{f['n_docs']:,}", "documents",
                            f"in run {run_id}" if run_id else ""))
    if f.get("n_languages"):
        tiles.append(R.stat(str(f["n_languages"]), "languages",
                            f"across {f.get('n_types', '?')} document types, "
                            f"the largest of them {f.get('top_type_share', '?')} of the dataset"))
    if f.get("vendi"):
        nn = (diversity or {}).get("nn") or {}
        tiles.append(R.stat(f["vendi"], "effectively distinct documents",
                            f"a diversity score that counts near-duplicates as fractions of a "
                            f"document; {nn.get('over_0.90', 0):.0%} sit above 0.90 cosine "
                            f"similarity to their nearest neighbour"))
    if tiles:
        out.append(R.tiles(tiles))
    elif audit is None:
        out.append(R.note("No audit output was supplied for this dataset, so no figure here is "
                          "measured. Build with `--sdf-run <run directory>`."))
    warnings = derived_warnings(audit, manifest, f)
    if warnings:
        out.append(R.sub("sdf-weak", "Where it is weak"))
        out.append("<p class='muted'>Derived from this run's own audit output. The rest of the "
                   "dataset-level checks arrive with the full report.</p>")
        out.append(C.warnings_table(warnings))
    out.append(C.prose(content, "sdf_soon", f))
    links = []
    if hf_href:
        links.append(R.linkbutton(hf_href, "Browse the records", "hf", meta="dataset viewer"))
    if repo_href:
        links.append(R.linkbutton(repo_href, "The pipeline", "github",
                                  meta="and this run's audit output"))
    if links:
        out.append(f"<div class='lbtns'>{''.join(links)}</div>")
    return "".join(out)
