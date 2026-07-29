#!/usr/bin/env python3
"""Build the standalone HTML story report for a DAD run.

The audience is a technical reader at another lab — someone deciding whether the
method and its measurement are sound, and whether to run the pipeline themselves.
That is a different job from the Streamlit corpus-audit page, which is organised
by what the eval measured; this is organised by what a reader needs to believe,
in order: the problem, a worked example, how it is built, how it is measured,
what the numbers say, what the data would teach a model, where it is weak, and
how to reproduce it.

Two rules make the artefact trustworthy, and both are enforced here rather than
left to an author's discipline:

  1. No number is ever typed into the prose. ``content.md`` may interpolate
     ``{{placeholders}}``, which resolve against facts computed from the run's
     own audit JSON. An unresolved placeholder is a build error, so prose cannot
     silently go stale against the data.

  2. The weaknesses section is DERIVED, not written. Every BAD/OK verdict in the
     audit, plus a fixed set of provenance rules (non-faithful backend, dirty
     git tree, length inflation, unmeasured sections, arm-size asymmetry), emits
     its own line whether or not anyone remembered to write it up. Editorial
     prose adds to that floor; it cannot replace it.

Output is one self-contained HTML file (see report/render.py). stdlib only, and
deliberately no imports from viewer/ or shared/: the report has to build in an
environment where the pipeline's own dependencies are not installed, which is
also what makes it portable enough to hand to someone else.

Usage:
    python report/build_report.py --run outputs/dad/runs/<run_id> \\
        [--content report/content.md] [--out report/dad_report.html] [--example AW-0007]
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from report import render as R  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Section ids the builder expects to find in content.md. A missing id, or an id
# in the file the builder doesn't know, is an error — a typo must never silently
# drop a section from the report.
CONTENT_IDS = (
    "title", "subtitle", "problem", "example_pick", "example_intro", "method_intro",
    "stage1", "stage2", "stage3", "control", "measurement_intro", "judge_limits",
    "results_intro", "footprint_intro", "weaknesses_intro", "reproduce",
)

TOC = [
    ("summary", "At a glance"),
    ("problem", "1 The problem"),
    ("example", "2 What it produces"),
    ("method", "3 How it is built"),
    ("measurement", "4 How it is measured"),
    ("results", "5 What the numbers say"),
    ("footprint", "6 What it would teach a model"),
    ("weaknesses", "7 Where it is weak"),
    ("checks", "8 Every check"),
    ("reproduce", "9 Run it yourself"),
]

_STAGE_KNOBS = ("scenario_model", "prompt_draft_model", "prompt_gate_model",
                "prompt_refine_model", "response_scope_model", "response_select_model",
                "response_draft_model", "constitution_rewrite_model")

# stage tag in cost_log.jsonl -> display name, in pipeline order.
_STAGE_LABELS = (
    ("scenario_plan", "1a · scenario plan"),
    ("prompt_draft", "1b · prompt draft"),
    ("prompt_gate", "1c · quality gate"),
    ("prompt_refine", "1d · refine"),
    ("baseline", "control · plain model"),
    ("response_scope", "2a · scope"),
    ("response_select", "2a.5 · library select"),
    ("response_draft", "2b · response draft"),
    ("constitution_rewrite", "3 · constitution rewrite"),
)


# ------------------------------------------------------------------ loading

def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _jsonl(path):
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


def parse_content(text):
    """content.md -> {section_id: markdown}. Sections are delimited by
    ``<!-- id: name -->``. Unknown or missing ids raise, so a typo is loud."""
    parts = re.split(r"<!--\s*id:\s*([a-z0-9_]+)\s*-->", text)
    if len(parts) < 3:
        raise ValueError("content file has no '<!-- id: ... -->' section markers")
    found = {}
    for i in range(1, len(parts), 2):
        found[parts[i]] = parts[i + 1].strip()
    unknown = sorted(set(found) - set(CONTENT_IDS))
    if unknown:
        raise ValueError(f"content file has unknown section id(s): {', '.join(unknown)}")
    missing = sorted(set(CONTENT_IDS) - set(found))
    if missing:
        raise ValueError(f"content file is missing section id(s): {', '.join(missing)}")
    return found


def load_inputs(run_dir, content_path=None):
    """All filesystem access, in one place. Returns build_report() kwargs."""
    run_dir = Path(run_dir)
    audit = _json(run_dir / "audit" / "audit_report.json")
    if audit is None:
        raise SystemExit(f"No audit report at {run_dir / 'audit' / 'audit_report.json'} — "
                         f"run: python evals/audit_dad.py --input {run_dir} --reasons")
    content_path = Path(content_path or (REPO_ROOT / "report" / "content.md"))
    return {
        "audit": audit,
        "diversity": _json(run_dir / "audit" / "diversity_report.json"),
        "manifest": _json(run_dir / "run_manifest.json"),
        "content": parse_content(content_path.read_text(encoding="utf-8")),
        "corpus": _jsonl(run_dir / "final" / "dad_corpus.jsonl"),
        "baseline": _jsonl(run_dir / "baseline" / "baseline_responses.jsonl"),
        "rewrites": _jsonl(run_dir / "step3" / "rewrites.jsonl"),
        "costs": _jsonl(run_dir / "cost_log.jsonl"),
        "run_id": run_dir.name,
    }


# ------------------------------------------------------------------ facts

def _considerations(audit):
    """The headline pair, from either schema.

    Modern reports carry ``valuable_welfare_considerations``; older ones are
    reconstructed from ``moral_patient_reasons`` + ``moves.alternatives`` exactly
    as evals/audit_dad.py's own legacy branch does, so a pre-merge run still
    renders its headline instead of showing a hole.
    """
    vwc = audit.get("valuable_welfare_considerations") or {}
    if vwc.get("available") and vwc.get("parent"):
        subs = {s["name"]: s for s in (vwc.get("subsets") or [])}
        return {
            "pipeline": vwc["parent"].get("pipeline"),
            "plain": vwc["parent"].get("plain"),
            "subsets": [(name, s.get("plain"), s.get("pipeline")) for name, s in subs.items()],
            "source": "modern",
        }
    mpr = audit.get("moral_patient_reasons") or {}
    pipe, plain = mpr.get("pipeline") or {}, mpr.get("plain") or {}
    if not pipe:
        return None
    alts = (audit.get("moves") or {}).get("alternatives") or {}
    reasoning_p, reasoning_b = pipe.get("mean_unique"), plain.get("mean_unique")
    alt_p, alt_b = alts.get("pipeline_mean"), alts.get("plain_mean")
    if reasoning_p is None:
        return None
    return {
        "pipeline": reasoning_p + (alt_p or 0),
        "plain": (reasoning_b or 0) + (alt_b or 0) if reasoning_b is not None else None,
        "subsets": [("welfare reasoning", reasoning_b, reasoning_p),
                    ("humane alternatives", alt_b, alt_p)] if alt_p is not None else
                   [("welfare reasoning", reasoning_b, reasoning_p)],
        "source": "reconstructed",
    }


def _models(manifest):
    cfg = (manifest or {}).get("config") or {}
    dad = cfg.get("dad") or {}
    glob = cfg.get("model")
    used = sorted({(dad.get(k) or glob) for k in _STAGE_KNOBS if (dad.get(k) or glob)})
    return {
        "stage_models": used,
        "global": glob,
        "backend": cfg.get("backend"),
        "per_stage": {k: (dad.get(k) or glob) for k in _STAGE_KNOBS},
    }


def _costs_by_stage(costs):
    agg = {}
    for rec in costs or []:
        stage = rec.get("stage") or "(untagged)"
        entry = agg.setdefault(stage, {"calls": 0, "cost": 0.0, "models": set()})
        entry["calls"] += 1
        entry["cost"] += rec.get("cost_usd") or 0.0
        if rec.get("model"):
            entry["models"].add(rec["model"])
    return agg


def facts(audit, manifest=None, diversity=None, costs=None):
    """Every number the prose can interpolate, computed once, in one place."""
    mpr = audit.get("moral_patient_reasons") or {}
    surv = mpr.get("survival") or {}
    rl = audit.get("response_lengths") or {}
    cons = _considerations(audit)
    models = _models(manifest)
    cost_agg = _costs_by_stage(costs)
    total_cost = sum(v["cost"] for v in cost_agg.values())
    n = audit.get("n_prompts") or 0
    anchored = (surv.get("kept") or 0) + (surv.get("weakened") or 0) + (surv.get("dropped") or 0)
    f = {
        "n": n,
        "n_pipeline": (mpr.get("pipeline") or {}).get("n"),
        "n_plain": (mpr.get("plain") or {}).get("n"),
        "extraction_failures": mpr.get("failures"),
        "judge_model": (audit.get("delivery") or {}).get("model") or mpr.get("model") or "?",
        "gen_models": ", ".join(models["stage_models"]) or "?",
        "backend": models["backend"] or "?",
        "cost_total": f"${total_cost:,.2f}" if total_cost else None,
        "cost_per_example": f"${total_cost / n:,.2f}" if total_cost and n else None,
    }
    if cons and cons.get("plain"):
        f["considerations_pipeline"] = f"{cons['pipeline']:.1f}"
        f["considerations_plain"] = f"{cons['plain']:.1f}"
        f["lift_pct"] = f"{(cons['pipeline'] / cons['plain'] - 1) * 100:.0f}%"
    if anchored:
        f["retention_pct"] = f"{(surv.get('kept', 0) + surv.get('weakened', 0)) / anchored:.0%}"
        f["dropped_n"] = surv.get("dropped")
        f["added_total"] = surv.get("added_total")
    if rl.get("mean_ratio"):
        f["length_ratio"] = f"{rl['mean_ratio']:.2f}"
        f["length_pct"] = f"{(rl['mean_ratio'] - 1) * 100:.0f}%"
    stance = (audit.get("moves") or {}).get("stance") or {}
    if stance.get("pipeline"):
        f["moralizes_pipeline"] = f"{stance['pipeline'].get('moralizes', 0):.0%}"
        f["moralizes_plain"] = f"{(stance.get('plain') or {}).get('moralizes', 0):.0%}"
    if diversity:
        vendi = diversity.get("vendi") or {}
        f["vendi"] = f"{vendi.get('score', 0):.1f}"
        f["vendi_ratio"] = f"{vendi.get('ratio', 0):.2f}"
    f = {k: v for k, v in f.items() if v is not None}
    # Prose may only interpolate facts that EVERY run has, so a run missing the
    # paid pass degrades its charts rather than failing the build. Conditional
    # numbers belong in the sections that own them, not in the narrative.
    for key, default in (("cost_total", "not logged"), ("cost_per_example", "not logged")):
        f.setdefault(key, default)
    return f


_PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+)\}\}")


def fill(text, f):
    """Resolve {{placeholders}} from the facts dict. Unknown key -> build error."""
    def sub(m):
        key = m.group(1)
        if key not in f:
            raise KeyError(f"content.md references unknown fact '{{{{{key}}}}}' "
                           f"(available: {', '.join(sorted(f))})")
        return str(f[key])
    return _PLACEHOLDER.sub(sub, text or "")


# ------------------------------------------------------------------ helpers

def _labels(audit):
    """prompt_id -> stable display id (response gid), from the report's own map."""
    out = {}
    for pid, gids in (audit.get("gid_map") or {}).items():
        out[pid] = (gids or {}).get("response") or pid
    return out


def _prose(content, key, f):
    return R.paragraphs(fill(content.get(key, ""), f))


def _example_pick(content):
    """The prompt_id pinned in content.md, or None for automatic selection.

    Pinned in the prose file rather than passed on the command line so a rebuild
    reproduces the same worked example without anyone having to remember a flag.
    """
    raw = (content.get("example_pick") or "").strip()
    return None if raw.lower() in ("", "auto") else raw.split()[0]


def _section(sid, heading, *blocks):
    body = "".join(b for b in blocks if b)
    return f"<section id='{sid}'><h2>{R.esc(heading)}</h2>{body}</section>"


# ------------------------------------------------------------------ sections

def section_summary(audit, f, cons):
    items = []
    if cons and cons.get("plain"):
        items.append(R.stat(f"+{f.get('lift_pct', '?')}",
                            "more valuable welfare considerations per answer",
                            f"pipeline {cons['pipeline']:.1f} vs plain {cons['plain']:.1f}",
                            tone="good"))
    delivery = audit.get("delivery") or {}
    if delivery.get("pipeline_mean") is not None:
        pm, bm = delivery["pipeline_mean"], delivery.get("plain_mean")
        items.append(R.stat(f"{pm * 10:.0f}%", "delivery quality (helpful, unobtrusive)",
                            f"plain {bm * 10:.0f}% — the pipeline is worse here"
                            if bm is not None and pm < bm
                            else f"plain {bm * 10:.0f}%" if bm is not None else "",
                            tone="good" if bm is None or pm >= bm else "bad"))
    if "retention_pct" in f:
        items.append(R.stat(f["retention_pct"], "of plain's considerations retained",
                            f"{f.get('dropped_n', 0)} dropped · {f.get('added_total', 0)} added"))
    if "length_pct" in f:
        items.append(R.stat(f"+{f['length_pct']}", "longer than the plain-model control",
                            "length is the most visible thing this data teaches", tone="warn"))
    if "moralizes_pipeline" in f:
        items.append(R.stat(f["moralizes_pipeline"], "of pipeline answers moralize",
                            f"plain {f['moralizes_plain']} — the pipeline's one failing check",
                            tone="bad"))
    return R.tiles(items)


def section_example(audit, content, f, corpus, baseline, rewrites, labels, pick=None):
    """One full dilemma, plain vs pipeline. Judge-selected when the paid showcase
    pass has run; otherwise the record with the most pipeline-added considerations,
    labelled as mechanically chosen so the reader knows which they are reading."""
    blocks = [_prose(content, "example_intro", f)]
    showcase = (audit.get("showcase") or {}).get("examples") or []
    by_pid_base = {r.get("prompt_id"): r for r in baseline or []}
    by_pid_rw = {r.get("prompt_id"): r for r in rewrites or []}

    chosen = None
    pick = pick or _example_pick(content)
    if pick:
        chosen = {"prompt_id": pick, "label": "A worked example", "summary": "",
                  "highlights": [],
                  "provenance": "chosen by hand for this report from the run's records, and "
                                "pinned in the prose file so a rebuild shows the same case"}
    elif showcase:
        ex = showcase[0]
        chosen = {"prompt_id": ex.get("prompt_id"), "label": ex.get("label", "Showcase"),
                  "summary": ex.get("summary", ""), "highlights": ex.get("highlights") or [],
                  "user_message": ex.get("user_message"), "plain": ex.get("plain_response"),
                  "pipeline": ex.get("pipeline_response"),
                  "provenance": "selected by the showcase judge from the retention and "
                                "delivery data; the highlighted spans are the judge's, "
                                "validated against the response text"}
    else:
        per_case = (audit.get("moral_patient_reasons") or {}).get("per_case") or {}
        ranked = sorted(per_case.items(),
                        key=lambda kv: -len(((kv[1].get("survival") or {}).get("added") or [])))
        if ranked:
            pid = ranked[0][0]
            chosen = {"prompt_id": pid, "label": "Most pipeline-added considerations",
                      "summary": "", "highlights": [],
                      "provenance": "selected mechanically (the record where the pipeline "
                                    "added the most considerations beyond the control); no "
                                    "judge-selected showcase exists for this run"}

    if not chosen or not chosen.get("prompt_id"):
        blocks.append(R.note("No worked example could be built: this run has neither a showcase "
                             "pass nor per-record retention data."))
        return _section("example", "2 · What it produces", *blocks)

    pid = chosen["prompt_id"]
    base = by_pid_base.get(pid) or {}
    rw = by_pid_rw.get(pid) or {}
    user_msg = chosen.get("user_message") or base.get("user_message") or rw.get("user_message", "")
    plain = chosen.get("plain") or base.get("baseline_response", "")
    pipeline = chosen.get("pipeline") or rw.get("rewritten_response", "")
    if not (user_msg and pipeline):
        blocks.append(R.note(f"Worked example {R.esc(pid)} could not be assembled from this "
                             "run's files."))
        return _section("example", "2 · What it produces", *blocks)

    gid = labels.get(pid, pid)
    blocks.append(f"<h3>{R.esc(chosen['label'])} — <span class='mono'>{R.esc(gid)}</span></h3>")
    if chosen.get("summary"):
        blocks.append(f"<p>{R.inline_md(chosen['summary'])}</p>")
    blocks.append(f"<p class='muted'>How this example was chosen: {R.esc(chosen['provenance'])}.</p>")
    blocks.append("<h4>The user asked</h4>")
    blocks.append(R.quote(user_msg))
    blocks.append(R.sidebyside(
        "Plain model, no system prompt (control)", R.highlight(plain, []),
        "Pipeline", R.highlight(pipeline, chosen.get("highlights"))))

    per_case = ((audit.get("moral_patient_reasons") or {}).get("per_case") or {}).get(pid) or {}
    surv = per_case.get("survival") or {}
    if surv:
        kept = [a["reason"] for a in (surv.get("anchored") or []) if a.get("verdict") == "kept"]
        weak = [a["reason"] for a in (surv.get("anchored") or []) if a.get("verdict") == "weakened"]
        drop = [a["reason"] for a in (surv.get("anchored") or []) if a.get("verdict") == "dropped"]
        added = surv.get("added") or []
        rows = [("kept from the control", len(kept), "; ".join(kept[:3])),
                ("weakened", len(weak), "; ".join(weak[:3])),
                ("dropped", len(drop), "; ".join(drop[:3])),
                ("added by the pipeline", len(added), "; ".join(added[:3]))]
        blocks.append(R.details(
            "What the retention judge found in this example",
            R.table(["fate", "n", "examples"], rows)))

    if rw.get("draft_response") and rw.get("rewritten_response"):
        blocks.append(R.details(
            "What the constitution rewrite (stage 3) changed in this answer",
            "<p class='muted'>Stage 2's draft on the left, the shipped answer on the right. "
            "These two ARE a revision of one another, so the differences are meaningful; "
            "the control and the pipeline answer are independently generated, so diffing "
            "those would be noise.</p>"
            + _word_diff(rw["draft_response"], rw["rewritten_response"])))
    return _section("example", "2 · What it produces", *blocks)


def _word_diff(before, after):
    """Word-level diff of the stage-2 draft against the shipped answer."""
    a, b = before.split(), after.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            out.append(R.esc(" ".join(b[j1:j2])))
        elif tag in ("replace", "insert"):
            if tag == "replace":
                out.append(f"<del>{R.esc(' '.join(a[i1:i2]))}</del>")
            out.append(f"<ins>{R.esc(' '.join(b[j1:j2]))}</ins>")
        elif tag == "delete":
            out.append(f"<del>{R.esc(' '.join(a[i1:i2]))}</del>")
    return ("<style>ins{background:var(--mark);text-decoration:none}"
            "del{opacity:.55;text-decoration:line-through}</style>"
            f"<div class='resp'>{' '.join(out)}</div>")


def section_method(content, f, manifest, costs):
    blocks = [_prose(content, "method_intro", f)]
    for key, heading in (("stage1", "Stage 1 · the dilemma"),
                         ("stage2", "Stage 2 · the reasoning"),
                         ("stage3", "Stage 3 · the constitution rewrite"),
                         ("control", "The control arm")):
        blocks.append(f"<h3>{R.esc(heading)}</h3>{_prose(content, key, f)}")
    models = _models(manifest)
    agg = _costs_by_stage(costs)
    if agg:
        rows = []
        for tag, label in _STAGE_LABELS:
            entry = agg.get(tag)
            if not entry:
                continue
            rows.append((label, ", ".join(sorted(entry["models"])) or "—",
                         entry["calls"], f"${entry['cost']:,.2f}"))
        other = sorted(set(agg) - {t for t, _ in _STAGE_LABELS})
        for tag in other:
            rows.append((tag, ", ".join(sorted(agg[tag]["models"])) or "—",
                         agg[tag]["calls"], f"${agg[tag]['cost']:,.2f}"))
        blocks.append("<h4>What each stage cost on this run</h4>")
        blocks.append(R.table(["stage", "model", "calls", "cost"], rows))
    elif models["stage_models"]:
        blocks.append(R.table(["stage", "model"],
                              [(k.replace("_model", "").replace("_", " "), v)
                               for k, v in models["per_stage"].items()]))
    return _section("method", "3 · How it is built", *blocks)


def section_measurement(audit, content, f, diversity):
    blocks = [_prose(content, "measurement_intro", f)]
    mpr = audit.get("moral_patient_reasons") or {}
    checks = [
        ("Valuable welfare considerations", bool(mpr.get("pipeline")),
         "Distinct welfare points and concrete lower-harm actions per answer, both arms",
         mpr.get("model")),
        ("Retention / survival", bool(mpr.get("survival")),
         "Item by item, which of the control's considerations the pipeline kept, weakened, "
         "dropped, and what it added", mpr.get("model")),
        ("Delivery quality", bool(audit.get("delivery")),
         "How helpful, unobtrusive and non-preachy each answer is, scored 0-10",
         (audit.get("delivery") or {}).get("model")),
        ("Showcase examples", bool(audit.get("showcase")),
         "Concrete pipeline-beats-plain cases with verbatim improved spans",
         (audit.get("showcase") or {}).get("model")),
        ("Response stance", bool((audit.get("moves") or {}).get("stance")),
         "Whether an answer defers, stays calibrated, or moralizes", mpr.get("model")),
        ("Tracked tics / rhetorical moves", bool(audit.get("tracked_tics")
                                                 or audit.get("rhetorical_moves")),
         "Recurring phrasing and argumentative habits, as a share of each arm", None),
        ("Response lengths, structure, jargon", bool(audit.get("response_lengths")),
         "Offline corpus measurements against the control", None),
        ("Semantic diversity", bool(diversity),
         "Embedding near-duplicate rate, topic spread, Vendi effective count",
         (diversity or {}).get("embed_model")),
    ]
    rows = [(name, "measured" if ok else "not run on this run", what, model or "offline")
            for name, ok, what, model in checks]
    blocks.append(R.table(["check", "status", "what it establishes", "model"], rows))
    blocks.append(_prose(content, "judge_limits", f))
    return _section("measurement", "4 · How it is measured", *blocks)


def section_results(audit, content, f, cons, labels, diversity):
    blocks = [_prose(content, "results_intro", f)]
    if cons and cons.get("plain") is not None:
        blocks.append("<h3>Substance: valuable welfare considerations per answer</h3>")
        blocks.append(R.hbar([("plain model (control)", round(cons["plain"], 2)),
                              ("pipeline", round(cons["pipeline"], 2))],
                             color=None, fmt="{:.1f}"))
        subset_rows = [{"label": name, "plain Claude": b, "pipeline": p}
                       for name, b, p in cons["subsets"] if p is not None]
        if subset_rows:
            blocks.append("<h4>Split by kind</h4>")
            blocks.append(R.grouped_hbar(
                subset_rows, series=[("plain Claude", R.PLAIN), ("pipeline", R.PIPELINE)],
                fmt="{:.2f}"))
        if cons["source"] == "reconstructed":
            blocks.append("<p class='muted'>Reconstructed from this run's separate reasoning "
                          "and alternatives measures — it predates the unified extraction.</p>")

    mpr = audit.get("moral_patient_reasons") or {}
    if mpr.get("failures"):
        blocks.append(R.note(
            f"Means are over pipeline {f.get('n_pipeline', '?')} / plain "
            f"{f.get('n_plain', '?')} answers: {mpr['failures']} extractions failed and are "
            "excluded. This is not a fully matched comparison, and a missing answer is a gap, "
            "not a zero."))

    per_case = mpr.get("per_case") or {}
    if per_case:
        blocks.append("<h3>What happened to the control's considerations, record by record</h3>")
        blocks.append(_survival_chart(per_case, labels))

    types_p = (mpr.get("pipeline") or {}).get("type_hist") or _type_hist(per_case, "pipeline")
    types_b = (mpr.get("plain") or {}).get("type_hist") or _type_hist(per_case, "plain")
    if types_p and types_b:
        blocks.append("<h3>Not just more points — different kinds of point</h3>")
        keys = [k for k in dict(types_p, **types_b)]
        rows = [{"label": k, "plain Claude": types_b.get(k, 0), "pipeline": types_p.get(k, 0)}
                for k in keys]
        blocks.append(R.grouped_hbar(rows, series=[("plain Claude", R.PLAIN),
                                                   ("pipeline", R.PIPELINE)]))

    delivery = audit.get("delivery") or {}
    blocks.append("<h3>Manner: delivery quality</h3>")
    if delivery.get("per_case"):
        pm, bm = delivery.get("pipeline_mean"), delivery.get("plain_mean")
        blocks.append(f"<p>Mean delivery quality: pipeline <b>{pm * 10:.0f}%</b>"
                      + (f" vs plain <b>{bm * 10:.0f}%</b>" if bm is not None else "") + ".</p>")
        if bm is not None and pm < bm:
            blocks.append(R.note(
                "This is the wrong direction, and it is the most important number on this page "
                "after the substance gain. The extra welfare substance was not free: judged on "
                "manner alone, the control answers read as more helpful and less obtrusive than "
                "the pipeline's. Any claim that the pipeline adds substance *without* costing "
                "delivery is not supported by this run.", tone="bad"))
        blocks.append(_pareto(delivery, mpr, labels))
        dims = delivery.get("dimensions") or {}
        if dims.get("pipeline"):
            keys = [k for k in ("goal_responsiveness", "proportionality", "tone", "calibration")
                    if k in dims["pipeline"]]
            rows = [(arm, *[f"{dims[arm][k] * 10:.0f}%" if dims.get(arm, {}).get(k) is not None
                            else "—" for k in keys])
                    for arm in ("pipeline", "plain") if dims.get(arm)]
            blocks.append(R.table(["arm"] + [k.replace("_", " ") for k in keys], rows))
    else:
        blocks.append(R.note(
            "Delivery quality was **not measured on this run**. The pipeline's eval suite scores "
            "it — this run predates that pass. Populate it with "
            "`python evals/audit_dad.py --input <run> --reasons`. Without it there is no evidence "
            "here that the added substance did not cost manner."))

    if diversity:
        blocks.append("<h3>Corpus diversity</h3>")
        vendi = diversity.get("vendi") or {}
        nn = diversity.get("nn") or {}
        combined = ((diversity.get("scopes") or {}).get("combined") or {})
        clusters = combined.get("clusters") or {}
        blocks.append(R.tiles([
            R.stat(f"{vendi.get('score', 0):.1f}", "effectively distinct records (Vendi)",
                   f"of {diversity.get('n_records', '?')} · ratio {vendi.get('ratio', 0):.2f}"),
            R.stat(f"{nn.get('over_0.90', 0):.0%}", "near-duplicate records (>0.90)",
                   f"{nn.get('over_0.80', 0):.0%} above 0.80"),
            R.stat(f"{clusters.get('evenness', 0):.2f}", "topic-spread evenness",
                   f"largest cluster holds {clusters.get('largest_share', 0):.0%}"),
        ]))
    return _section("results", "5 · What the numbers say", *blocks)


def _type_hist(per_case, arm):
    out = {}
    for case in (per_case or {}).values():
        for k, v in ((case.get(arm) or {}).get("type_hist") or {}).items():
            out[k] = out.get(k, 0) + v
    return out


_SURVIVAL_CATS = (("dropped", "var(--series-8)"), ("weakened", "var(--series-4)"),
                  ("kept", "var(--series-2)"), ("added", "var(--series-3)"))


def _survival_chart(per_case, labels):
    rows = []
    for pid in sorted(per_case):
        surv = (per_case[pid] or {}).get("survival") or {}
        anchored = surv.get("anchored") or []
        if not anchored and not surv.get("added"):
            continue
        seg = {"kept": 0, "weakened": 0, "dropped": 0,
               "added": len(surv.get("added") or [])}
        for a in anchored:
            if a.get("verdict") in seg:
                seg[a["verdict"]] += 1
        rows.append({"label": labels.get(pid, pid), "segments": seg,
                     "tips": {k: f"{labels.get(pid, pid)} — {k}: {v}" for k, v in seg.items()}})
    return (R.stacked_bar(rows, categories=list(_SURVIVAL_CATS),
                          ylabel="considerations", xlabel="one column per record")
            + "<p class='muted'>The lower three segments are the control's considerations and "
              "their fate; the top segment is what the pipeline added beyond them.</p>")


def _pareto(delivery, mpr, labels):
    per_d = delivery.get("per_case") or {}
    per_r = mpr.get("per_case") or {}
    pts, sums = [], {"plain": [0, 0, 0], "pipeline": [0, 0, 0]}
    for pid, entry in per_d.items():
        for arm in ("plain", "pipeline"):
            score = (entry.get(arm) or {}).get("score")
            reasons = ((per_r.get(pid) or {}).get(arm) or {}).get("reasons")
            if score is None or reasons is None:
                continue
            y = len(reasons)
            pts.append({"x": score, "y": y, "color": R.ARM_COLORS[arm],
                        "tip": f"{labels.get(pid, pid)} · {arm}: {y} considerations, "
                               f"delivery {score}/10"})
            sums[arm][0] += score
            sums[arm][1] += y
            sums[arm][2] += 1
    marks = [{"x": s[0] / s[2], "y": s[1] / s[2], "color": R.ARM_COLORS[arm],
              "tip": f"{arm} mean: delivery {s[0] / s[2]:.1f}/10, {s[1] / s[2]:.1f} considerations"}
             for arm, s in sums.items() if s[2]]
    return (R.scatter(pts, xlabel="delivery quality (0-10)",
                      ylabel="valuable welfare considerations", xdomain=(0, 10), marks=marks)
            + "<p class='muted'>Each dot is one answer: manner on the x-axis, substance on the "
              "y-axis. Up and to the right is the goal — more substance without losing manner. "
              "Diamonds are each arm's mean.</p>")


def section_footprint(audit, content, f):
    blocks = [_prose(content, "footprint_intro", f)]
    rl = audit.get("response_lengths") or {}
    if rl.get("pipeline_mean"):
        blocks.append("<h3>Length</h3>")
        blocks.append(R.hbar([("plain model (control)", round(rl.get("plain_mean", 0))),
                              ("pipeline", round(rl["pipeline_mean"]))],
                             unit=" chars", fmt="{:,.0f}"))
    stance = (audit.get("moves") or {}).get("stance") or {}
    if stance.get("pipeline"):
        blocks.append("<h3>Stance — including the check this run fails</h3>")
        rows = [{"label": k, "plain Claude": (stance.get("plain") or {}).get(k),
                 "pipeline": stance["pipeline"].get(k)}
                for k in ("defers", "calibrated", "moralizes")
                if stance["pipeline"].get(k) is not None]
        blocks.append(R.grouped_hbar(rows, series=[("plain Claude", R.PLAIN),
                                                   ("pipeline", R.PIPELINE)], percent=True))
        blocks.append(R.note(
            f"Moralizing is a fault, and the pipeline is worse than the control on it: "
            f"{f.get('moralizes_pipeline', '?')} of pipeline answers versus "
            f"{f.get('moralizes_plain', '?')} of control answers. Stage 3 is supposed to prevent "
            "exactly this. It is the pipeline's one failing check on this run and it is not "
            "explained away here.", tone="bad"))
    tics = audit.get("tracked_tics") or audit.get("stock_phrases") or {}
    watch = tics.get("watch") or {}
    n_pipe, n_plain = tics.get("n_pipeline") or 0, tics.get("n_plain") or 0
    if watch and n_pipe:
        rows = sorted(({"label": phrase,
                        "plain Claude": (d.get("plain") or 0) / n_plain if n_plain else 0,
                        "pipeline": (d.get("pipeline") or 0) / n_pipe}
                       for phrase, d in watch.items()
                       if (d.get("pipeline") or d.get("plain"))),
                      key=lambda r: -r["pipeline"])[:10]
        if rows:
            blocks.append("<h3>Tracked phrases</h3>")
            blocks.append(R.grouped_hbar(
                rows, series=[("plain Claude", R.PLAIN), ("pipeline", R.PIPELINE)],
                percent=True, rule=0.40, rule_label="flag line · 40%", label_w=200))
    moves = (audit.get("rhetorical_moves") or {}).get("moves") or {}
    if moves:
        rows = sorted(({"label": name, "plain Claude": d.get("plain_share"),
                        "pipeline": d.get("pipeline_share")} for name, d in moves.items()),
                      key=lambda r: -(r["pipeline"] or 0))
        blocks.append("<h3>Rhetorical habits</h3>")
        blocks.append(R.grouped_hbar(rows, series=[("plain Claude", R.PLAIN),
                                                   ("pipeline", R.PIPELINE)],
                                     percent=True, rule=0.50, rule_label="flag line · 50%",
                                     label_w=200))
        blocks.append(R.details("What each move is", R.table(
            ["move", "what it is"],
            [(name, (d.get("description") or "")) for name, d in moves.items()])))
    structure = audit.get("structure") or {}
    if (structure.get("pipeline") or {}).get("effective_shapes") is not None:
        p = structure["pipeline"]
        b = structure.get("plain") or {}
        blocks.append("<h3>Structural variety</h3>")
        blocks.append(R.hbar([("plain model (control)", b.get("effective_shapes", 0)),
                              ("pipeline", p.get("effective_shapes", 0))], fmt="{:.1f}"))
        if b.get("effective_shapes") and p["effective_shapes"] < b["effective_shapes"]:
            blocks.append(R.note(
                "The pipeline's answers are **less** structurally varied than the control's "
                f"({p['effective_shapes']:.1f} vs {b['effective_shapes']:.1f} effective shapes). "
                "More substance came at some cost in format variety."))
    return _section("footprint", "6 · What this data would teach a model", *blocks)


def derived_warnings(audit, manifest, f):
    """The weaknesses floor: computed from the run, never author-supplied.

    Anything BAD or OK in the audit, plus a fixed set of provenance rules. If a
    future run regresses, its warning appears here whether or not the prose was
    updated.
    """
    out = []
    for sec in audit.get("sections") or []:
        for row in sec.get("rows") or []:
            if row.get("verdict") in ("BAD", "OK"):
                out.append((row["verdict"], f"{sec.get('title', '?')} — "
                                            f"{row.get('label', '')}: {row.get('value', '')}"
                                            + (f" {row.get('note')}" if row.get("note") else "")))
    cfg = (manifest or {}).get("config") or {}
    if cfg.get("backend") and cfg["backend"] != "api":
        out.append(("BAD" if cfg["backend"] == "claude_code" else "OK",
                    f"This run was generated on the `{cfg['backend']}` backend, not `api`. "
                    "`api` is the documented faithful mode — the environment the spec's consumer "
                    "runs in. Treat these numbers as representative, not exact."))
    if (manifest or {}).get("git_dirty"):
        out.append(("OK", "The working tree was dirty when this run was generated, so the "
                          "recorded git commit does not fully describe the code that ran."))
    if "length_pct" in f:
        out.append(("OK", f"Answers are {f['length_pct']} longer than the control. Length is the "
                          "most visible property a trained model would inherit."))
    delivery = audit.get("delivery") or {}
    pm, bm = delivery.get("pipeline_mean"), delivery.get("plain_mean")
    if not delivery:
        out.append(("BAD", "Delivery quality and the showcase pass were not run on this run, so "
                           "there is no measurement here of whether the added substance cost "
                           "manner."))
    elif pm is not None and bm is not None and pm < bm:
        # The substance/manner trade the whole method is supposed to avoid. If it
        # goes the wrong way it leads the weaknesses, whatever the prose says.
        dims = delivery.get("dimensions") or {}
        worse = sorted(k for k, v in (dims.get("pipeline") or {}).items()
                       if (dims.get("plain") or {}).get(k) is not None and v < dims["plain"][k])
        out.append(("BAD", f"**Delivery quality went the wrong way**: pipeline {pm * 10:.0f}% "
                           f"versus plain {bm * 10:.0f}%. The pipeline bought more welfare "
                           "substance at a measurable cost in manner, which is the trade this "
                           "method is meant to avoid."
                           + (f" Worse on every judged dimension ({', '.join(worse)})."
                              if len(worse) == len(dims.get("pipeline") or {}) and worse
                              else f" Worse on: {', '.join(worse)}." if worse else "")))
    if (audit.get("moral_patient_reasons") or {}).get("failures"):
        out.append(("OK", f"{audit['moral_patient_reasons']['failures']} extraction failures mean "
                          f"the arms are unequal ({f.get('n_pipeline', '?')} pipeline vs "
                          f"{f.get('n_plain', '?')} plain)."))
    if f.get("n") and f["n"] < 100:
        out.append(("OK", f"n = {f['n']} from a single run and a single seed. Treat every "
                          "percentage here as indicative."))
    # Most severe first — a reader skimming this table should hit the failures
    # before the merely-noteworthy.
    return sorted(out, key=lambda w: 0 if w[0] == "BAD" else 1)


def section_weaknesses(audit, content, f, manifest):
    blocks = [_prose(content, "weaknesses_intro", f)]
    warnings = derived_warnings(audit, manifest, f)
    if warnings:
        blocks.append(R.table(
            ["severity", "what the data says"],
            [(R.Raw(R.chip(sev, "bad" if sev == "BAD" else "warn")), R.Raw(R.inline_md(text)))
             for sev, text in warnings]))
    return _section("weaknesses", "7 · Where it is weak", *blocks)


def section_checks(audit):
    rows = []
    for sec in audit.get("sections") or []:
        verdicts = [r.get("verdict") for r in (sec.get("rows") or []) if r.get("verdict")]
        worst = ("BAD" if "BAD" in verdicts else "OK" if "OK" in verdicts
                 else "GOOD" if "GOOD" in verdicts else "—")
        tone = {"BAD": "bad", "OK": "warn", "GOOD": "good"}.get(worst, "")
        counts = " ".join(f"{verdicts.count(v)} {v}" for v in ("GOOD", "OK", "BAD")
                          if verdicts.count(v))
        rows.append((sec.get("title", "?"), sec.get("group", "—"),
                     R.Raw(R.chip(worst, tone)) if tone else "informational", counts or "—"))
    if not rows:
        return ""
    return _section(
        "checks", "8 · Every check that ran",
        "<p>The complete list, including the checks that produced nothing interesting and the "
        "ones that failed. It is here so you can confirm nothing was left out of the sections "
        "above.</p>",
        R.table(["check", "group", "worst verdict", "counts"], rows))


def section_reproduce(content, f, run_id):
    cmd = ("# generate a corpus\n"
           "python dad_pipeline/run.py --config config.yaml --label my-run\n\n"
           "# the standard evals run automatically at the end of a full run;\n"
           "# to re-run them on an existing run dir:\n"
           "python evals/audit_dad.py --input outputs/dad/latest --reasons\n"
           "python evals/diversity.py --input outputs/dad/latest\n\n"
           "# rebuild this report\n"
           f"python report/build_report.py --run outputs/dad/runs/{run_id}")
    return _section("reproduce", "9 · Run it yourself",
                    _prose(content, "reproduce", f),
                    f"<pre>{R.esc(cmd)}</pre>")


# ------------------------------------------------------------------ assembly

def build_report(*, audit, content, diversity=None, manifest=None, corpus=None,
                 baseline=None, rewrites=None, costs=None, run_id="", example=None):
    """The whole report as one HTML string. Pure: no filesystem, no argv.

    Only `audit` and `content` are required; every other input is optional and
    every section degrades to omission or an explicit not-measured note.
    """
    f = facts(audit, manifest, diversity, costs)
    cons = _considerations(audit)
    labels = _labels(audit)
    body = "".join([
        _section("summary", "At a glance", section_summary(audit, f, cons)),
        _section("problem", "1 · The problem", _prose(content, "problem", f)),
        section_example(audit, content, f, corpus, baseline, rewrites, labels, example),
        section_method(content, f, manifest, costs),
        section_measurement(audit, content, f, diversity),
        section_results(audit, content, f, cons, labels, diversity),
        section_footprint(audit, content, f),
        section_weaknesses(audit, content, f, manifest),
        section_checks(audit),
        section_reproduce(content, f, run_id or (manifest or {}).get("run_id", "<run_id>")),
    ])
    cfg = (manifest or {}).get("config") or {}
    meta = (f"{f.get('n', '?')} examples · run <span class='mono'>"
            f"{R.esc(run_id or (manifest or {}).get('run_id', '?'))}</span> · git "
            f"<span class='mono'>{R.esc(str((manifest or {}).get('git_commit', '?'))[:8])}</span>"
            f"{' (dirty)' if (manifest or {}).get('git_dirty') else ''} · backend "
            f"<code>{R.esc(cfg.get('backend', '?'))}</code> · generated with "
            f"<code>{R.esc(f.get('gen_models', '?'))}</code> · judged by "
            f"<code>{R.esc(f.get('judge_model', '?'))}</code>")
    return R.document(title=fill(content["title"], f).strip(),
                      subtitle=fill(content["subtitle"], f).strip(),
                      meta_line=meta, toc=TOC, body=body)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="DAD run directory")
    parser.add_argument("--content", default=None, help="prose file (default report/content.md)")
    parser.add_argument("--out", default=None,
                        help="output HTML (default report/dad_report.html)")
    parser.add_argument("--example", default=None,
                        help="prompt_id to feature as the worked example")
    args = parser.parse_args()

    kwargs = load_inputs(args.run, args.content)
    html = build_report(example=args.example, **kwargs)
    out = Path(args.out or (REPO_ROOT / "report" / "dad_report.html"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    audit = kwargs["audit"]
    print(f"wrote {out} ({len(html):,} bytes)")
    print(f"n={audit.get('n_prompts')} delivery={'yes' if audit.get('delivery') else 'NO'} "
          f"showcase={'yes' if audit.get('showcase') else 'NO'} "
          f"diversity={'yes' if kwargs.get('diversity') else 'NO'}")


if __name__ == "__main__":
    main()
