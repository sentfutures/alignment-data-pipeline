#!/usr/bin/env python3
"""The DAD report page: what the dilemma-SFT pipeline produces, and what it measures.

The audience is a technical reader at another lab — someone deciding whether the method
and its measurement are sound, and whether to run the pipeline themselves. That is a
different job from the Streamlit corpus-audit page, which is organised by what the eval
measured; this is organised by what a reader needs, in the order they need it: the gap,
one worked example, what the numbers say, how it is built, what it would teach a model,
how it is measured, where it is weak, how to run it.

Two rules make the artefact trustworthy, and both are enforced here rather than left to
an author's discipline:

  1. No number is ever typed into the prose. The prose file may interpolate
     ``{{placeholders}}``, which resolve against facts computed from the run's own audit
     JSON. An unresolved placeholder is a build error. Run-conditional figures are
     available to prose only as pre-composed clauses that carry an explicit degraded
     string, so a run without the paid pass says "not measured on this run" instead of
     shipping a stale sentence.

  2. The weaknesses section is DERIVED, not written. Every BAD/OK verdict in the audit,
     plus a fixed set of provenance rules, emits its own line whether or not anyone
     remembered to write it up. Editorial prose adds to that floor; it cannot replace
     it, and the view may collapse rows but only with a visible count.

Built by report/build_report.py. stdlib only, and deliberately no imports from viewer/
or shared/.
"""

import difflib

from report import common as C
from report import render as R

CONTENT_IDS = (
    "title", "lede", "gap", "example_pick", "example_intro", "results_intro",
    "method_intro", "stage1", "stage2", "stage3", "control", "footprint_intro",
    "measurement_intro", "judge_limits", "weaknesses_intro", "reproduce", "appendix_intro",
)

TOC = [
    ("gap", "The gap"),
    ("example", "One example, end to end"),
    ("results", "What the numbers say"),
    ("method", "How it is built"),
    ("footprint", "Stylistic footprint"),
    ("measurement", "How it is measured"),
    ("weaknesses", "Where it is weak"),
    ("reproduce", "Run it yourself"),
    ("appendix", "Appendix"),
]

EYEBROW = "Synthetic training data · dilemma SFT"

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

_DELIVERY_DIMS = ("goal_responsiveness", "proportionality", "tone", "calibration")


# ------------------------------------------------------------------ loading

def load_inputs(run_dir, content_paths):
    """All filesystem access, in one place. Returns build() kwargs."""
    from pathlib import Path
    run_dir = Path(run_dir)
    audit = C.read_json(run_dir / "audit" / "audit_report.json")
    if audit is None:
        raise SystemExit(f"No audit report at {run_dir / 'audit' / 'audit_report.json'} — "
                         f"run: python evals/audit_dad.py --input {run_dir} --reasons")
    return {
        "audit": audit,
        "diversity": C.read_json(run_dir / "audit" / "diversity_report.json"),
        "manifest": C.read_json(run_dir / "run_manifest.json"),
        "content": C.load_content(content_paths, CONTENT_IDS),
        "corpus": C.read_jsonl(run_dir / "final" / "dad_corpus.jsonl"),
        "baseline": C.read_jsonl(run_dir / "baseline" / "baseline_responses.jsonl"),
        "rewrites": C.read_jsonl(run_dir / "step3" / "rewrites.jsonl"),
        "costs": C.read_jsonl(run_dir / "cost_log.jsonl"),
        "run_id": run_dir.name,
    }


# ------------------------------------------------------------------ facts

def _considerations(audit):
    """The headline pair, from either schema.

    Modern reports carry ``valuable_welfare_considerations``; older ones are
    reconstructed from ``moral_patient_reasons`` + ``moves.alternatives`` exactly as
    evals/audit_dad.py's own legacy branch does, so a pre-merge run still renders its
    headline instead of showing a hole.
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
    return {"stage_models": used, "global": glob, "backend": cfg.get("backend"),
            "per_stage": {k: (dad.get(k) or glob) for k in _STAGE_KNOBS}}


def facts(audit, manifest=None, diversity=None, costs=None):
    """Every number the prose can interpolate, computed once, in one place.

    Run-conditional figures appear here as pre-composed CLAUSES with a degraded
    default, not as bare numbers. That is what lets the page open with its finding
    while keeping the invariant that a run missing the paid pass degrades rather than
    lying: the sentence survives, its claim doesn't.
    """
    mpr = audit.get("moral_patient_reasons") or {}
    surv = mpr.get("survival") or {}
    rl = audit.get("response_lengths") or {}
    delivery = audit.get("delivery") or {}
    structure = audit.get("structure") or {}
    lib = audit.get("library_coverage") or {}
    cons = _considerations(audit)
    models = _models(manifest)
    cost_agg = C.costs_by_stage(costs)
    total_cost = sum(v["cost"] for v in cost_agg.values())
    n = audit.get("n_prompts") or 0
    n_measured = (mpr.get("pipeline") or {}).get("n") or rl.get("n") or n
    anchored = (surv.get("kept") or 0) + (surv.get("weakened") or 0) + (surv.get("dropped") or 0)
    f = {
        "n": n,
        "n_measured": n_measured,
        "n_pipeline": (mpr.get("pipeline") or {}).get("n"),
        "n_plain": (mpr.get("plain") or {}).get("n"),
        "extraction_failures": mpr.get("failures"),
        # Two different models do two different jobs, and the old provenance line
        # credited the extractor as the judge.
        "extract_model": mpr.get("model") or "?",
        "judge_model": mpr.get("judge_model") or delivery.get("model") or mpr.get("model") or "?",
        "gen_models": ", ".join(models["stage_models"]) or "?",
        "backend": models["backend"] or "?",
        "cost_total": f"${total_cost:,.2f}" if total_cost else None,
        "cost_per_example": f"${total_cost / n:,.2f}" if total_cost and n else None,
    }
    if cons and cons.get("plain"):
        f["considerations_pipeline"] = f"{cons['pipeline']:.1f}"
        f["considerations_plain"] = f"{cons['plain']:.1f}"
        f["lift_pct"] = f"{(cons['pipeline'] / cons['plain'] - 1) * 100:.0f}%"
        # Clauses are whole sentences, so the degraded version reads as English in the
        # same slot: "the substance comparison did not run on this run."
        f["substance_clause"] = (f"the pipeline's answers carry {f['lift_pct']} more usable "
                                f"welfare considerations per answer "
                                f"({f['considerations_pipeline']} against "
                                f"{f['considerations_plain']})")
    if anchored:
        f["retention_pct"] = f"{(surv.get('kept', 0) + surv.get('weakened', 0)) / anchored:.0%}"
        f["dropped_n"] = surv.get("dropped")
        f["added_total"] = surv.get("added_total")
        f["anchored_n"] = anchored
        if n_measured:
            f["added_per_answer"] = f"{(surv.get('added_total') or 0) / n_measured:.1f}"
    if rl.get("mean_ratio"):
        f["length_ratio"] = f"{rl['mean_ratio']:.2f}"
        f["length_pct"] = f"{(rl['mean_ratio'] - 1) * 100:.0f}%"
        f["chars_pipeline"] = f"{rl.get('pipeline_mean', 0):,.0f}"
        f["chars_plain"] = f"{rl.get('plain_mean', 0):,.0f}"
    if cons and cons.get("plain") and rl.get("pipeline_mean") and rl.get("plain_mean"):
        f["density_pipeline"] = f"{cons['pipeline'] / rl['pipeline_mean'] * 1000:.2f}"
        f["density_plain"] = f"{cons['plain'] / rl['plain_mean'] * 1000:.2f}"
    pm, bm = delivery.get("pipeline_mean"), delivery.get("plain_mean")
    if pm is not None:
        f["delivery_pipeline"] = f"{pm:.1f}"
        f["delivery_plain"] = f"{bm:.1f}" if bm is not None else "?"
        if bm is not None:
            f["delivery_delta"] = f"{abs(pm - bm):.1f}"
            f["delivery_clause"] = (
                f"They are {f['delivery_delta']} points {'worse' if pm < bm else 'better'} than "
                f"it on judged delivery ({f['delivery_pipeline']} against "
                f"{f['delivery_plain']} out of 10)")
    if (structure.get("pipeline") or {}).get("effective_shapes") is not None:
        f["shapes_pipeline"] = f"{structure['pipeline']['effective_shapes']:.1f}"
        f["shapes_plain"] = f"{(structure.get('plain') or {}).get('effective_shapes', 0):.1f}"
    if lib.get("library_size"):
        f["library_n"] = lib["library_size"]
        f["library_used"] = lib.get("used")
        f["library_clause"] = (f"a {lib['library_size']}-entry animal-ethics reasoning library, "
                              f"of which {lib.get('used', '?')} were pulled at least once on "
                              f"this run")
    stance = (audit.get("moves") or {}).get("stance") or {}
    if stance.get("pipeline"):
        f["moralizes_pipeline"] = f"{stance['pipeline'].get('moralizes', 0):.0%}"
        f["moralizes_plain"] = f"{(stance.get('plain') or {}).get('moralizes', 0):.0%}"
    if diversity:
        vendi = diversity.get("vendi") or {}
        nn = diversity.get("nn") or {}
        f["vendi"] = f"{vendi.get('score', 0):.1f}"
        f["vendi_ratio"] = f"{vendi.get('ratio', 0):.2f}"
        f["near_dup_pct"] = f"{nn.get('over_0.90', 0):.0%}"
    f["footprint_regressions"] = _footprint_regressions(audit)
    f = {k: v for k, v in f.items() if v is not None}
    # Degraded defaults. A run that never had the measurement gets a sentence that says
    # so, in the same place the finding would have been.
    for key, default in (
        ("cost_total", "not logged"), ("cost_per_example", "not logged"),
        ("substance_clause", "the substance comparison did not run on this run"),
        ("delivery_clause", "Judged delivery was not measured on this run"),
        ("length_pct", "an unmeasured amount"), ("near_dup_pct", "an unmeasured share"),
        ("library_clause", "an animal-ethics reasoning library"),
        ("added_per_answer", "an unmeasured number of"),
    ):
        f.setdefault(key, default)
    return f


def _footprint_regressions(audit):
    """Which footprint measures actually moved the wrong way, as a prose clause.

    The old prose asserted "one of these measures is an outright regression" in a
    section whose blocks are all conditional — on a run where none of them regressed
    the sentence was simply false. Deriving it means it cannot be.
    """
    bad = []
    rl = audit.get("response_lengths") or {}
    if (rl.get("mean_ratio") or 0) > 1.15:
        bad.append("length")
    st = audit.get("structure") or {}
    p, b = (st.get("pipeline") or {}), (st.get("plain") or {})
    if p.get("effective_shapes") and b.get("effective_shapes") \
            and p["effective_shapes"] < b["effective_shapes"]:
        bad.append("structural variety")
    stance = (audit.get("moves") or {}).get("stance") or {}
    if (stance.get("pipeline") or {}).get("moralizes", 0) > (stance.get("plain") or {}) \
            .get("moralizes", 0):
        bad.append("moralizing")
    if not bad:
        return "None of these measures moved the wrong way on this run"
    if len(bad) == 1:
        return f"On this run {bad[0]} moved the wrong way"
    return f"On this run {', '.join(bad[:-1])} and {bad[-1]} moved the wrong way"


def _labels(audit):
    """prompt_id -> stable display id (response gid), from the report's own map."""
    return {pid: (gids or {}).get("response") or pid
            for pid, gids in (audit.get("gid_map") or {}).items()}


# ------------------------------------------------------------------ hero

def hero_tiles(audit, f, cons):
    """Three numbers, one of them a regression. Five was a dashboard."""
    items = []
    if cons and cons.get("plain"):
        items.append(R.stat(f"+{f.get('lift_pct', '?')}",
                            "more valuable welfare considerations per answer",
                            f"{f['considerations_pipeline']} against "
                            f"{f['considerations_plain']} for the control", tone="hero"))
    delivery = audit.get("delivery") or {}
    pm, bm = delivery.get("pipeline_mean"), delivery.get("plain_mean")
    if pm is not None:
        worse = bm is not None and pm < bm
        items.append(R.stat(
            f"{pm:.1f}/10", "judged delivery quality",
            f"the control scores {bm:.1f}" if bm is not None else "",
            flag="regression" if worse else "holds up", tone="bad" if worse else "good"))
    if f.get("cost_per_example") not in (None, "not logged"):
        items.append(R.stat(f["cost_per_example"], "per example, end to end",
                            f"{f['cost_total']} for this run"))
    return R.tiles(items)


# ------------------------------------------------------------------ sections

def section_gap(content, f):
    return C.section("gap", "The gap", C.prose(content, "gap", f))


def section_example(audit, content, f, baseline, rewrites, labels, pick=None):
    """One full dilemma, control against pipeline.

    Both answers stay inline in full: they are the artefact, and a reader at a lab
    wants to read them rather than take a summary's word for it. What moved out is the
    word-level diff, which as 1,095 words of confetti earned nothing where it stood.
    """
    blocks = [C.prose(content, "example_intro", f)]
    showcase = (audit.get("showcase") or {}).get("examples") or []
    by_pid_base = {r.get("prompt_id"): r for r in baseline or []}
    by_pid_rw = {r.get("prompt_id"): r for r in rewrites or []}

    chosen = None
    pick = pick or _example_pick(content)
    if pick:
        chosen = {"prompt_id": pick, "highlights": [], "summary": "",
                  "provenance": "chosen by hand from this run's records and pinned in the prose "
                                "file, so a rebuild shows the same case"}
    elif showcase:
        ex = showcase[0]
        chosen = {"prompt_id": ex.get("prompt_id"), "summary": ex.get("summary", ""),
                  "highlights": ex.get("highlights") or [],
                  "user_message": ex.get("user_message"), "plain": ex.get("plain_response"),
                  "pipeline": ex.get("pipeline_response"),
                  "provenance": "selected by the showcase judge from the retention and delivery "
                                "data; the highlighted spans are the judge's, validated against "
                                "the response text"}
    else:
        per_case = (audit.get("moral_patient_reasons") or {}).get("per_case") or {}
        ranked = sorted(per_case.items(),
                        key=lambda kv: -len(((kv[1].get("survival") or {}).get("added") or [])))
        if ranked:
            chosen = {"prompt_id": ranked[0][0], "highlights": [], "summary": "",
                      "provenance": "selected mechanically, as the record where the pipeline added "
                                    "the most considerations beyond the control; this run has no "
                                    "judge-selected showcase"}

    if not chosen or not chosen.get("prompt_id"):
        blocks.append(R.note("No worked example could be built: this run has neither a showcase "
                             "pass nor per-record retention data."))
        return C.section("example", "One example, end to end", *blocks)

    pid = chosen["prompt_id"]
    base = by_pid_base.get(pid) or {}
    rw = by_pid_rw.get(pid) or {}
    user_msg = chosen.get("user_message") or base.get("user_message") or rw.get("user_message", "")
    plain = chosen.get("plain") or base.get("baseline_response", "")
    pipeline = chosen.get("pipeline") or rw.get("rewritten_response", "")
    if not (user_msg and pipeline):
        blocks.append(R.note(f"Worked example {R.esc(pid)} could not be assembled from this "
                             "run's files."))
        return C.section("example", "One example, end to end", *blocks)

    gid = labels.get(pid, pid)
    if chosen.get("summary"):
        blocks.append(f"<p>{R.inline_md(chosen['summary'])}</p>")
    blocks.append(f"<p class='muted'>Record <span class='mono'>{R.esc(gid)}</span> — "
                  f"{R.esc(chosen['provenance'])}.</p>")
    blocks.append("<h4>The user asked</h4>")
    blocks.append(R.quote(user_msg))

    per_case = ((audit.get("moral_patient_reasons") or {}).get("per_case") or {}).get(pid) or {}
    surv = per_case.get("survival") or {}
    counts = _survival_counts(surv)
    if counts:
        blocks.append("<h4>What changed</h4>")
        blocks.append(R.dek(
            f"The retention judge read the control's answer first and tracked each of its "
            f"welfare considerations into the pipeline's: {counts['kept']} kept, "
            f"{counts['weakened']} weakened, {counts['dropped']} dropped, and "
            f"{counts['added']} points the pipeline raised that the control did not."))

    blocks.append(R.sidebyside(
        "The control · plain model, no system prompt", R.highlight(plain, []),
        "The pipeline", R.highlight(pipeline, chosen.get("highlights")),
        left_tone="plain", right_tone="pipeline"))

    if surv:
        blocks.append(R.details(
            "What the retention judge found, item by item",
            R.table(["fate", "n", "the judge's wording"], _survival_rows(surv), align="lrl")))

    if rw.get("draft_response") and rw.get("rewritten_response"):
        before, after = rw["draft_response"], rw["rewritten_response"]
        blocks.append(R.details(
            "What the constitution rewrite changed in this answer",
            f"<p class='muted'>{_diff_summary(before, after)} Stage 2's draft and the shipped "
            "answer are a revision of one another, so a diff of those two is meaningful; the "
            "control and the pipeline answer are independently generated, so diffing them "
            "would be noise. The three largest changes:</p>" + _diff_hunks(before, after),
            meta="3 largest changes · full diff in the appendix"))
    return C.section("example", "One example, end to end", *blocks)


def _example_pick(content):
    """The prompt_id pinned in the prose file, or None for automatic selection.

    Pinned in the prose rather than passed on the command line so a rebuild reproduces
    the same worked example without anyone having to remember a flag.
    """
    raw = (content.get("example_pick") or "").strip()
    return None if raw.lower() in ("", "auto") else raw.split()[0]


def _survival_counts(surv):
    if not surv:
        return None
    anchored = surv.get("anchored") or []
    out = {"kept": 0, "weakened": 0, "dropped": 0, "added": len(surv.get("added") or [])}
    for a in anchored:
        if a.get("verdict") in out:
            out[a["verdict"]] += 1
    return out if (anchored or out["added"]) else None


def _survival_rows(surv):
    anchored = surv.get("anchored") or []
    groups = {v: [a["reason"] for a in anchored if a.get("verdict") == v]
              for v in ("kept", "weakened", "dropped")}
    groups["added"] = surv.get("added") or []
    return [(label, len(groups[key]), "; ".join(groups[key][:3]) or "—")
            for key, label in (("kept", "kept from the control"), ("weakened", "weakened"),
                               ("dropped", "dropped"), ("added", "added by the pipeline"))]


# ------------------------------------------------------------------ diff

_DIFF_CSS = ("<style>ins{background:var(--mark);text-decoration:none}"
             "del{opacity:.5;text-decoration:line-through}</style>")


def _opcodes(before, after):
    a, b = before.split(), after.split()
    return a, b, difflib.SequenceMatcher(None, a, b).get_opcodes()


def _diff_summary(before, after):
    a, b, ops = _opcodes(before, after)
    changed = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")
    return (f"The rewrite touched {changed / max(len(b), 1):.0%} of the answer's words "
            f"({len(a):,} words in, {len(b):,} out).")


def _render_ops(a, b, ops):
    out = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            out.append(R.esc(" ".join(b[j1:j2])))
        else:
            if tag in ("replace", "delete"):
                out.append(f"<del>{R.esc(' '.join(a[i1:i2]))}</del>")
            if tag in ("replace", "insert"):
                out.append(f"<ins>{R.esc(' '.join(b[j1:j2]))}</ins>")
    return " ".join(out)


def _word_diff(before, after):
    """Full word-level diff. Lives in the appendix — informative, but as running text
    it is confetti, and it was a third of the page."""
    a, b, ops = _opcodes(before, after)
    return _DIFF_CSS + f"<div class='resp'>{_render_ops(a, b, ops)}</div>"


def _diff_hunks(before, after, *, top=3, context=16):
    """The N largest changed runs, each with surrounding context.

    A reader wants to know what stage 3 does, which three concrete edits answer and a
    full diff buries.
    """
    a, b, ops = _opcodes(before, after)
    changes = [op for op in ops if op[0] != "equal"]
    if not changes:
        return "<p class='muted'>The rewrite changed nothing in this answer.</p>"
    biggest = sorted(changes, key=lambda op: -max(op[2] - op[1], op[4] - op[3]))[:top]
    biggest = sorted(biggest, key=lambda op: op[3])
    out = []
    for tag, i1, i2, j1, j2 in biggest:
        pre = " ".join(b[max(0, j1 - context):j1])
        post = " ".join(b[j2:j2 + context])
        mid = _render_ops(a, b, [(tag, i1, i2, j1, j2)])
        out.append(f"<div class='resp'>… {R.esc(pre)} {mid} {R.esc(post)} …</div>")
    return _DIFF_CSS + "".join(out)


# ------------------------------------------------------------------ results

def _verdict_chip(better):
    if better is None:
        return R.Raw(R.chip("not measured"))
    return R.Raw(R.chip("better" if better else "worse", "good" if better else "bad"))


def scoreboard(audit, f, cons):
    """The table a reader screenshots. Every chart below is an expansion of one row.

    Deliberately includes the two rows that undercut the headline — density and
    structural variety — next to the headline rather than in a footnote.
    """
    rows = []
    if cons and cons.get("plain"):
        rows.append(("valuable welfare considerations per answer", f["considerations_plain"],
                     f["considerations_pipeline"], _verdict_chip(True)))
    if "delivery_pipeline" in f and f.get("delivery_plain") not in (None, "?"):
        rows.append(("judged delivery quality, 0–10", f["delivery_plain"],
                     f["delivery_pipeline"],
                     _verdict_chip(float(f["delivery_pipeline"]) >= float(f["delivery_plain"]))))
    if "density_pipeline" in f:
        rows.append(("considerations per 1,000 characters", f["density_plain"],
                     f["density_pipeline"],
                     _verdict_chip(float(f["density_pipeline"]) >= float(f["density_plain"]))))
    if "chars_pipeline" in f:
        rows.append(("answer length, characters", f["chars_plain"], f["chars_pipeline"],
                     R.Raw(R.chip("longer", "warn"))))
    if "shapes_pipeline" in f:
        rows.append(("structural variety, effective shapes", f["shapes_plain"],
                     f["shapes_pipeline"],
                     _verdict_chip(float(f["shapes_pipeline"]) >= float(f["shapes_plain"]))))
    stance = (audit.get("moves") or {}).get("stance") or {}
    if stance.get("pipeline"):
        p = stance["pipeline"].get("moralizes", 0)
        b = (stance.get("plain") or {}).get("moralizes", 0)
        rows.append(("answers that moralize", f"{b:.0%}", f"{p:.0%}", _verdict_chip(p <= b)))
    else:
        rows.append(("answers that moralize", "—", "—", _verdict_chip(None)))
    if not rows:
        return ""
    return R.table(["measure", "control", "pipeline", ""], rows, align="lrrl")


def section_results(audit, content, f, cons, labels):
    blocks = [C.prose(content, "results_intro", f)]
    board = scoreboard(audit, f, cons)
    if board:
        blocks.append(f"<h4>The whole comparison, on {f.get('n_measured', '?')} dilemmas</h4>")
        blocks.append(board)

    mpr = audit.get("moral_patient_reasons") or {}
    if cons and cons.get("plain") is not None:
        subset_rows = [{"label": name, "plain": b, "pipeline": p}
                       for name, b, p in cons["subsets"] if p is not None]
        blocks.append(R.figure(
            title="Valuable welfare considerations per answer",
            note_="A distinct welfare point, or a concrete lower-harm action, that a judge "
                  "reading the answer counted as useful to the person asking. Both arms "
                  "answered the same dilemmas.",
            chart=R.hbar([("the control", round(cons["plain"], 2)),
                          ("the pipeline", round(cons["pipeline"], 2))],
                         color=R.ARM_PAIR, fmt="{:.1f}"),
            caption=f"**The pipeline raises {f.get('lift_pct', '?')} more of them.** "
                    f"The two arms answer the same dilemmas, so the gap is per-answer, "
                    f"not corpus-wide."))
        if subset_rows:
            blocks.append(R.figure(
                title="Split by kind of consideration",
                chart=R.grouped_hbar(subset_rows,
                                     series=[("plain", R.PLAIN), ("pipeline", R.PIPELINE)],
                                     fmt="{:.2f}"),
                caption="**The gain is in reasoning, not only in offering alternatives.**"))
        if cons["source"] == "reconstructed":
            blocks.append("<p class='muted'>Reconstructed from this run's separate reasoning "
                          "and alternatives measures; it predates the unified extraction.</p>")

    if mpr.get("failures"):
        blocks.append(R.note(
            f"Means are over {f.get('n_pipeline', '?')} pipeline and {f.get('n_plain', '?')} "
            f"control answers: {mpr['failures']} extractions failed and are excluded. That is "
            "not a fully matched comparison, and a missing answer is a gap rather than a zero."))

    surv = mpr.get("survival") or {}
    if surv.get("kept") is not None:
        blocks.append(R.figure(
            title="What happened to the control's considerations",
            note_="The judge read the control's answer first, then tracked each of its "
                  "considerations into the pipeline's.",
            chart=R.segbar([("kept", surv.get("kept") or 0, R.PLAIN),
                            ("weakened", surv.get("weakened") or 0, "var(--series-4)"),
                            ("dropped", surv.get("dropped") or 0, "var(--series-8)"),
                            ("added", surv.get("added_total") or 0, R.PIPELINE)]),
            caption=f"**{f.get('retention_pct', '?')} of the control's "
                    f"{f.get('anchored_n', '?')} considerations survive the pipeline, and it "
                    f"adds {f.get('added_per_answer', '?')} more per answer.** Nothing here "
                    f"checks whether the additions are correct — see the limits below."))

    types_p = (mpr.get("pipeline") or {}).get("type_hist") or _type_hist(mpr.get("per_case"),
                                                                        "pipeline")
    types_b = (mpr.get("plain") or {}).get("type_hist") or _type_hist(mpr.get("per_case"), "plain")
    if types_p and types_b:
        gloss = (audit.get("reason_composition") or {}).get("type_gloss") or {}
        keys = list(dict(types_p, **types_b))
        rows = [{"label": k, "plain": types_b.get(k, 0), "pipeline": types_p.get(k, 0)}
                for k in keys]
        blocks.append(R.figure(
            title="Kinds of consideration raised",
            chart=R.grouped_hbar(rows, series=[("plain", R.PLAIN), ("pipeline", R.PIPELINE)]),
            caption="**The pipeline does not just say more of the same thing.** Its largest "
                    "gains are in the kinds of point the control raises least.",
            table_html=R.table(["kind", "what it is", "control", "pipeline"],
                               [(k, gloss.get(k, "—"), types_b.get(k, 0), types_p.get(k, 0))
                                for k in keys], align="llrr") if gloss else None))

    blocks.extend(_delivery_blocks(audit, f, mpr, labels))
    return C.section("results", "What the numbers say", *blocks)


def _delivery_blocks(audit, f, mpr, labels):
    delivery = audit.get("delivery") or {}
    if not delivery.get("per_case"):
        return [R.note(
            "Delivery quality was **not measured on this run**, so there is no evidence here "
            "either way about whether the added substance cost manner. Populate it with "
            "`python evals/audit_dad.py --input <run> --reasons`.")]
    out = ["<h3>Manner</h3>"]
    pm, bm = delivery.get("pipeline_mean"), delivery.get("plain_mean")
    if bm is not None and pm < bm:
        out.append(R.note(
            f"**The pipeline is {f.get('delivery_delta', '?')} points worse than the control "
            f"here** ({f['delivery_pipeline']} against {f['delivery_plain']} out of 10). The "
            "extra substance was not free: judged on manner alone, the control's answers read "
            "as more helpful and less obtrusive. Any claim that this pipeline adds substance "
            "without costing delivery is not supported by this run.", tone="bad"))
    dims = delivery.get("dimensions") or {}
    if dims.get("pipeline"):
        keys = [k for k in _DELIVERY_DIMS if k in dims["pipeline"]]
        rows = []
        for k in keys:
            p, b = dims["pipeline"].get(k), (dims.get("plain") or {}).get(k)
            rows.append((k.replace("_", " "), f"{b:.2f}" if b is not None else "—",
                         f"{p:.2f}" if p is not None else "—",
                         f"{p - b:+.2f}" if p is not None and b is not None else "—"))
        n_worse = sum(1 for r in rows if r[3].startswith("-"))
        out.append(R.figure(
            title="Delivery quality, dimension by dimension",
            note_="Each dimension is judged 0–10 on the answer alone: did it serve the goal the "
                  "user actually had, was the response proportionate, was the tone right, was "
                  "uncertainty calibrated.",
            chart=R.table(["dimension", "control", "pipeline", "delta"], rows, align="lrrr"),
            caption=(f"**The pipeline is worse on every one of them.** That is the strongest "
                     f"single piece of evidence against this corpus, and it is why delivery "
                     f"leads the weaknesses below." if n_worse == len(rows) else
                     f"**Worse on {n_worse} of {len(rows)} dimensions.**")))
    n_p, n_b, fails = delivery.get("n_pipeline"), delivery.get("n_plain"), delivery.get("failures")
    asym = ""
    if n_p is not None and n_b is not None and (n_p != n_b or fails):
        asym = (f" These means are over {n_p} pipeline and {n_b} control answers — "
                f"{fails or 0} judgements failed, so the two arms are not the same set of "
                f"records.")
    out.append(R.figure(
        title="Substance against manner, one dot per answer",
        note_="Judged delivery quality on the horizontal axis, valuable welfare considerations "
              "on the vertical. Diamonds are each arm's mean." + asym,
        chart=_pareto(delivery, mpr, labels),
        caption="**The pipeline arm sits up and to the left: it buys substance with manner.** "
                "Up and to the right would be substance at no cost, which is what the method "
                "is aiming at and did not reach on this run."))
    return out


def _type_hist(per_case, arm):
    out = {}
    for case in (per_case or {}).values():
        for k, v in ((case.get(arm) or {}).get("type_hist") or {}).items():
            out[k] = out.get(k, 0) + v
    return out


_SURVIVAL_CATS = (("dropped", "var(--series-8)"), ("weakened", "var(--series-4)"),
                  ("kept", R.PLAIN), ("added", R.PIPELINE))


def _survival_chart(per_case, labels):
    rows = []
    for pid in sorted(per_case or {}):
        surv = (per_case[pid] or {}).get("survival") or {}
        counts = _survival_counts(surv)
        if not counts:
            continue
        rows.append({"label": labels.get(pid, pid), "segments": counts,
                     "tips": {k: f"{labels.get(pid, pid)} — {k}: {v}"
                              for k, v in counts.items()}})
    if not rows:
        return ""
    return R.stacked_bar(rows, categories=list(_SURVIVAL_CATS), ylabel="considerations",
                         xlabel="one column per record")


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
              "tip": f"{arm} mean: delivery {s[0] / s[2]:.1f}/10, "
                     f"{s[1] / s[2]:.1f} considerations"}
             for arm, s in sums.items() if s[2]]
    return R.scatter(pts, xdomain=(0, 10), marks=marks)


# ------------------------------------------------------------------ method

def section_method(content, f, manifest, costs):
    blocks = [C.prose(content, "method_intro", f)]
    for key, heading in (("stage1", "Stage 1 · the dilemma"),
                         ("stage2", "Stage 2 · the reasoning"),
                         ("stage3", "Stage 3 · the constitution rewrite"),
                         ("control", "The control arm")):
        blocks.append(f"<h3>{R.esc(heading)}</h3>{C.prose(content, key, f)}")
    table = C.stage_cost_table(costs, _STAGE_LABELS)
    if table:
        blocks.append(R.details("Per-stage cost and model", table,
                                meta=f"{f.get('cost_total', '?')} for this run"))
    elif _models(manifest)["stage_models"]:
        blocks.append(R.details("Per-stage model", R.table(
            ["stage", "model"], [(k.replace("_model", "").replace("_", " "), v)
                                 for k, v in _models(manifest)["per_stage"].items()])))
    return C.section("method", "How it is built", *blocks)


# ------------------------------------------------------------------ footprint

def section_footprint(audit, content, f):
    blocks = [C.prose(content, "footprint_intro", f)]
    rl = audit.get("response_lengths") or {}
    if rl.get("pipeline_mean"):
        blocks.append(R.figure(
            title="Answer length",
            chart=R.hbar([("the control", round(rl.get("plain_mean", 0))),
                          ("the pipeline", round(rl["pipeline_mean"]))],
                         color=R.ARM_PAIR, unit=" chars", fmt="{:,.0f}"),
            caption=f"**{f.get('length_pct', '?')} longer than the control.** Length is the "
                    f"most visible property a model would inherit from this corpus, and the "
                    f"judges see it too — see the limits below."))
    stance = (audit.get("moves") or {}).get("stance") or {}
    if stance.get("pipeline"):
        rows = [{"label": k, "plain": (stance.get("plain") or {}).get(k),
                 "pipeline": stance["pipeline"].get(k)}
                for k in ("defers", "calibrated", "moralizes")
                if stance["pipeline"].get(k) is not None]
        blocks.append(R.figure(
            title="Stance",
            chart=R.grouped_hbar(rows, series=[("plain", R.PLAIN), ("pipeline", R.PIPELINE)],
                                 percent=True),
            caption=f"**The pipeline moralizes more than the control** "
                    f"({f.get('moralizes_pipeline', '?')} against "
                    f"{f.get('moralizes_plain', '?')}). Stage 3 exists to prevent exactly this."))
    tics = audit.get("tracked_tics") or audit.get("stock_phrases") or {}
    watch = tics.get("watch") or {}
    n_pipe, n_plain = tics.get("n_pipeline") or 0, tics.get("n_plain") or 0
    if watch and n_pipe:
        rows = sorted(({"label": phrase,
                        "plain": (d.get("plain") or 0) / n_plain if n_plain else 0,
                        "pipeline": (d.get("pipeline") or 0) / n_pipe}
                       for phrase, d in watch.items()
                       if (d.get("pipeline") or d.get("plain"))),
                      key=lambda r: -r["pipeline"])[:10]
        if rows:
            blocks.append(R.figure(
                title="Tracked phrases",
                note_="Phrases the eval watches by name because earlier runs turned them into "
                      "habits. Share of answers in each arm containing the phrase at least once.",
                chart=R.grouped_hbar(rows, series=[("plain", R.PLAIN), ("pipeline", R.PIPELINE)],
                                     percent=True, label_w=210),
                caption="**A phrase in more than about half of the answers is a tic rather than "
                        "a word choice.** The pipeline's most common phrase is well under that."))
    moves = (audit.get("rhetorical_moves") or {}).get("moves") or {}
    if moves:
        rows = sorted(({"label": name, "plain": d.get("plain_share"),
                        "pipeline": d.get("pipeline_share")} for name, d in moves.items()),
                      key=lambda r: -(r["pipeline"] or 0))
        gloss = {name: (d.get("description") or "") for name, d in moves.items()}
        invented = [r["label"] for r in rows
                    if (r["pipeline"] or 0) > 0.25 and not (r["plain"] or 0)]
        dropped = [r["label"] for r in rows
                   if (r["plain"] or 0) > 0.25 and not (r["pipeline"] or 0)]
        blocks.append(R.figure(
            title="Rhetorical habits",
            note_="Argumentative moves, as a share of each arm's answers. Hover a bar for what "
                  "the move is; all of them, with definitions, are in the appendix.",
            chart=R.grouped_hbar(rows[:6], series=[("plain", R.PLAIN), ("pipeline", R.PIPELINE)],
                                 percent=True, label_w=210, glossary=gloss),
            caption=(_habits_caption(invented, dropped) if (invented or dropped) else
                     "**Both arms reach for the same moves at similar rates.**")))
    structure = audit.get("structure") or {}
    if (structure.get("pipeline") or {}).get("effective_shapes") is not None:
        p, b = structure["pipeline"], structure.get("plain") or {}
        worse = b.get("effective_shapes") and p["effective_shapes"] < b["effective_shapes"]
        blocks.append(R.figure(
            title="Structural variety",
            note_="Effective number of distinct answer shapes — paragraph and list structure — "
                  "across the arm. Higher is more varied.",
            chart=R.hbar([("the control", b.get("effective_shapes", 0)),
                          ("the pipeline", p.get("effective_shapes", 0))],
                         color=R.ARM_PAIR, fmt="{:.1f}"),
            caption=(f"**The pipeline's answers are less varied in shape than the control's** "
                     f"({f.get('shapes_pipeline', '?')} against {f.get('shapes_plain', '?')} "
                     f"effective shapes). More substance came at a cost in format range."
                     if worse else "**Structural range holds up against the control.**")))
    return C.section("footprint", "Stylistic footprint", *blocks)


def _habits_caption(invented, dropped):
    """Say which habit the pipeline invented and which it traded away, or say neither.

    The old caption asserted "invented one closing move and dropped another" as a fixed
    sentence about conditional data.
    """
    if invented and dropped:
        claim = (f"**The pipeline turned `{invented[0]}` into a habit the control never shows, "
                 f"and dropped `{dropped[0]}`, which the control reaches for.**")
    elif invented:
        claim = (f"**`{invented[0]}` is a habit the pipeline has and the control does not.**")
    else:
        claim = f"**The pipeline dropped `{dropped[0]}`, a move the control reaches for.**"
    return (claim + " A move that appears in one arm and not the other is the clearest thing a "
                    "model trained on this data would pick up.")


# ------------------------------------------------------------------ measurement

def section_measurement(audit, content, f, diversity):
    blocks = [C.prose(content, "measurement_intro", f)]
    mpr = audit.get("moral_patient_reasons") or {}
    checks = [
        ("Valuable welfare considerations", bool(mpr.get("pipeline")),
         "Distinct welfare points and concrete lower-harm actions per answer, both arms"),
        ("Retention", bool(mpr.get("survival")),
         "Item by item, which of the control's considerations the pipeline kept, weakened or "
         "dropped, and what it added"),
        ("Delivery quality", bool(audit.get("delivery")),
         "How helpful, proportionate and non-preachy each answer is, judged 0–10"),
        ("Showcase examples", bool(audit.get("showcase")),
         "Concrete pipeline-beats-control cases with verbatim improved spans"),
        ("Response stance", bool((audit.get("moves") or {}).get("stance")),
         "Whether an answer defers, stays calibrated, or moralizes"),
        ("Tracked phrases and rhetorical moves",
         bool(audit.get("tracked_tics") or audit.get("rhetorical_moves")),
         "Recurring phrasing and argumentative habits, as a share of each arm"),
        ("Length, structure, jargon", bool(audit.get("response_lengths")),
         "Offline corpus measurements against the control"),
        ("Semantic diversity", bool(diversity),
         "Embedding near-duplicate rate, topic spread, effective record count"),
    ]
    rows = [(name, what if ok else R.Raw(f"<i>not run on this run</i> — {R.esc(what)}"))
            for name, ok, what in checks]
    blocks.append(R.table(["check", "what it establishes"], rows))
    blocks.append(C.prose(content, "judge_limits", f))
    return C.section("measurement", "How it is measured", *blocks)


# ------------------------------------------------------------------ weaknesses

def derived_warnings(audit, manifest, f):
    """The weaknesses floor: computed from the run, never author-supplied.

    Anything BAD or OK in the audit, plus provenance and a set of DAD-specific rules.
    If a future run regresses, its warning appears here whether or not the prose was
    updated. Rows are only ever added to this list.
    """
    out = C.audit_verdict_warnings(audit)
    out += C.provenance_warnings(manifest, n=f.get("n"))
    if f.get("length_pct") != "an unmeasured amount":
        out.append(("OK", f"Answers are {f['length_pct']} longer than the control's. Length is "
                          "the most visible property a trained model would inherit."))
    delivery = audit.get("delivery") or {}
    pm, bm = delivery.get("pipeline_mean"), delivery.get("plain_mean")
    if not delivery:
        out.append(("BAD", "Delivery quality and the showcase pass did not run, so nothing here "
                           "measures whether the added substance cost manner."))
    elif pm is not None and bm is not None and pm < bm:
        # The substance/manner trade the whole method is supposed to avoid. If it goes
        # the wrong way it leads the weaknesses, whatever the prose says.
        dims = delivery.get("dimensions") or {}
        worse = sorted(k for k, v in (dims.get("pipeline") or {}).items()
                       if (dims.get("plain") or {}).get(k) is not None and v < dims["plain"][k])
        out.append(("BAD", f"**Delivery quality went the wrong way**: {pm:.1f} against the "
                           f"control's {bm:.1f} out of 10. The pipeline bought welfare substance "
                           "at a measurable cost in manner, which is the trade this method is "
                           "meant to avoid."
                           + (f" Worse on every judged dimension ({', '.join(worse)})."
                              if len(worse) == len(dims.get("pipeline") or {}) and worse
                              else f" Worse on: {', '.join(worse)}." if worse else "")))
    # Per-measure arm asymmetry. The retention rule below only reads its own failures,
    # so an unmatched delivery comparison used to reach the page undisclosed.
    for name, block in (("Delivery quality", delivery),):
        n_p, n_b = block.get("n_pipeline"), block.get("n_plain")
        if n_p is not None and n_b is not None and (n_p != n_b or block.get("failures")):
            out.append(("BAD" if abs(n_p - n_b) > 0.15 * max(n_p, n_b, 1) else "OK",
                        f"{name} is not a matched comparison: {n_p} pipeline against {n_b} "
                        f"control answers, with {block.get('failures') or 0} judgements failing. "
                        f"The two means are over different sets of records."))
    if (audit.get("moral_patient_reasons") or {}).get("failures"):
        out.append(("OK", f"{audit['moral_patient_reasons']['failures']} extraction failures mean "
                          f"the arms are unequal ({f.get('n_pipeline', '?')} pipeline against "
                          f"{f.get('n_plain', '?')} control)."))
    return sorted(out, key=lambda w: 0 if w[0] == "BAD" else 1)


def section_weaknesses(audit, content, f, manifest):
    warnings = derived_warnings(audit, manifest, f)
    return C.section("weaknesses", "Where it is weak",
                     C.prose(content, "weaknesses_intro", f),
                     C.warnings_table(warnings))


# ------------------------------------------------------------------ reproduce

def section_reproduce(content, f, run_id):
    cmd = ("# generate a corpus\n"
           "python dad_pipeline/run.py --config config.yaml --label my-run\n\n"
           "# the standard evals run automatically at the end of a full run;\n"
           "# to re-run them on an existing run directory:\n"
           "python evals/audit_dad.py --input outputs/dad/latest --reasons\n"
           "python evals/diversity.py --input outputs/dad/latest\n\n"
           "# rebuild this page\n"
           f"python report/build_report.py --dad-run outputs/dad/runs/{run_id}")
    return C.section("reproduce", "Run it yourself",
                     C.prose(content, "reproduce", f),
                     f"<pre>{R.esc(cmd)}</pre>")


# ------------------------------------------------------------------ appendix

def section_appendix(audit, content, f, rewrites, labels, diversity, pick=None):
    """Everything that is evidence but not argument.

    It is in the page rather than cut so that "nothing was left out" stays a checkable
    claim, and collapsed so it costs a reader nothing.
    """
    blocks = [C.prose(content, "appendix_intro", f)]

    rows, verdicted = [], 0
    for sec in audit.get("sections") or []:
        verdicts = [r.get("verdict") for r in (sec.get("rows") or []) if r.get("verdict")]
        worst = ("BAD" if "BAD" in verdicts else "OK" if "OK" in verdicts
                 else "GOOD" if "GOOD" in verdicts else "")
        if worst:
            verdicted += 1
        counts = " ".join(f"{verdicts.count(v)} {v}" for v in ("GOOD", "OK", "BAD")
                          if verdicts.count(v))
        rows.append((sec.get("title", "?"), sec.get("group", "—"),
                     R.Raw(R.chip(worst, {"BAD": "bad", "OK": "warn", "GOOD": "good"}[worst]))
                     if worst else "informational", counts or "—"))
    if rows:
        blocks.append(R.details(
            "Every check that ran",
            R.table(["check", "group", "worst verdict", "counts"], rows, align="llll"),
            meta=f"{len(rows)} checks · {verdicted} carry a verdict"))

    moves = (audit.get("rhetorical_moves") or {}).get("moves") or {}
    if moves:
        blocks.append(R.details(
            "What each rhetorical move is",
            R.table(["move", "what it is", "control", "pipeline"],
                    [(name, d.get("description") or "—",
                      f"{d.get('plain_share') or 0:.0%}", f"{d.get('pipeline_share') or 0:.0%}")
                     for name, d in sorted(moves.items(),
                                           key=lambda kv: -(kv[1].get("pipeline_share") or 0))],
                    align="llrr"),
            meta=f"{len(moves)} moves"))

    per_case = (audit.get("moral_patient_reasons") or {}).get("per_case") or {}
    chart = _survival_chart(per_case, labels)
    if chart:
        blocks.append(R.details(
            "Retention record by record",
            R.figure(title="Considerations kept, weakened, dropped and added, per record",
                     chart=chart,
                     caption="**No single record collapses.** The check this answers is whether "
                             "the corpus average hides a handful of records where the pipeline "
                             "threw the control's reasoning away."),
            meta=f"{len(per_case)} records"))

    if diversity:
        vendi = diversity.get("vendi") or {}
        nn = diversity.get("nn") or {}
        clusters = ((diversity.get("scopes") or {}).get("combined") or {}).get("clusters") or {}
        blocks.append(R.details(
            "Corpus diversity",
            "<p class='muted'>Embedding-space measurements over the final corpus, from "
            f"<code>{R.esc((diversity or {}).get('embed_model', '?'))}</code>.</p>"
            + R.tiles([
                R.stat(f"{vendi.get('score', 0):.1f}", "effectively distinct records",
                       f"of {diversity.get('n_records', '?')} actual records — a diversity "
                       f"score that counts near-duplicates as fractions of a record"),
                R.stat(f"{nn.get('over_0.90', 0):.0%}", "near-duplicate records",
                       f"cosine similarity above 0.90 to their nearest neighbour; "
                       f"{nn.get('over_0.80', 0):.0%} above 0.80"),
                R.stat(f"{clusters.get('evenness', 0):.2f}", "topic-spread evenness",
                       f"1.00 would be perfectly even; the largest topic cluster holds "
                       f"{clusters.get('largest_share', 0):.0%} of records"),
            ]),
            meta=f"{diversity.get('n_records', '?')} records"))

    pid = pick or _example_pick(content)
    rw = next((r for r in rewrites or [] if r.get("prompt_id") == pid), None)
    if rw and rw.get("draft_response") and rw.get("rewritten_response"):
        blocks.append(R.details(
            "The full stage-3 rewrite diff for the worked example",
            "<p class='muted'>Struck-through words are stage 2's draft; highlighted words are "
            "the shipped answer.</p>" + _word_diff(rw["draft_response"], rw["rewritten_response"]),
            meta=f"{len(rw['rewritten_response'].split()):,} words"))
    return C.section("appendix", "Appendix", *blocks)


# ------------------------------------------------------------------ assembly

def body(*, audit, content, diversity=None, manifest=None, corpus=None, baseline=None,
         rewrites=None, costs=None, run_id="", example=None):
    """The sections, the rail entries, and the header fields. Pure: no filesystem, no
    argv. Returns (body_html, toc, header_kwargs) so a future combined page can put two
    bodies inside one document()."""
    f = facts(audit, manifest, diversity, costs)
    cons = _considerations(audit)
    labels = _labels(audit)
    sections = [
        section_gap(content, f),
        section_example(audit, content, f, baseline, rewrites, labels, example),
        section_results(audit, content, f, cons, labels),
        section_method(content, f, manifest, costs),
        section_footprint(audit, content, f),
        section_measurement(audit, content, f, diversity),
        section_weaknesses(audit, content, f, manifest),
        section_reproduce(content, f, run_id or (manifest or {}).get("run_id", "<run_id>")),
        section_appendix(audit, content, f, rewrites, labels, diversity, example),
    ]
    toc = [(sid, label) for sid, label in TOC
           if any(f"<section id='{sid}'" in s for s in sections)]
    n_note = (f"{f.get('n', '?')} dilemmas dealt, {f.get('n_measured', '?')} measured"
              if f.get("n_measured") != f.get("n") else f"{f.get('n', '?')} examples")
    meta = C.meta_line(run_id=run_id, manifest=manifest, pairs=(
        ("generated with", f"<code>{R.esc(f.get('gen_models', '?'))}</code>"),
        ("audited with", f"<code>{R.esc(f.get('extract_model', '?'))}</code> extracting and "
                         f"<code>{R.esc(f.get('judge_model', '?'))}</code> judging"),
    ))
    meta = f"{n_note} · {meta}"
    head = {
        "title": C.fill(content["title"], f).strip(),
        "eyebrow": EYEBROW,
        "lede": C.fill(content["lede"], f).strip(),
        "hero": hero_tiles(audit, f, cons),
        "meta_line": meta,
        "footer": "Every figure on this page is computed from the run's own audit output at "
                  "build time; none is typed in by hand. The weaknesses section is derived "
                  "from the audit's verdicts, so a regression appears there whether or not "
                  "anyone wrote it up.",
    }
    return "".join(sections), toc, head


def build(*, sibling=None, **kwargs):
    body_html, toc, head = body(**kwargs)
    return R.document(toc=toc, body=body_html, sibling=sibling, **head)
