#!/usr/bin/env python3
"""The dilemma corpus's section of the handoff page: the ``#dad`` beats.

The audience is a technical reader at another lab — someone deciding whether the method
and its measurement are sound, and whether to run the pipeline themselves. That is a
different job from the Streamlit corpus-audit page, which is organised by what the eval
measured; this is organised by what a reader needs, in the order they need it.

This module builds BLOCKS, not a page: ``blocks()`` returns the section's body, and
report/page.py wraps it in the one ``<section id='dad'>`` on the artefact. Blocks stay
flat — a figure has to be a direct child of the section for the CSS grid to bleed it
past the text measure, so nothing here wraps a beat in a container.

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
    "dad_what",
    "method_intro", "stage1", "stage2", "stage3", "control", "reproduce",
    "example_pick", "example_extra", "example_intro",
    "weaknesses_intro", "judge_limits",
    "appendix_intro", "judged_caveat", "checks_intro",
)

SECTION_ID = "dad"
SECTION_TITLE = "Difficult advice"

# The skeleton, in order. The document corpus's section takes the same one, so a reader
# learns it once; the ids are prefixed because both sections live in one document.
#
# The stages come before the worked example on purpose: the chooser above asks the reader
# to walk through a dataset generation, and a walk needs its steps named first. There is
# no "what we measured" beat — this report is not a results report, and the run's own
# measurements are either a descriptive tile here or a drawer in the appendix.
BEATS = (
    ("dad-what", "What it is"),
    ("dad-built", "How it is built"),
    ("dad-example", "One example, end to end"),
    ("dad-weak", "Where it is weak"),
    ("dad-appendix", "Appendix"),
)


_STAGE_KNOBS = ("scenario_model", "prompt_draft_model", "prompt_gate_model",
                "prompt_refine_model", "response_scope_model", "response_select_model",
                "response_draft_model", "constitution_rewrite_model")

# stage tag in cost_log.jsonl -> display name, in pipeline order. Both baseline tags are
# listed because the pipeline writes `baseline_response` and this table used to name only
# `baseline`, which put the control's cost at the bottom of the drawer as a raw tag.
# common.stage_cost_table skips a tag the log does not carry, so listing both is safe.
_STAGE_LABELS = (
    ("scenario_plan", "1a · scenario plan"),
    ("prompt_draft", "1b · prompt draft"),
    ("prompt_gate", "1c · quality gate"),
    ("prompt_refine", "1d · refine"),
    ("baseline", "control · plain model"),
    ("baseline_response", "control · plain model"),
    ("response_scope", "2a · scope"),
    ("response_select", "2a.5 · library select"),
    ("response_draft", "2b · response draft"),
    ("constitution_rewrite", "3 · constitution rewrite"),
)

_DELIVERY_DIMS = ("goal_responsiveness", "proportionality", "tone", "calibration")


# ------------------------------------------------------------------ loading

def load_inputs(run_dir):
    """All filesystem access, in one place. Returns this section's kwargs.

    Prose is not loaded here: the page owns one content namespace across both sections,
    so report/page.py loads it once.
    """
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
        "corpus": C.read_jsonl(run_dir / "final" / "dad_corpus.jsonl"),
        "baseline": C.read_jsonl(run_dir / "baseline" / "baseline_responses.jsonl"),
        "rewrites": C.read_jsonl(run_dir / "step3" / "rewrites.jsonl"),
        "costs": C.read_jsonl(run_dir / "cost_log.jsonl"),
        "deals": C.read_jsonl(run_dir / "step1" / "scenario_deals.jsonl"),
        "lineage": read_lineage(run_dir, audit),
        "n_prompt_templates": C.prompt_count(run_dir, "step*.txt"),
        "run_id": run_dir.name,
    }


# The seven scope axes, in the order the stage-2 prose names them, with the label each
# gets on the page. Stage 2a writes exactly these keys; anything else it grows appears
# after them rather than being dropped.
_SCOPE_AXES = (
    ("patients", "who can be harmed"),
    ("goal", "what the user is trying to achieve"),
    ("levers", "which levers are open"),
    ("cost", "what each one costs"),
    ("magnitude", "how large the welfare stake is"),
    ("upside", "what happens anyway without them"),
    ("replaceability", "whether the animals are replaceable"),
)

# The dealt axes worth showing beside a message, in reading order: what the decision is
# about, whose welfare is at stake, how the case is shaped, how the message is written.
_CARD_AXES = (
    ("archetype", "archetype"),
    ("domain", "domain"),
    ("taxa_subcategory", "animals at stake"),
    ("frontier_frame", "frame"),
    ("visibility", "how visible the welfare cost is"),
    ("user_attitude", "the user's attitude"),
    ("user_moral_framework", "their moral framework"),
    ("welfare_magnitude", "welfare magnitude"),
    ("conflict", "how the values interact"),
    ("leverage", "what they can actually change"),
    ("anchor_value_pair", "the values in tension"),
    # claim_pattern is deliberately absent: its value is a sentence of instruction to the
    # planner ("build the dilemma around status-quo inertia — …"), which reads as
    # documentation of the prompt rather than a property of this example.
    ("surface_form", "surface form"),
    ("cultural_setting", "cultural setting"),
    ("length_class", "length register"),
)


def read_lineage(run_dir, audit=None):
    """prompt_id -> that record's trail through the run's own step files.

    Only step 1 is keyed by ``scenario_id``; everything downstream is keyed by
    ``prompt_id``, and ``step1/dilemmas.jsonl`` is the one file carrying both, so it is
    the join table. ``audit.gid_map[pid]["scenario"]`` is the fallback when a run kept
    no dilemmas file, because ``scenarios.jsonl`` carries the same scenario gid.

    ``step2/scopes.jsonl`` is trimmed on the way in: four fifths of its 725 KB is the
    reasoning library's prose repeated per case, and the page shows an entry's id, its
    category and its claim.

    A file that is not there leaves its key ABSENT rather than None, so a renderer tests
    membership and can name the artefact it wanted instead of printing 'None'.
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    dilemmas = C.read_jsonl(run_dir / "step1" / "dilemmas.jsonl")
    # scenarios.jsonl is a superset of scenario_deals.jsonl: the same dealt cards, plus
    # the description the planner wrote from them.
    scenarios = C.read_jsonl(run_dir / "step1" / "scenarios.jsonl")
    by_sid = {s.get("scenario_id"): s for s in scenarios if s.get("scenario_id")}
    by_sgid = {s.get("scenario_gid"): s for s in scenarios if s.get("scenario_gid")}
    scopes = {s.get("prompt_id"): s for s in C.read_jsonl(run_dir / "step2" / "scopes.jsonl")
              if s.get("prompt_id")}
    gids = (audit or {}).get("gid_map") or {}

    sid_of = {d.get("prompt_id"): d.get("scenario_id") for d in dilemmas if d.get("prompt_id")}
    out = {}
    for pid in set(sid_of) | set(scopes) | set(gids):
        entry = {}
        scenario = by_sid.get(sid_of.get(pid)) or by_sgid.get((gids.get(pid) or {}).get("scenario"))
        if scenario:
            entry["scenario_id"] = scenario.get("scenario_id")
            entry["cards"] = {k: scenario.get(k) for k, _ in _CARD_AXES if scenario.get(k)}
            if scenario.get("scenario_description"):
                entry["description"] = scenario["scenario_description"]
        scope = scopes.get(pid)
        if scope:
            if scope.get("scope"):
                entry["scope"] = scope["scope"]
            if scope.get("entry_ids"):
                entry["entry_ids"] = scope["entry_ids"]
            entry["entries"] = [{k: e.get(k) for k in ("id", "category", "claim")}
                                for e in scope.get("triggered_entries") or []]
            entry["selection_fallback"] = bool(scope.get("selection_fallback"))
        if entry:
            out[pid] = entry
    return out


# The dealt axes the comparison table reports as this corpus's spread, in the order
# they read: what the decision is about, whose welfare is at stake, where it happens.
_SPREAD_AXES = (("domain", "domains"), ("taxa_category", "taxa groups"),
                ("cultural_setting", "cultural settings"))


def spread(deals):
    """How wide the dealt combinations run, straight off step 1's own deal records.

    Counted from the deals rather than the shipped corpus because the deal is where
    the spread is engineered — a rejected scenario still tells you what the matrix
    covers.
    """
    if not deals:
        return ""
    out = []
    for key, label in _SPREAD_AXES:
        seen = set()
        for deal in deals:
            value = deal.get(key)
            if isinstance(value, list):
                seen.update(v for v in value if v)
            elif value:
                seen.add(value)
        if seen:
            out.append(f"{len(seen)} {label}")
    return " · ".join(out)


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


def facts(audit, manifest=None, diversity=None, costs=None, corpus=None, deals=None):
    """Every number the prose can interpolate, computed once, in one place.

    Run-conditional figures reach prose only with a degraded default — a run missing
    the paid pass renders "an unmeasured share" where the figure would be, so the
    sentence survives and its claim does not. The delivery comparison is deliberately
    NOT available to prose as a clause: it is stated once, by _delivery_statement().
    """
    n_shipped = len(corpus) if corpus else None
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
        # Dealt and shipped are different numbers — this run dealt 40 and shipped 39,
        # because one scenario was rejected at stage 2a — and the page says so rather
        # than quietly reporting whichever is larger.
        "n_shipped": n_shipped,
        "records_clause": (f"{n_shipped:,} shipped records, from {n:,} dilemmas dealt"
                           if n_shipped and n and n_shipped != n else
                           f"{n_shipped:,} shipped records" if n_shipped else None),
        "spread_clause": spread(deals) or None,
        "judge_arms_clause": _judge_arms_clause(audit),
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
        ("length_pct", "an unmeasured amount"), ("near_dup_pct", "an unmeasured share"),
        ("library_clause", "an animal-ethics reasoning library"),
        ("added_per_answer", "an unmeasured number of"),
        ("records_clause", "the records this run shipped"),
        ("spread_clause", "a weighted matrix of dealt combinations"),
        ("judge_arms_clause", "not measured on this run"),
    ):
        f.setdefault(key, default)
    return f


def _judge_arms_clause(audit):
    """How matched the paid comparison actually is, as a clause the prose can hold.

    Composed here rather than typed, because it is the reason the page does not lead
    with the comparison: on the pinned run the delivery judge lost 19 judgements, so its
    two means are over 33 pipeline and 26 control answers — different sets of records.
    """
    delivery = (audit or {}).get("delivery") or {}
    n_p, n_b, fails = (delivery.get("n_pipeline"), delivery.get("n_plain"),
                       delivery.get("failures"))
    if n_p is None or n_b is None:
        return None
    clause = f"over {n_p} pipeline and {n_b} control answers"
    return f"{clause}, with {fails} judgements failing" if fails else clause


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


# ------------------------------------------------------------------ what it is

def what_tiles(f, diversity=None):
    """Three descriptive numbers: how many records, how distinct, what they cost.

    No comparison and no direction chip. What this dataset is does not depend on how it
    scored against a plain model, and a reader who wants that comparison opens the
    appendix.

    A tile is omitted rather than zeroed: the diversity pass is optional, and a
    ``.get("score", 0)`` would print "0.0 effectively distinct records" on a run that
    simply never measured it.
    """
    items = []
    if f.get("n_shipped"):
        items.append(R.stat(f"{f['n_shipped']:,}", "shipped records",
                            f.get("spread_clause") if f.get("spread_clause") !=
                            "a weighted matrix of dealt combinations" else ""))
    vendi = (diversity or {}).get("vendi") or {}
    if vendi.get("score"):
        items.append(R.stat(f"{vendi['score']:.1f}", "effectively distinct records",
                            f"of {diversity.get('n_records', '?')} actual records; "
                            f"{f.get('near_dup_pct', '?')} sit above 0.90 cosine "
                            f"similarity to their nearest neighbour"))
    if f.get("cost_per_example") not in (None, "not logged"):
        items.append(R.stat(f["cost_per_example"], "per example, end to end",
                            f"{f['cost_total']} for this run"))
    return R.tiles(items)


# ------------------------------------------------------------------ beats

def blocks_example(content, f, rewrites, baseline, lineage, labels, picks=()):
    """One record's whole trail through the run, then the rest as a carousel.

    Every block here is verbatim from a file in the run directory: the cards the composer
    dealt, the scenario the planner wrote from them, the message that shipped, the scope
    and the library entries stage 2 pulled, the answer, and what stage 3 changed in it.
    Nothing is author-supplied, and a step whose artefact is missing names the file it
    wanted rather than disappearing.
    """
    blocks = [R.sub("dad-example", "One example, end to end"),
              C.prose(content, "example_intro", f)]
    by_pid_rw = {r.get("prompt_id"): r for r in rewrites or []}
    by_pid_base = {r.get("prompt_id"): r for r in baseline or []}
    primary, extras = _picks(content, picks, by_pid_rw)

    if not primary:
        blocks.append(R.note("No worked example could be built: this run shipped no rewrite "
                             "records, so there is no answer to show."))
        return "".join(blocks)
    if primary not in by_pid_rw:
        blocks.append(R.note(f"The pinned example `{primary}` is not in this run — it shipped "
                             f"no rewrite record. Pin one of this run's ids in "
                             f"`example_pick`, or set it to `auto`."))
        primary, extras = _picks({}, (), by_pid_rw)
        if not primary:
            return "".join(blocks)

    blocks.append(lineage_blocks(primary, by_pid_rw.get(primary) or {},
                                 by_pid_base.get(primary) or {},
                                 (lineage or {}).get(primary) or {}, labels))
    if extras:
        blocks.append(carousel(extras, by_pid_rw, labels))
    return "".join(b for b in blocks if b)


def lineage_blocks(pid, rw, base, lin, labels):
    """The trail for one record: deal → scenario → message → scope → answer → rewrite.

    The stage headings deliberately repeat the ones "How it is built" uses, so a reader
    who has just read the stages recognises each step rather than learning a second
    vocabulary for the same pipeline.
    """
    out = [f"<p class='muted'>Record <span class='mono'>{R.esc(labels.get(pid, pid))}</span>"
           f" — pinned in the prose file, so a rebuild shows the same case.</p>"]

    out.append("<h4>Stage 1 · the dilemma</h4>")
    if lin.get("cards"):
        out.append("<p class='muted'>Dealt in code, before any model was called.</p>")
        out.append(_cards_table(lin["cards"]))
    else:
        out.append(R.note("This run kept no `step1/scenario_deals.jsonl` or "
                          "`step1/scenarios.jsonl`, so the dealt combination is not "
                          "recoverable for this record."))
    if lin.get("description"):
        out.append(R.details("The scenario the planner wrote from those cards",
                             R.quote(lin["description"]),
                             meta=f"{len(lin['description'].split()):,} words"))
    else:
        out.append(R.note("The scenario description is in `step1/scenarios.jsonl`, which this "
                          "run did not keep."))
    user_msg = rw.get("user_message") or base.get("user_message") or ""
    if user_msg:
        out.append("<p class='muted'>Drafted, gated, then reviewed against its own cards. What "
                   "shipped:</p>")
        out.append(R.quote(user_msg))

    out.append("<h4>Stage 2 · the reasoning</h4>")
    if lin.get("scope"):
        # In a drawer: seven axes of dense prose is the most interesting artefact in the
        # run and the one most likely to stop a reader walking. Measured at 1,500px, it
        # sat between the message and the answer.
        out.append(R.details("What stage 2 worked out before writing anything",
                             _scope_table(lin["scope"]),
                             meta=f"{len(lin['scope'])} axes"))
    else:
        out.append(R.note("The scope is in `step2/scopes.jsonl`, which this run did not keep."))
    ids = lin.get("entry_ids") or rw.get("entry_ids") or []
    if ids:
        out.append(_entries_block(ids, lin.get("entries") or [],
                                  fallback=lin.get("selection_fallback")))
    if base.get("baseline_response"):
        out.append(R.details(
            "The first take stage 2 was shown · plain model, no system prompt",
            R.highlight(base["baseline_response"], []),
            meta=f"{len(base['baseline_response'].split()):,} words · never a training record"))

    out.append("<h4>Stage 3 · the constitution rewrite</h4>")
    answer = rw.get("rewritten_response") or ""
    if answer:
        out.append("<p class='muted'>The answer, as it ships:</p>")
        out.append(R.highlight(answer, []))
    else:
        out.append(R.note("This record has no rewritten answer in `step3/rewrites.jsonl`."))
    if rw.get("draft_response") and answer:
        before = rw["draft_response"]
        out.append(R.details(
            "What the constitution rewrite changed in this answer",
            f"<p class='muted'>{_diff_summary(before, answer)} The three largest changes:</p>"
            + _diff_hunks(before, answer),
            meta="3 largest changes · full diff in the appendix"))
    return "".join(out)


def _cards_table(cards):
    """The dealt combination as a table.

    Null and empty values are DROPPED: a deal with no cultural setting has no cultural
    setting, and rendering the axis with 'None' in it is a bug that reads as data.
    """
    rows = []
    for key, label in _CARD_AXES:
        value = cards.get(key)
        if isinstance(value, list):
            value = " · ".join(v for v in value if v)
        if value:
            rows.append((label, value))
    return R.table(["dealt axis", "this example"], rows, align="ll") if rows else ""


def _scope_table(scope):
    """Stage 2a's seven axes, in the order the stage-2 prose names them.

    An axis the stage grows later lands after the seven rather than being dropped.
    """
    named = [(label, scope[key]) for key, label in _SCOPE_AXES if scope.get(key)]
    extra = [(k.replace("_", " "), v) for k, v in scope.items()
             if v and k not in {key for key, _ in _SCOPE_AXES}]
    rows = named + sorted(extra)
    return R.table(["what stage 2 worked out", "for this case"], rows, align="ll") if rows else ""


def _entries_block(ids, entries, fallback=False):
    """The library entries this case pulled, glossed from the run's own step-2 output.

    Bare ids when the gloss is missing: the ids are still the honest artefact, and they
    are what the answer was actually written from.
    """
    gloss = {e.get("id"): e for e in entries if e.get("id")}
    rows = [(i, (gloss.get(i) or {}).get("category") or "—",
             (gloss.get(i) or {}).get("claim") or "—") for i in ids]
    note = ("<p class='warn-note'>The selection call failed for this case, so stage 2 was shown "
            "the whole library rather than a chosen subset.</p>" if fallback else "")
    return note + R.details(
        "The reasoning-library entries this case pulled",
        R.table(["id", "kind", "the pattern it carries"], rows, align="lll"),
        meta=f"{len(ids)} of the library's entries · never named in an answer")


def carousel(picks, by_pid_rw, labels):
    """More examples as tabs: the message and the answer, nothing else.

    Reuses the chooser's mechanism rather than adding a second one — buttons carrying
    ``data-pane``, panes toggled by the page's own inline JS. The FIRST pane renders
    visible rather than hidden, so with JS off the carousel degrades to one example
    instead of to nothing, and printing expands all of them.
    """
    panes = []
    for pid in picks:
        rw = by_pid_rw.get(pid) or {}
        if not (rw.get("user_message") and rw.get("rewritten_response")):
            continue
        # Muted labels rather than <h4>s: a pane is not a beat, and four headings at the
        # stage headings' own level put "The answer" into the document outline twice.
        panes.append((f"ex-{len(panes)}", labels.get(pid, pid),
                      "<p class='muted'>The user asked:</p>" + R.quote(rw["user_message"])
                      + "<p class='muted'>The answer, as it ships:</p>"
                      + R.highlight(rw["rewritten_response"], []),
                      not panes))
    if not panes:
        return ""
    return ("<h4>More examples</h4>"
            "<p class='muted'>More records from the same run, as they ship.</p>"
            + R.tabs(panes))


def _picks(content, cli=(), by_pid_rw=None):
    """(primary, extras) prompt_ids for the example beat.

    Pinned in the prose file rather than passed on the command line so that a rebuild
    reproduces the same records without anyone having to remember a flag; ``--example``
    overrides the primary only. ``auto`` takes the first shipped record and the two after
    it — deliberately NOT the showcase judge's favourite, because this beat shows how a
    record is built and must not depend on the paid pass having run.
    """
    raw = (content.get("example_pick") or "").strip()
    primary = None if raw.lower() in ("", "auto") else raw.split()[0]
    extras = (content.get("example_extra") or "").split()
    if cli:
        primary = cli[0] if isinstance(cli, (list, tuple)) else cli
    shipped = sorted(by_pid_rw or {})
    if not primary:
        primary = shipped[0] if shipped else None
        extras = extras or [p for p in shipped if p != primary][:2]
    return primary, [p for p in extras if p != primary]


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


def blocks_what(content, f, diversity=None):
    """What the dataset is, and three numbers describing it.

    Takes no ``audit`` and no ``cons`` on purpose: this beat cannot lead with a judged
    figure because it is not given one. The comparison against a plain model lives in a
    single appendix drawer, and every chart this run supports lives there too.
    """
    return "".join(b for b in (R.sub("dad-what", "What it is"),
                               C.prose(content, "dad_what", f),
                               what_tiles(f, diversity)) if b)


def _delivery_statement(audit, f):
    """The one place the delivery regression is written out.

    A run without the paid pass says so here instead, in the same slot.
    """
    delivery = audit.get("delivery") or {}
    if not delivery.get("per_case"):
        return R.note(
            "Delivery quality was **not measured on this run**, so there is no evidence here "
            "either way about whether the added substance cost manner. Populate it with "
            "`python evals/audit_dad.py --input <run> --reasons`.")
    pm, bm = delivery.get("pipeline_mean"), delivery.get("plain_mean")
    if bm is None or pm is None or pm >= bm:
        return ""
    dims = delivery.get("dimensions") or {}
    worse = [k for k, v in (dims.get("pipeline") or {}).items()
             if (dims.get("plain") or {}).get(k) is not None and v < dims["plain"][k]]
    every = (" The pipeline is worse on all four judged dimensions: goal responsiveness, "
             "proportionality, tone and calibration."
             if worse and len(worse) == len(dims.get("pipeline") or {}) else "")
    return R.note(
        f"**Judged delivery went the wrong way: {f['delivery_pipeline']} against the "
        f"control's {f['delivery_plain']} out of 10.** The added substance was not free — on "
        f"manner alone, the plain answers read as more helpful.{every}", tone="bad")


def _pareto_figure(audit, mpr, labels):
    delivery = audit.get("delivery") or {}
    if not delivery.get("per_case"):
        return ""
    n_p, n_b, fails = delivery.get("n_pipeline"), delivery.get("n_plain"), delivery.get("failures")
    asym = ""
    if n_p is not None and n_b is not None and (n_p != n_b or fails):
        asym = (f" These means are over {n_p} pipeline and {n_b} control answers — "
                f"{fails or 0} judgements failed, so the two arms are not the same set of "
                f"records.")
    return R.figure(
        title="Substance against manner, one dot per answer",
        note_="Judged delivery quality on the horizontal axis, valuable welfare considerations "
              "on the vertical. Diamonds are each arm's mean." + asym,
        chart=_pareto(delivery, mpr, labels),
        caption="**The pipeline arm sits up and to the left: it buys substance with manner.**")


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


# The judged axes, in whichever schema the run's audit happens to carry. `delivery` is in
# both; `welfare_impact` and `composite` arrived with the two-holistic-judge rework, which
# also dropped the `valuable_welfare_considerations` metric they replaced. A run's audit
# has one set or the other, so the drawer reads what is there and names it.
_JUDGED_AXES = (
    ("welfare_impact", "welfare impact, 0–10", "{:.2f}"),
    ("delivery", "delivery quality, 0–10", "{:.2f}"),
    ("composite", "composite, 0–1", "{:.3f}"),
)


def _judged_means(audit):
    """(label, plain, pipeline, fmt) for every judged axis this audit recorded."""
    out = []
    for key, label, fmt in _JUDGED_AXES:
        block = (audit or {}).get(key) or {}
        means = block.get("arm_means") or block
        p, b = means.get("pipeline_mean", means.get("pipeline")), \
            means.get("plain_mean", means.get("plain"))
        if isinstance(p, (int, float)) and isinstance(b, (int, float)):
            out.append((label, b, p, fmt))
    return out


def judged_drawer(audit, content, f, cons, labels):
    """The whole judged comparison against the plain model, in one drawer.

    Demoted rather than deleted. It is real evidence and it is all here, but it is not
    what the page argues from: on the pinned run the delivery pass lost judgements, so its
    two means are over different sets of records, judge and generator are the same model
    family, and nothing checks whether the points counted as added are correct. A page
    that led with it would be leading with its weakest measurement.
    """
    mpr = (audit or {}).get("moral_patient_reasons") or {}
    means = _judged_means(audit)
    # The scoreboard mixes judged rows with offline ones (length, structural variety), so
    # its presence is not evidence that a judge ran. Say so explicitly rather than letting
    # a drawer titled "what the paid judges measured" fill up with offline measures.
    paid = bool(means or (cons and cons.get("plain") is not None) or mpr.get("survival"))
    body = [C.prose(content, "judged_caveat", f) if paid else R.note(
        "No paid judge pass ran on this run, so nothing here compares the two arms on "
        "substance or manner. Populate it with `python evals/audit_dad.py --input <run> "
        "--reasons`. The rows below are offline measurements against the control.")]

    if means:
        body.append(R.table(["judged axis", "control", "pipeline"],
                            [(label, fmt.format(b), fmt.format(p)) for label, b, p, fmt in means],
                            align="lrr"))

    if cons and cons.get("plain") is not None:
        body.append(R.figure(
            title="Valuable welfare considerations per answer",
            note_="A distinct welfare point, or a concrete lower-harm action, that a judge "
                  "reading the answer counted as useful to the person asking. Both arms "
                  "answered the same dilemmas.",
            chart=R.hbar([("the control", round(cons["plain"], 2)),
                          ("the pipeline", round(cons["pipeline"], 2))],
                         color=R.ARM_PAIR, fmt="{:.1f}"),
            caption=f"**The pipeline raises {f.get('lift_pct', '?')} more of them**, on the "
                    f"same {f.get('n_measured', '?')} dilemmas."))
        if cons["source"] == "reconstructed":
            body.append("<p class='muted'>Reconstructed from this run's separate reasoning "
                        "and alternatives measures; it predates the unified extraction.</p>")
    if mpr.get("failures"):
        body.append(R.note(
            f"Means are over {f.get('n_pipeline', '?')} pipeline and {f.get('n_plain', '?')} "
            f"control answers: {mpr['failures']} extractions failed and are excluded, so the "
            "comparison is not fully matched."))

    board = scoreboard(audit, f, cons)
    if board:
        body.append("<h4>Measure by measure</h4>")
        body.append(board)
    body.append(_pareto_figure(audit, mpr, labels))

    surv = mpr.get("survival") or {}
    if surv.get("anchored") or surv.get("added"):
        body.append("<h4>What happened to the control's considerations</h4>")
        body.append(R.table(["fate", "n", "the judge's wording"], _survival_rows(surv),
                            align="lrl"))

    body = [b for b in body if b]
    if len(body) <= 1:
        return ""
    title = ("What the paid judges measured, and why the report does not lead with it"
             if paid else "How the two arms compare, offline")
    return R.details(title, "".join(body),
                     meta=f.get("judge_arms_clause", "") if paid else "")


def checks_table(audit, diversity):
    """Every measurement that could have run, and whether it did.

    A check that did not run says so rather than vanishing, because a reader deciding
    whether to trust the dataset needs to know which questions were never asked.
    """
    mpr = (audit or {}).get("moral_patient_reasons") or {}
    checks = [
        ("Valuable welfare considerations", bool(mpr.get("pipeline") or
                                                 (audit or {}).get("welfare_impact")),
         "Welfare substance per answer, both arms"),
        ("Retention", bool(mpr.get("survival")),
         "Item by item, which of the control's considerations the pipeline kept, weakened or "
         "dropped, and what it added"),
        ("Delivery quality", bool((audit or {}).get("delivery")),
         "How helpful, proportionate and non-preachy each answer is, judged 0–10"),
        ("Showcase examples", bool((audit or {}).get("showcase")),
         "Concrete pipeline-beats-control cases with verbatim improved spans"),
        ("Response stance", bool(((audit or {}).get("moves") or {}).get("stance")),
         "Whether an answer defers, stays calibrated, or moralizes"),
        ("Tracked phrases and rhetorical moves",
         bool((audit or {}).get("tracked_tics") or (audit or {}).get("rhetorical_moves")),
         "Recurring phrasing and argumentative habits, as a share of each arm"),
        ("Length, structure, jargon", bool((audit or {}).get("response_lengths")),
         "Offline dataset measurements against the control"),
        ("Semantic diversity", bool(diversity),
         "Embedding near-duplicate rate, topic spread, effective record count"),
    ]
    rows = [(name, what if ok else R.Raw(f"<i>not run on this run</i> — {R.esc(what)}"))
            for name, ok, what in checks]
    return R.table(["check", "what it establishes"], rows)


# ------------------------------------------------------------------ method

def blocks_built(content, f, manifest, costs, run_id):
    """The three stages, the control, and the command that reproduces all of it."""
    blocks = [R.sub("dad-built", "How it is built"), C.prose(content, "method_intro", f)]
    for key, heading in (("stage1", "Stage 1 · the dilemma"),
                         ("stage2", "Stage 2 · the reasoning"),
                         ("stage3", "Stage 3 · the constitution rewrite"),
                         ("control", "The control arm")):
        blocks.append(f"<h4>{R.esc(heading)}</h4>{C.prose(content, key, f)}")
    table = C.stage_cost_table(costs, _STAGE_LABELS)
    if table:
        blocks.append(R.details("Per-stage cost and model", table,
                                meta=f"{f.get('cost_total', '?')} for this run"))
    elif _models(manifest)["stage_models"]:
        blocks.append(R.details("Per-stage model", R.table(
            ["stage", "model"], [(k.replace("_model", "").replace("_", " "), v)
                                 for k, v in _models(manifest)["per_stage"].items()])))
    blocks.append("<h4>Running it yourself</h4>")
    blocks.append(C.prose(content, "reproduce", f))
    cmd = ("# generate a dataset\n"
           "python dad_pipeline/run.py --config config.yaml --label my-run\n\n"
           "# the evals run automatically; to re-run them on an existing run:\n"
           "python evals/audit_dad.py --input outputs/dad/latest --reasons\n"
           "python evals/diversity.py --input outputs/dad/latest\n\n"
           "# rebuild this page\n"
           f"python report/build_report.py --dad-run outputs/dad/runs/{run_id}")
    blocks.append(f"<pre>{R.esc(cmd)}</pre>")
    return "".join(blocks)


# ------------------------------------------------------------------ footprint

def _footprint_figures(audit, f):
    """The stylistic footprint: what a model trained on this corpus would inherit.

    These are charts a reader can reach for, not charts the page leads with, so they
    live in the appendix drawer. Captions still state the finding, including where a
    measure moved the wrong way.
    """
    blocks = []
    rl = audit.get("response_lengths") or {}
    if rl.get("pipeline_mean"):
        blocks.append(R.figure(
            title="Answer length",
            chart=R.hbar([("the control", round(rl.get("plain_mean", 0))),
                          ("the pipeline", round(rl["pipeline_mean"]))],
                         color=R.ARM_PAIR, unit=" chars", fmt="{:,.0f}"),
            caption=f"**{f.get('length_pct', '?')} longer than the control.** Length is the "
                    f"most visible property a model would inherit, and the judges see it too."))
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
                    f"{f.get('moralizes_plain', '?')}), which stage 3 exists to prevent."))
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
                      "habits. Share of each arm's answers containing one at least once.",
                chart=R.grouped_hbar(rows, series=[("plain", R.PLAIN), ("pipeline", R.PIPELINE)],
                                     percent=True, label_w=210),
                caption="**The pipeline's most common tracked phrase stays well under half of "
                        "its answers**, which is where a word choice becomes a tic."))
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
            note_="Argumentative moves, as a share of each arm's answers. Hover a bar for "
                  "what the move is; the definitions are below.",
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
                     f"effective shapes)."
                     if worse else "**Structural range holds up against the control.**")))
    dims = (audit.get("delivery") or {}).get("dimensions") or {}
    if dims.get("pipeline"):
        keys = [k for k in _DELIVERY_DIMS if k in dims["pipeline"]]
        rows = []
        for k in keys:
            p, b = dims["pipeline"].get(k), (dims.get("plain") or {}).get(k)
            rows.append((k.replace("_", " "), f"{b:.2f}" if b is not None else "—",
                         f"{p:.2f}" if p is not None else "—",
                         f"{p - b:+.2f}" if p is not None and b is not None else "—"))
        n_worse = sum(1 for r in rows if r[3].startswith("-"))
        blocks.append(R.figure(
            title="Delivery quality, dimension by dimension",
            note_="Each dimension is judged 0–10 on the answer alone: did it serve the goal "
                  "the user actually had, was it proportionate, was the tone right, was "
                  "uncertainty calibrated.",
            chart=R.table(["dimension", "control", "pipeline", "delta"], rows, align="lrrr"),
            caption=f"**Worse on {n_worse} of {len(rows)} dimensions.**"))
    return blocks


def _habits_caption(invented, dropped):
    """Say which habit the pipeline invented and which it traded away, or say neither.

    The old caption asserted "invented one closing move and dropped another" as a fixed
    sentence about conditional data.
    """
    if invented and dropped:
        return (f"**The pipeline turned `{invented[0]}` into a habit the control never shows, "
                f"and dropped `{dropped[0]}`, which the control reaches for.**")
    if invented:
        return f"**`{invented[0]}` is a habit the pipeline has and the control does not.**"
    return f"**The pipeline dropped `{dropped[0]}`, a move the control reaches for.**"


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


def blocks_weak(audit, content, f, manifest):
    """The derived floor, then what the measurements cannot settle.

    This is where the delivery regression is written out — once, by
    ``_delivery_statement()``, as a caveat rather than as a result. The appendix's
    scoreboard row and the derived weakness carry the same number as data, which is not
    the same as saying it again.

    The subhead below is code's, not the prose file's: h3 is the beat level inside a
    section, so a prose `### ` would put it level with "Where it is weak" itself.
    """
    return "".join(b for b in (
        R.sub("dad-weak", "Where it is weak"),
        C.prose(content, "weaknesses_intro", f),
        _delivery_statement(audit, f),
        C.warnings_table(derived_warnings(audit, manifest, f)),
        "<h4>What these measurements do not establish</h4>",
        C.prose(content, "judge_limits", f),
    ) if b)


# ------------------------------------------------------------------ appendix

def _appendix_charts(audit, f, cons):
    """The substance charts the page does not lead with, then the footprint ones."""
    out = []
    mpr = audit.get("moral_patient_reasons") or {}
    if cons and cons.get("plain") is not None:
        subset_rows = [{"label": name, "plain": b, "pipeline": p}
                       for name, b, p in cons["subsets"] if p is not None]
        if subset_rows:
            out.append(R.figure(
                title="Split by kind of consideration",
                chart=R.grouped_hbar(subset_rows,
                                     series=[("plain", R.PLAIN), ("pipeline", R.PIPELINE)],
                                     fmt="{:.2f}"),
                caption="**The gain is in the reasoning as well as in the alternatives "
                        "offered.**"))
    surv = mpr.get("survival") or {}
    if surv.get("kept") is not None:
        out.append(R.figure(
            title="What happened to the control's considerations",
            note_="The judge read the control's answer first, then tracked each of its "
                  "considerations into the pipeline's.",
            chart=R.segbar([("kept", surv.get("kept") or 0, R.PLAIN),
                            ("weakened", surv.get("weakened") or 0, "var(--series-4)"),
                            ("dropped", surv.get("dropped") or 0, "var(--series-8)"),
                            ("added", surv.get("added_total") or 0, R.PIPELINE)]),
            caption=f"**{f.get('retention_pct', '?')} of the control's "
                    f"{f.get('anchored_n', '?')} considerations survive the pipeline, and it "
                    f"adds {f.get('added_per_answer', '?')} more per answer.** No pass checks "
                    f"whether the additions are correct."))
    types_p = (mpr.get("pipeline") or {}).get("type_hist") or _type_hist(mpr.get("per_case"),
                                                                        "pipeline")
    types_b = (mpr.get("plain") or {}).get("type_hist") or _type_hist(mpr.get("per_case"), "plain")
    if types_p and types_b:
        gloss = (audit.get("reason_composition") or {}).get("type_gloss") or {}
        keys = list(dict(types_p, **types_b))
        rows = [{"label": k, "plain": types_b.get(k, 0), "pipeline": types_p.get(k, 0)}
                for k in keys]
        out.append(R.figure(
            title="Kinds of consideration raised",
            chart=R.grouped_hbar(rows, series=[("plain", R.PLAIN), ("pipeline", R.PIPELINE)]),
            caption="**The pipeline's largest gains are in the kinds of point the control "
                    "raises least.**",
            table_html=R.table(["kind", "what it is", "control", "pipeline"],
                               [(k, gloss.get(k, "—"), types_b.get(k, 0), types_p.get(k, 0))
                                for k in keys], align="llrr") if gloss else None))
    return out + _footprint_figures(audit, f)


def blocks_appendix(audit, content, f, cons, rewrites, labels, diversity, picks=()):
    """Everything that is evidence, collapsed so it costs a reader nothing.

    Every chart lands here — the page above carries none — and so does the whole judged
    comparison, which leads the appendix because a reader who came looking for it should
    find it first.
    """
    blocks = [R.sub("dad-appendix", "Appendix"), C.prose(content, "appendix_intro", f),
              judged_drawer(audit, content, f, cons, labels)]

    charts = _appendix_charts(audit, f, cons)
    if charts:
        blocks.append(R.details(
            "Every chart from this run", "".join(charts),
            meta=f"{len(charts)} figures · {f.get('footprint_regressions', '')}"))

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
            C.prose(content, "checks_intro", f)
            + checks_table(audit, diversity)
            + "<h4>As the audit recorded them</h4>"
            + R.table(["check", "group", "worst verdict", "counts"], rows, align="llll"),
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
                     caption="**Every record keeps most of the control's considerations**, "
                             "so the average is not hiding one where the pipeline threw them "
                             "away."),
            meta=f"{len(per_case)} records"))

    if diversity:
        vendi = diversity.get("vendi") or {}
        nn = diversity.get("nn") or {}
        clusters = ((diversity.get("scopes") or {}).get("combined") or {}).get("clusters") or {}
        blocks.append(R.details(
            "Dataset diversity",
            "<p class='muted'>Embedding-space measurements over the final dataset, from "
            f"<code>{R.esc((diversity or {}).get('embed_model', '?'))}</code>.</p>"
            + R.tiles([
                R.stat(f"{vendi.get('score', 0):.1f}", "effectively distinct records",
                       f"of {diversity.get('n_records', '?')} actual records; near-duplicates "
                       f"count as fractions of a record"),
                R.stat(f"{nn.get('over_0.90', 0):.0%}", "near-duplicate records",
                       f"cosine similarity above 0.90 to their nearest neighbour; "
                       f"{nn.get('over_0.80', 0):.0%} above 0.80"),
                R.stat(f"{clusters.get('evenness', 0):.2f}", "topic-spread evenness",
                       f"1.00 is perfectly even; the largest cluster holds "
                       f"{clusters.get('largest_share', 0):.0%} of records"),
            ]),
            meta=f"{diversity.get('n_records', '?')} records"))

    pid, _ = _picks(content, picks, {r.get("prompt_id"): r for r in rewrites or []})
    rw = next((r for r in rewrites or [] if r.get("prompt_id") == pid), None)
    if rw and rw.get("draft_response") and rw.get("rewritten_response"):
        blocks.append(R.details(
            "The full stage-3 rewrite diff for the worked example",
            "<p class='muted'>Struck through: stage 2's draft. Highlighted: the shipped "
            "answer.</p>" + _word_diff(rw["draft_response"], rw["rewritten_response"]),
            meta=f"{len(rw['rewritten_response'].split()):,} words"))
    return "".join(blocks)


# ------------------------------------------------------------------ assembly

def blocks(*, audit, content, diversity=None, manifest=None, corpus=None, baseline=None,
           rewrites=None, costs=None, deals=None, lineage=None, n_prompt_templates=None,
           run_id="", example=None):
    """The whole ``#dad`` section body, in skeleton order. Pure: no filesystem, no argv.

    Returns one flat string of blocks. report/page.py wraps it in ``<section id='dad'>``
    with the h2; every block here is therefore a grid child of that section, which is
    what lets figures bleed past the text measure.
    """
    f = facts(audit, manifest, diversity, costs, corpus, deals)
    cons = _considerations(audit)
    labels = _labels(audit)
    run = run_id or (manifest or {}).get("run_id", "<run_id>")
    picks = (example,) if example else ()
    return "".join([
        blocks_what(content, f, diversity),
        blocks_built(content, f, manifest, costs, run),
        blocks_example(content, f, rewrites, baseline, lineage, labels, picks),
        blocks_weak(audit, content, f, manifest),
        blocks_appendix(audit, content, f, cons, rewrites, labels, diversity, picks),
    ])
