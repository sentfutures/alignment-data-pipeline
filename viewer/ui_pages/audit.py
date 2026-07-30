"""Corpus audit: the offline corpus-level audit report for the selected run.

Renders <run>/audit/audit_report.json. The report's ``sections`` (rows +
verdicts + group/gloss) are written by evals/audit_dad.py itself, so the
thresholds live in one place and this page shows exactly what the terminal
report showed.

New audit sections need no viewer change: give ``_section()`` a ``group`` and
``gloss`` in the eval and this page buckets, glosses, and tallies them
automatically (old reports fall back to ``rendering.AUDIT_SECTION_META``).
Only add a block here when a section needs a custom chart.
"""

import html as _html
import re
import sys
from pathlib import Path

import altair as alt
import pandas as pd

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from viewer import loader, rendering
from viewer.ui_pages import common

st.title("Corpus audit")

run = common.pick_run()
if run is None:
    st.stop()

st.markdown(f"**{run.label or run.run_id}** · `{run.run_id}`")

_AUDIT_SCRIPTS = {"dad": "audit_dad.py", "sdf": "audit_sdf.py"}
cmd = f"python evals/{_AUDIT_SCRIPTS.get(run.pipeline, 'audit_dad.py')} --input {run.run_dir}"

def _grouped_arm_chart(rows: list[dict], value_label: str) -> alt.Chart:
    """Side-by-side plain-vs-pipeline bars. Built via Altair (not st.bar_chart)
    with interactivity OFF — the built-in charts scroll-zoom, which pans the
    y-axis into nonsense on an accidental scroll."""
    df = pd.DataFrame(rows).melt(id_vars="record", var_name="arm", value_name=value_label)
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("record:N", title="record"),
        xOffset=alt.XOffset("arm:N", sort=list(rendering.AUDIT_ARM_COLUMNS)),
        y=alt.Y(f"{value_label}:Q", title=value_label),
        color=alt.Color("arm:N", title="", scale=alt.Scale(
            domain=list(rendering.AUDIT_ARM_COLUMNS),
            range=list(rendering.AUDIT_ARM_COLORS))),
        tooltip=["record", "arm", alt.Tooltip(value_label, title=value_label)],
    )


def _grouped_barh(df: pd.DataFrame, cat_field: str, cat_title: str,
                  percent: bool = False) -> alt.Chart:
    """Horizontal plain-vs-pipeline grouped bars, one group per category,
    sorted by value. Backs the tracked-tic frequency view that replaces the
    old wall of gray detail captions. percent=True plots a 0-1 `share` column
    as % of each arm's responses — the honest comparison when the arms have
    different sizes."""
    field, title = ("share", "share of responses") if percent else ("count", "responses")
    return alt.Chart(df).mark_bar().encode(
        y=alt.Y(f"{cat_field}:N", title=cat_title, sort="-x"),
        yOffset=alt.YOffset("arm:N", sort=list(rendering.AUDIT_ARM_COLUMNS)),
        x=alt.X(f"{field}:Q", title=title,
                axis=alt.Axis(format="%") if percent else alt.Axis(),
                scale=alt.Scale(domain=[0, 1]) if percent else alt.Scale()),
        color=alt.Color("arm:N", title="", scale=alt.Scale(
            domain=list(rendering.AUDIT_ARM_COLUMNS),
            range=list(rendering.AUDIT_ARM_COLORS))),
        tooltip=[alt.Tooltip(cat_field, title=cat_title or "phrase"), "arm",
                 alt.Tooltip(f"{field}:Q", title=title,
                             format=".0%" if percent else "d")],
    )


# --- shared redundancy/spread/cloud charts, used by BOTH the semantic and the
# lexical diversity sections (same visuals, different feature space) -----------

def _nn_hist(sims: list, rule_at: float, x_title: str) -> alt.Chart:
    bars = alt.Chart(pd.DataFrame({"sim": sims})).mark_bar(color="#4C78A8").encode(
        x=alt.X("sim:Q", bin=alt.Bin(maxbins=20), title=x_title, scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("count()", title="records"))
    rule = alt.Chart(pd.DataFrame({"x": [rule_at]})).mark_rule(
        strokeDash=[5, 3], color="#E5484D").encode(x="x:Q")
    return (bars + rule).properties(height=210)


def _cluster_bars(sizes: list) -> alt.Chart:
    df = pd.DataFrame({"cluster (sorted)": range(1, len(sizes) + 1), "size": sizes})
    return alt.Chart(df).mark_bar(color="#3FB366").encode(
        x=alt.X("cluster (sorted):O", axis=alt.Axis(labels=len(sizes) <= 12)),
        y="size:Q").properties(height=210)


def _cloud_scatter(cloud: list) -> alt.Chart:
    # No axis titles: the axes are PCA layout directions with no nameable
    # meaning ("PC1"/"PC2" read as jargon) — only the distances between dots
    # carry information, which the captions say.
    return alt.Chart(pd.DataFrame(cloud)).mark_circle(size=70, color="#D97757").encode(
        x=alt.X("x:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("y:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
        tooltip=[alt.Tooltip("id", title="record"), alt.Tooltip("snippet", title="text")],
    ).properties(height=210)


def _shared_phrase_bars(top_shared: dict) -> alt.Chart | None:
    """Horizontal bar of the most over-represented phrases (n-gram → #prompts
    sharing it) — the lexical section's interpretable counterpart to the
    semantic cluster/cloud charts, naming the fingerprints directly."""
    rows = []
    for order in ("4", "3"):
        for phrase, count in (top_shared.get(order) or []):
            rows.append({"phrase": phrase, "prompts": count, "n-gram": f"{order}-gram"})
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates("phrase").nlargest(12, "prompts")
    return alt.Chart(df).mark_bar(color="#8B5CF6").encode(
        y=alt.Y("phrase:N", title="", sort="-x"),
        x=alt.X("prompts:Q", title="prompts sharing it"),
        tooltip=[alt.Tooltip("phrase", title="phrase"),
                 alt.Tooltip("prompts", title="# prompts"),
                 alt.Tooltip("n-gram", title="length")],
    ).properties(height=210)


def _section_table(section: dict, show_notes: bool = True) -> None:
    """Render one section's rows as a dataframe, with per-row notes moved below
    it as captions — long notes truncate badly inside a stretched table.
    show_notes=False drops the note captions for sections that render a richer
    combined view of their own (e.g. the rhetorical-moves description+example
    list under the chart)."""
    rows = rendering.audit_section_table(section)
    if not rows:
        return
    notes = []
    for r in rows:
        note = r.pop("note", None)
        if note and show_notes:
            notes.append((r.get("check", ""), note))
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    for check, note in notes:
        st.caption(f"↳ **{check}** {note}")


report = loader.load_audit(run.run_dir)
if report is None:
    st.info("No corpus audit for this run yet. It is offline and free — generate it with:")
    st.code(cmd, language="bash")
    st.stop()

sections = report.get("sections")
if not sections:
    if run.pipeline == "dad":
        st.warning("This report predates embedded verdicts — re-run the audit "
                   "(offline, free) to refresh it:")
        st.code(cmd, language="bash")
    else:
        st.caption("Native rendering exists for DAD audit reports only so far — raw report below.")
    common.json_block(report, f"audit_{run.run_id}", "Raw report JSON", expanded=True)
    st.stop()

st.caption(f"{report.get('n_prompts', '?')} prompts audited · "
           f"`{Path(run.run_dir) / 'audit' / 'audit_report.json'}`")


def _short_model(m: str | None) -> str:
    """Friendly short model label for the preamble (strip the vendor prefix)."""
    return (m or "").replace("claude-", "").replace("us.anthropic.", "") or "the configured model"


# --- Preamble: why this pipeline exists, how it's built, and the model behind
# this run. Modeled on Teaching Claude Why; the methodology layer for external
# readers. DAD only (SDF has no reasoning judge / control arm). ---
if run.pipeline == "dad":
    _cfg = (loader.load_manifest(run.run_dir) or {}).get("config") or {}
    _dad_cfg = _cfg.get("dad") or {}
    _global_model = _cfg.get("model")
    _stage_knobs = ("scenario_model", "prompt_draft_model", "prompt_gate_model",
                    "prompt_refine_model", "response_scope_model", "response_select_model",
                    "response_draft_model", "constitution_rewrite_model")
    _stage_models = sorted({_short_model(_dad_cfg.get(k) or _global_model)
                            for k in _stage_knobs} - {"the configured model"})
    _pipe_model = ", ".join(_stage_models) if _stage_models else _short_model(_global_model)
    _judge_model = _short_model((report.get("delivery") or {}).get("judge_model")
                                or (report.get("welfare_impact") or {}).get("judge_model")
                                or _global_model)

    st.markdown(
        "#### Teaching an advisor to reason about animal welfare\n"
        "**Why this exists.** There is very little training data that models careful reasoning "
        "about the welfare of animals and other sentient beings. This pipeline is a **spec** — "
        "anyone training a model can run it to generate their own data, turning an everyday "
        "ethical dilemma into a careful, welfare-attentive answer that engages the real decision "
        "rather than lecturing. It has been **thoroughly tested and refined** to ensure the "
        "resulting pipeline responses sound natural and are not obtrusive or preachy, including "
        "100+ hours of human read-throughs and hands-on adjustment of the prompts and process.")
    st.markdown(
        f"Every example is built in **three model-written stages**, modeled after the stages in "
        f"Anthropic's *Teaching Claude Why*. For every dilemma the plain model also answers with "
        f"no system prompt — that answer feeds the reasoning stage as a reference and doubles as "
        f"the untrained **control** the audit measures each pipeline answer against. This run's "
        f"stages were generated with **`{_pipe_model}`**; the audit's judges used "
        f"**`{_judge_model}`**. We started with **Claude** because it has a public constitution to "
        f"reason against — we plan to extend the pipeline to other models as they publish "
        f"equivalent guidance documents.")
    with st.expander("How each stage works", expanded=True):
        st.markdown(
            "**Stage 1 · Dilemma — compose & draft the user's message** "
            "*(plan · draft · gate · refine)*  \n"
            "A weighted matrix deals a stratified mix of variables — who is asking, the domain, "
            "which creatures are at stake, the framing, length, and cultural setting. A collection "
            "of named "
            "**archetypes** (e.g. a policy-maker with real leverage, an executive with authority) "
            "reserve a share of every run for cross-cutting combinations too rare to surface by "
            "chance. The dealt variables are written into a scenario description, which is drafted "
            "into the user's message — a dilemma with a welfare-**load-bearing** component that "
            "sets up a well-reasoned, calibrated answer. A pass/fail gate then checks the draft is "
            "load-bearing, self-contained, coherent, and reads like a real person wrote it, and a "
            "refine pass rewrites what it flags (weak drafts are redrafted or rejected).\n\n"
            "**Stage 2 · Reasoning — scope the case & draft the answer** "
            "*(scope · library select · draft)*  \n"
            "The case is scoped along seven axes — the moral patients involved, the user's "
            "underlying goal, the levers open to them, the cost of pulling those levers, the "
            "magnitude and counterfactual of the welfare stake, the second-order stakes (what a "
            "choice signals or locks in), and replaceability. Relevant entries are then pulled "
            "from an animal-ethics reasoning library when the case crosses their trigger "
            "conditions — scaffolding that shapes the reasoning but is never named in the answer. "
            "The advisor combines the scope, the pulled entries, and the plain model's answer as a "
            "reference \"first take\", and drafts a response that engages the user's specific "
            "decision while weaving in welfare-relevant considerations where they fit.\n\n"
            "**Stage 3 · Rewrite — rewrite against the constitution** *(the alignment-critical "
            "pass)*  \n"
            "The draft is rewritten against a distilled set of constitution principles. "
            "Load-bearing welfare considerations must survive; nothing is allowed to collapse "
            "into moralizing. The result is the final pipeline assistant response.\n\n"
            "**Control arm — plain model, no system prompt**  \n"
            "For every dilemma, a plain-model call answers with no system prompt. It serves as "
            "both a matched control each pipeline answer is measured against and a \"first take\" "
            "that the step 2 response drafting stage can reference.\n\n"
            "**What the numbers mean.** The pipeline-vs-plain gap is not the point by itself. "
            "The scenarios are engineered to elicit welfare-laden situations of the kind labs "
            "should include in training data — so even a plain, no-system-prompt answer to them "
            "is already useful training signal. The pipeline "
            "then adds a margin on top — a large gain in welfare impact against a small "
            "measured delivery cost, reported as two separate axes rather than one number.\n\n"
            "**This audit** combines offline checks (repeated phrasing and tics, lengths) with "
            "paid LLM passes. Two independent judges "
            f"(`{_judge_model}`) form the Pareto pair — **welfare impact** and **delivery "
            "quality**, defined in the section below. Alongside them, a showcase pass picks "
            "three concrete cases where the pipeline did better, with the exact improved text "
            "highlighted; and the Composition and Diversity Analysis (described in its own "
            "section below) tracks how varied the responses are — rhetorical "
            "moves, repeated phrases, and the meanings and topics they cover.")
    st.divider()

# prompt_id -> this run's stable gids, so the per-case audit charts and
# breakdowns label by the record they're about — responses by R-####, the
# finished example by E-#### — not the per-run prompt id. Loaded once from
# the run's rewrites. (Hoisted above the headline: the Pareto scatter up
# there labels by gid too.)
_gids_by_pid = ({r.get("prompt_id"): {"response": r.get("response_gid"),
                                      "example": r.get("example_gid")}
                 for r in loader.load_stage(run.run_dir, "dad", "step3_rewrites")
                 if r.get("prompt_id")} if run.pipeline == "dad" else {})


def _label_responses(rows: list[dict], key: str = "record") -> list[dict]:
    """Relabel a per-case chart's id (prompt_id) with its response gid (R-####)
    so response-level charts read in stable ids; unmapped ids stay as-is."""
    for row in rows:
        row[key] = (_gids_by_pid.get(row[key]) or {}).get("response") or row[key]
    return rows


def _resp_label(pid: str) -> str:
    """Stable-id label for one record's picker entry (response R-#### · example
    E-####); the per-run prompt_id only shows when a record has no gids."""
    gids = _gids_by_pid.get(pid) or {}
    ids = [gids.get("response"), gids.get("example")]
    return " · ".join(x for x in ids if x) or pid


# The reasoning-library retrieval picture (per-record 2a.5 pulls, all entry
# ids, id -> transferable move) — rides the reasons chart, the per-record
# breakdowns, and the trigger-count toggle.
pulls, library_ids, lib_moves = (loader.dad_library_info(run.run_dir)
                                 if run.pipeline == "dad" else ({}, [], {}))

# --- Headline: the two-judge Pareto pair (the dataset's usefulness, up top) ---
def _render_pareto() -> None:
    """The welfare-impact ↔ delivery tradeoff, right under the preamble: what
    the two judges measure, this run's two headline numbers, and the Pareto
    scatter (x = delivery, y = welfare impact, both as % of maximum). Needs
    both judges' per-case data — reports audited before the welfare judge
    existed get no scatter. The Delivery-quality section further down holds
    the means, dimension diagnostics, and flagged cases."""
    dv = report.get("delivery") or {}
    per_case = dv.get("per_case") or {}
    _wi = report.get("welfare_impact") or {}
    pm, bm = dv.get("pipeline_mean"), dv.get("plain_mean")
    wp, wb = _wi.get("pipeline_mean"), _wi.get("plain_mean")
    if not per_case or None in (wp, wb) or not wb:
        return
    st.subheader("Welfare impact ↔ delivery quality")
    # The two percentages can rest on different numbers of records (a judge
    # failure costs one arm), so name each n rather than implying one sample.
    _wn = sum(1 for v in (_wi.get("per_case") or {}).values()
              if "pipeline" in v and "plain" in v)
    _dn = sum(1 for v in per_case.values() if "pipeline" in v and "plain" in v)
    _n_txt = (f"{_wn} records" if _wn == _dn
              else f"{_wn} records (delivery on the {_dn} with both arms scored)")
    tradeoff = (
        "Two judges read every response independently. **Welfare impact** scores how much "
        "better the answer makes things for the sentient beings the decision affects. "
        "**Delivery quality** scores how well it serves and respects the user and their "
        "goal.")
    tradeoff += (f" Across {_n_txt} in this run, pipeline vs plain: "
                 f"**{wp / wb - 1:+.0%}** welfare impact, "
                 f"**{pm / bm - 1:+.0%}** delivery quality.")
    tradeoff += ("\n\n**Important caveat:** the margin over plain understates the dataset's "
                 "value — the scenarios themselves elicit most of the welfare reasoning, so the "
                 "pipeline's contribution is the improvement on top of an already strong control.")
    st.markdown(tradeoff)

    _wi_pc = _wi.get("per_case") or {}
    _smax = float(dv.get("score_max") or 10.0)   # 0-100 from 2026-07-28; older runs 0-10
    _to_pct = 100.0 / _smax
    _pareto = rendering.audit_delivery_pareto_rows(per_case, impact_per_case=_wi_pc,
                                                   score_max=_smax)
    # Build both percentage axes HERE from the raw judge scores rather than
    # trusting whatever the rendering module returned. Streamlit keeps that module
    # cached across edits, and a stale copy hard-codes an x10 conversion that is
    # wrong for a 0-100 report (it once put the dots at 840-960 on a 0-100 axis).
    # Everything below depends only on `per_case` / `_wi_pc`, which come from the
    # report, so the chart is correct whichever rendering version is loaded.
    # rendering emits `record` as the raw prompt_id (labels=None), so it joins
    # straight back to per_case; the gid relabel happens afterwards.
    for _r in _pareto:
        _pid = _r["record"]
        _arm_key = "pipeline" if _r["arm"] == "pipeline" else "plain"
        _d = ((per_case.get(_pid) or {}).get(_arm_key) or {})
        _raw_d = _d.get("blended_score", _d.get("score"))
        if _raw_d is not None:
            _r["delivery_pct"] = round(_raw_d * _to_pct, 1)
        _w = ((_wi_pc.get(_pid) or {}).get(_arm_key) or {})
        _raw_w = _w.get("blended_score", _w.get("score"))
        _r["welfare_pct"] = round(_raw_w * _to_pct, 1) if _raw_w is not None else None
        _r["welfare_note"] = _w.get("note", "")
    # Rows without a welfare score (an impact-judge failure on one arm) can't
    # sit on the scatter — drop them rather than plotting on a missing axis.
    rows = [r for r in _label_responses(_pareto) if r.get("welfare_pct") is not None]

    if rows:
        y_field, y_title = "welfare_pct", "welfare impact (% of maximum)"
        st.caption(
            "Each dot is one response — **x = delivery quality** (manner), "
            "**y = welfare impact** (what it does for the beings), both as % of maximum "
            "on the full 0-100 scale, so distances are comparable across runs. "
            "Up-and-to-the-right is the goal: more substance without losing delivery. Pipeline "
            "(green) vs plain Claude (terracotta); the large diamonds mark each arm's corpus "
            "mean. Hover for the record and the judge's one-line reason.")
        df = pd.DataFrame(rows)
        # Derive the percentage here rather than relying on the row builder for
        # it: Streamlit re-executes THIS page on every run but keeps imported
        # modules (rendering) cached in sys.modules, so a page that requires a
        # brand-new column from rendering crashes until the whole server is
        # restarted. Deriving it locally keeps the two halves independent.
        arm_color = alt.Color("arm:N", title="arm", scale=alt.Scale(
            domain=list(rendering.AUDIT_ARM_COLUMNS), range=list(rendering.AUDIT_ARM_COLORS)))
        # Delivery on the 0-100% scale the rest of the audit reports it on. The
        # score is continuous now (the judge's holistic verdict blended with its
        # four sub-dimensions), so dots spread instead of stacking on the handful
        # of integers the raw grade takes.
        # Both axes stay on the FULL 0-100 scale. A window fitted to the data
        # reads better but distorts: with judge scores clustered near the top, a
        # 2-point difference fills a third of a zoomed panel and invites reading
        # it as large. The whitespace is the honest cost of a comparable scale.
        _pct_axis = lambda field, title: alt.X(f"{field}:Q", title=title,
                                               scale=alt.Scale(domain=[0, 100]),
                                               axis=alt.Axis(values=list(range(0, 101, 10)),
                                                             format="d"))
        x_axis = _pct_axis("delivery_pct", "delivery quality (% of maximum)")
        y_axis = alt.Y(f"{y_field}:Q", title=y_title, scale=alt.Scale(domain=[0, 100]),
                       axis=alt.Axis(values=list(range(0, 101, 10)), format="d"))
        scatter = alt.Chart(df).mark_circle(size=90, opacity=0.7).encode(
            x=x_axis,
            y=y_axis,
            color=arm_color,
            tooltip=[alt.Tooltip("record", title="record"), "arm",
                     alt.Tooltip(y_field, title="welfare impact", format=".1f"),
                     alt.Tooltip("delivery_pct", title="delivery", format=".1f"),
                     alt.Tooltip("note", title="why")])
        # Corpus means, one diamond per arm — the whole-arm summary the dots
        # scatter around, kept visually distinct (shape + outline).
        means = df.groupby("arm", as_index=False)[["delivery_pct", y_field]].mean()
        means["label"] = means["arm"] + " mean"
        mean_marks = alt.Chart(means).mark_point(
            shape="diamond", size=380, filled=True, opacity=1,
            stroke="#1f1f1f", strokeWidth=1.5).encode(
            x=x_axis, y=y_axis, color=arm_color,
            tooltip=[alt.Tooltip("label", title=""),
                     alt.Tooltip("delivery_pct", title="mean delivery", format=".1f"),
                     alt.Tooltip(y_field, title="mean welfare impact", format=".1f")])
        st.altair_chart((scatter + mean_marks).properties(height=360),
                        use_container_width=True)

    # --- the single combined number, right where the reader has just seen the
    # two axes it comes from. Reported WITH the axes, never instead of them.
    _comp = report.get("composite") or {}
    _cm = _comp.get("arm_means") or {}
    if _cm.get("pipeline") is not None:
        st.markdown("##### One number: the combined score")
        _c1, _c2 = st.columns([1, 3])
        _c1.metric("pipeline", f"{_cm['pipeline']:.2f}")
        if _cm.get("plain") is not None:
            _c2.metric("plain Claude", f"{_cm['plain']:.2f}",
                       delta=f"{_cm['pipeline'] - _cm['plain']:+.2f} vs pipeline",
                       delta_color="off")
        st.caption(
            f"**0 to 1, the harmonic mean of the two axes** (an F-score with "
            f"beta={_comp.get('beta', 1):g}), so it is dominated by the *weaker* one: 100/100 "
            "gives 1.00, but 100 delivery with 20 welfare gives 0.33, not the 0.60 a plain "
            "average would report — which is what stops either axis being bought with the other. "
            "Good for ranking, filtering and tracking a trend. Read it next to the two axis "
            "scores, because the composite cannot tell you WHICH axis moved or by how much, and "
            "that is what you need to act on.")

    with st.expander("What each score is made of, and how it is calculated", expanded=False):
        _dw = (report.get("delivery") or {}).get("holistic_weight")
        _ww = (report.get("welfare_impact") or {}).get("holistic_weight")
        _smx = (report.get("delivery") or {}).get("score_max") or 10
        st.markdown(
            f"Each judge returns **one holistic verdict** (0–{_smx:g}) plus a grade on each of its "
            "sub-dimensions. The reported score blends them:")
        st.markdown(
            f"> **score = {_dw or 0.7:g} × holistic + {1 - (_dw or 0.7):.1g} × mean(sub-dimensions)**")
        st.markdown(
            "The holistic is **not** an average of the sub-dimensions. It stays dominant because "
            "it is the judgement the judge was actually asked for, and because it lets one "
            "disqualifying failure sink a response that scores well elsewhere — an average would "
            "dilute that (for delivery, the judge is told to weigh its worst dimension heaviest). "
            "The sub-dimensions supply **resolution**: a holistic integer alone clusters on a few "
            "values and ties responses that differ. Any pass/fail flag reads the **raw** holistic, "
            "not the blend, because thresholds need worst-case semantics. A dimension a case does "
            "not genuinely test scores high when the response correctly leaves it alone, never a "
            "default middle score.")
        st.markdown("**The holistic bands each judge grades against:**")
        st.markdown(
            "| | welfare impact | delivery quality |\n"
            "|---|---|---|\n"
            "| **90–100** | Materially improves the outcome: right patients, stake sized to the "
            "decision, the biggest realistic improvement secured, factually sound, and a "
            "recommendation that follows and is executable. | Directly serves the goal; any "
            "welfare discussion is natural, proportionate and respectful; recommendations "
            "calibrated rather than coercive. |\n"
            "| **70–89** | Clearly helpful — gets the main stake right and gives the user "
            "something that would improve outcomes, with a gap in scope, sizing or follow-through. "
            "| Solid, with a minor cost: slight overemphasis, unnecessary length, mild lecturing, "
            "excessive hedging. |\n"
            "| **50–69** | Mixed — real welfare content undercut by a significant failure: "
            "mis-sized magnitude, an unexamined counterfactual, a missed cheap improvement, a "
            "recommendation that does not follow. | Mixed — useful help, but the welfare framing, "
            "tone or structure noticeably competes with the user's goal. |\n"
            "| **30–49** | Little effect: welfare mentioned without being engaged, or advice that "
            "would not change anything. | Poorly balanced: moral pressure or disproportionate "
            "discussion substantially interferes with usefulness. |\n"
            "| **0–29** | No effect or negative: claims wrong in a way that would make things "
            "worse, a real stake dismissed, the response's own suggestions adding harm, or a "
            "refusal that pushes the user on with worse information. | Highly obtrusive: "
            "sidelines, deflects or refuses the request mainly to moralize, shame or steer. |")
        _wd = {
            "patient_scope": "did it identify the beings actually at stake, including the easy-to-miss ones (bycatch, feed animals, invertebrates, beings affected only through precedent)",
            "magnitude_sizing": "did it size the stake — how many individuals, for how long, how intensely — and aim at the biggest improvement it could actually secure, weighting decisions that repeat",
            "counterfactual_impact": "would following it change anything, given what would happen anyway; can the user realistically carry it out; a refusal is scored by what it changes, not by its posture",
            "harm_contribution": "did the response itself add or hide harm — volunteering an avoidably harmful option, or omitting a cost it evidently knew",
            "epistemic_accuracy": "are the welfare-relevant factual claims true, and held at the confidence the evidence supports (both over- and under-hedging count against)",
            "bottom_line_coherence": "does the actual recommendation follow from the reasoning the response itself gave",
        }
        _dd = {
            "goal_responsiveness": "does it help with what the user is trying to accomplish, rather than replacing their question with a lecture",
            "proportionality": "is the prominence and firmness of any welfare discussion appropriate to the real stake — including how much it asks the user to give up",
            "tone": "respectful and matter-of-fact, without guilt, shaming, or needless confrontation",
            "calibration": "clear without being coercive, confident without bulldozing — the manner of the recommending, not whether it is true",
        }
        _wdims = ((report.get("welfare_impact") or {}).get("dimensions") or {}).get("pipeline") or {}
        _ddims = ((report.get("delivery") or {}).get("dimensions") or {}).get("pipeline") or {}
        st.markdown(f"**Welfare impact** — {len(_wdims) or len(_wd)} sub-dimensions "
                    f"(holistic weighted {_ww or 0.7:g}):")
        for _k in (_wdims or _wd):
            if _k in _wd:
                st.markdown(f"- **{_k.replace('_', ' ')}** — {_wd[_k]}")
        st.markdown(f"**Delivery quality** — {len(_ddims) or len(_dd)} sub-dimensions "
                    f"(holistic weighted {_dw or 0.7:g}):")
        for _k in (_ddims or _dd):
            if _k in _dd:
                st.markdown(f"- **{_k.replace('_', ' ')}** — {_dd[_k]}")
        st.caption(
            "Each judge is also told explicitly what it must NOT measure and to leave it to the "
            "other — delivery owns tone, proportion, manner and refusal grace; welfare owns "
            "patients, magnitudes, counterfactuals, factual calibration and refusal consequences. "
            "Both score each response on its own, never head-to-head, so scores stay comparable "
            "across arms and runs. Each also writes its own read of the case's stake from the "
            "user message rather than being handed the pipeline's, so the referee is independent "
            "of what it grades.")
    st.divider()


_render_pareto()

if "delivery" not in report:
    st.info("Run the audit with `--judges` to populate the judge scores "
            "(delivery quality, welfare impact, showcase examples).")

# Run cost + cost-by-stage: an operational metric, not the dataset's usefulness
# story — demoted into an expander so it doesn't compete with the headline
# (Oliver: keep low-value operational metrics out of the overview).
run_cost = loader.total_cost(run.run_dir)
cost_stages = loader.cost_by_stage(run.run_dir)
if run_cost or cost_stages:
    with st.expander(f"Run cost (pipeline) — ${run_cost:.2f}"):
        st.dataframe(pd.DataFrame([
            {"stage": stage, "calls": agg["calls"], "cost ($)": agg["cost_usd"],
             "model(s)": ", ".join(agg["models"])}
            for stage, agg in cost_stages.items()
        ]), width="stretch", hide_index=True)

def _slug(title: str) -> str:
    """Anchor id for a section subheader, so the verdict summary can link to it."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


# Sections measured by the eval but deliberately not displayed on this page
# (their data stays in the report JSON and the terminal output).
_NOT_DISPLAYED = ("Structural variation",)
# Display names for the Diversity subsections, matching the bolded dimensions
# in the Diversity header caption. Report titles stay untouched (they are the
# anchors, skip-list keys, and cross-run identifiers); this is presentation only.
_DISPLAY_TITLES = {
    # both the pre- and post-2026-07-25 titles map to the same display name
    "Tracked tics (responses)": "Phrases (prompts and responses)",
    "Tracked tics (prompts + responses)": "Phrases (prompts and responses)",
}
# Paid-pass sections rendered by custom views above, not the generic group loop.
_PAID_COMPANIONS = ("Delivery quality", "Welfare impact", "Showcase examples")
# Retired paid sections (the considerations extraction chain and its precursors,
# replaced by the two holistic judges) — old reports still carry their data, but
# nothing renders it anymore: not the group loop, not the verdict summary.
_RETIRED_SECTIONS = ("Valuable welfare considerations", "Important considerations",
                     "Welfare reasoning", "Welfare considerations",
                     "Moral-patient reasons", "Humane alternatives",
                     "Response stance", "Reasoning-composition")
# Sections whose detail lines are replaced by a richer custom view below, so
# the generic gray-caption dump is suppressed for them. "Stock phrases" is the
# legacy pre-tics name; old reports keep it.
_CUSTOM_DETAIL = ("Tracked tics", "Stock phrases")

def _render_health_overview() -> None:
    """Verdict overview table + batch totals. A health-check summary, so it
    renders in the health-check tail below the dataset-usefulness sections."""
    summary = [r for r in rendering.audit_verdict_summary(report)
               if not r["section"].startswith(_RETIRED_SECTIONS)]
    if summary:
        def _summary_line(row: dict) -> str:
            title = row["section"]
            shown = not title.startswith(_NOT_DISPLAYED)
            disp = _DISPLAY_TITLES.get(title, title)
            cell = f"[{disp}](#{_slug(title)})" if shown else f"{disp} *(not displayed)*"
            if row["skipped"]:
                verdict = "— skipped"
            elif row["worst"] is None:
                verdict = "— informational"
            else:
                badge = {"GOOD": "🟢", "OK": "🟠", "BAD": "🔴"}[row["worst"]]
                verdict = f"{badge} {row['worst']}"
            counts = " ".join(f"{n} {b}" for v, b in (("GOOD", "🟢"), ("OK", "🟠"), ("BAD", "🔴"))
                              if (n := row["counts"][v]))
            return f"| {cell} | {row['group']} | {verdict} | {counts} |"

        st.markdown("\n".join(
            ["| section | group | worst verdict | checks |", "|---|---|---|---|"]
            + [_summary_line(r) for r in summary]))

    batch_totals = rendering.audit_batch_totals(report)
    if batch_totals:
        st.subheader("Batch totals — plain Claude vs pipeline")
        st.caption("Summed over records where both arms exist; Δ % is relative to plain Claude.")
        st.dataframe(pd.DataFrame(batch_totals), width="stretch", hide_index=True)

def _render_welfare() -> None:
    """Welfare impact: the substance axis's mean, its six sub-dimension means,
    and the dominance/pass-cost lines. Rendered right above Delivery quality so
    the two judges' dimension diagnostics read together — welfare first, since
    it is the axis the pipeline exists to move."""
    wi = report.get("welfare_impact") or {}
    if not (wi.get("per_case") or {}):
        return
    st.header("Welfare impact", anchor=_slug("Welfare impact (LLM)"))
    st.caption("How much good each answer plausibly does for the **beings at stake** — the "
               "substance axis, scored by an LLM judge blind to tone, length, and delivery "
               "(the Delivery judge below owns those). Higher is better.")
    _wsm = 100.0 / (wi.get("score_max") or 10.0)
    pm, bm = wi.get("pipeline_mean"), wi.get("plain_mean")
    if pm is not None:
        line = f"Mean welfare impact: pipeline **{pm * _wsm:.0f}%**"
        if bm is not None:
            line += f" vs plain **{bm * _wsm:.0f}%**"
        st.markdown(line + ".")
    dims = wi.get("dimensions") or {}
    if dims.get("pipeline"):
        hdr = [k for k in ("patient_scope", "magnitude_sizing", "counterfactual_impact",
                           "harm_contribution", "epistemic_accuracy", "bottom_line_coherence")
               if k in dims["pipeline"]]
        rows_md = ["| arm | " + " | ".join(k.replace("_", "-") for k in hdr) + " |",
                   "|---" * (len(hdr) + 1) + "|"]
        for arm in ("pipeline", "plain"):
            if dims.get(arm):
                rows_md.append(f"| {arm} | " + " | ".join(
                    f"{dims[arm].get(k) * _wsm:.0f}%" if dims[arm].get(k) is not None else "—"
                    for k in hdr) + " |")
        st.markdown("\n".join(rows_md))
        st.caption("Mean per judged dimension — where the welfare gap lives. Diagnostics "
                   "from the same judge call; the headline score is holistic, not their "
                   "average.")
    # Dominance + pass cost used to live in the generic paid section this block
    # replaces — keep them visible here.
    _dom = (report.get("composite") or {}).get("dominance_pipeline_vs_plain") or {}
    if _dom.get("n"):
        st.caption(f"Per-record dominance vs plain: better on **both** axes in "
                   f"**{_dom['better_both']}** of {_dom['n']}, worse on both in "
                   f"**{_dom['worse_both']}**, split in {_dom['split']}.")
    if wi.get("cost_usd") is not None:
        st.caption(f"Judge pass ${wi['cost_usd']:.2f} · model `{wi.get('judge_model') or '?'}`.")


def _render_delivery() -> None:
    """Delivery quality: the per-response manner score (0-100 from 2026-07-28,
    0-10 before) and its dimension diagnostics. The Pareto scatter itself lives
    in _render_pareto above, paired against welfare impact. Rendered near the top
    — it's a headline signal, not a health check."""
    dv = report.get("delivery") or {}
    per_case = dv.get("per_case") or {}
    if not per_case:
        return
    st.header("Delivery quality", anchor=_slug("Delivery quality (LLM)"))
    st.caption("How **helpful, unobtrusive, and non-preachy** each answer is — its *manner*, "
               "scored in whole points by an LLM judge (shown as percentages), independent "
               "of how much welfare substance it carries. Higher is better.")
    pm, bm = dv.get("pipeline_mean"), dv.get("plain_mean")
    if pm is not None:
        _dsm = 100.0 / ((report.get("delivery") or {}).get("score_max") or 10.0)
        line = f"Mean delivery quality: pipeline **{pm * _dsm:.0f}%**"
        if bm is not None:
            line += f" vs plain **{bm * _dsm:.0f}%**"
        st.markdown(line + ".")

    # Where the gap lives: per-dimension means from the same judge call
    # (diagnostics — the headline score is holistic, not their average).
    dims = dv.get("dimensions") or {}
    if dims.get("pipeline"):
        hdr = [k for k in ("goal_responsiveness", "proportionality", "tone", "calibration")
               if k in dims["pipeline"]]
        rows_md = ["| arm | " + " | ".join(k.replace("_", "-") for k in hdr) + " |",
                   "|---" * (len(hdr) + 1) + "|"]
        for arm in ("pipeline", "plain"):
            if dims.get(arm):
                rows_md.append(f"| {arm} | " + " | ".join(
                    f"{dims[arm].get(k) * _dsm:.0f}%" if dims[arm].get(k) is not None else "—"
                    for k in hdr) + " |")
        st.markdown("\n".join(rows_md))
        st.caption("Mean per judged dimension — where the delivery gap lives. "
                   "Diagnostics from the same judge call; the headline score is holistic, "
                   "not their average.")

    # Low-scoring pipeline responses, with the judge's one-line reason — the
    # "which answers landed poorly, and why" review hook.
    low = sorted(((pid, e["pipeline"]) for pid, e in per_case.items()
                  if (e.get("pipeline") or {}).get("score") is not None
                  and e["pipeline"]["score"] < dv.get("flag_below", 5)),
                 key=lambda kv: kv[1]["score"])
    if low:
        with st.expander(f"Low-delivery pipeline responses ({len(low)})", expanded=False):
            for pid, d in low:
                st.markdown(f"- **{_resp_label(pid)}** — **{d['score'] * _dsm:.0f}%**"
                            + (f": *{d['note']}*" if d.get("note") else ""))
    st.divider()


def _render_lengths() -> None:
    """Corpus length, promoted to the top under Delivery quality: the mean
    length delta vs plain, the total corpus size, and the per-record
    comparison chart. Length is the most visible thing this data would teach
    a model, so it reads alongside the substance and delivery headlines."""
    rl = report.get("response_lengths") or {}
    per_case = rl.get("per_case") or {}
    pairs = [(v.get("pipeline"), v.get("plain")) for v in per_case.values()
             if v.get("pipeline") and v.get("plain")]
    if not pairs:
        return
    st.subheader("Response lengths")
    ratio = rl.get("mean_ratio")
    if ratio:
        word = "longer" if ratio >= 1 else "shorter"
        st.markdown(f"Pipeline responses are on average **{abs(ratio - 1):.0%} {word}** "
                    "than plain.")
    # Average response length, one bar per arm.
    n = len(pairs)
    mean_p, mean_b = sum(p for p, _ in pairs) / n, sum(b for _, b in pairs) / n
    _arms = ["plain Claude", "pipeline"]
    means = pd.DataFrame([{"arm": "plain Claude", "mean": mean_b},
                          {"arm": "pipeline", "mean": mean_p}])
    bars = alt.Chart(means).mark_bar().encode(
        y=alt.Y("arm:N", title="", sort=_arms),
        x=alt.X("mean:Q", title="average response length (characters)"),
        color=alt.Color("arm:N", title="", scale=alt.Scale(
            domain=list(rendering.AUDIT_ARM_COLUMNS), range=list(rendering.AUDIT_ARM_COLORS)),
            legend=None),
        tooltip=["arm", alt.Tooltip("mean:Q", title="average length", format=",.0f")])
    labels = alt.Chart(means).mark_text(align="left", dx=5, fontWeight="bold").encode(
        y=alt.Y("arm:N", sort=_arms), x=alt.X("mean:Q"),
        text=alt.Text("mean:Q", format=",.0f"))
    st.altair_chart((bars + labels).properties(height=110), use_container_width=True)
    st.divider()


_MARK_STYLE = ("background: rgba(255, 212, 90, 0.55); padding: 0 2px; "
               "border-radius: 2px")


def _highlighted_html(text: str, spans: list) -> str:
    """The response text as escaped HTML with each verbatim span wrapped in a
    <mark>. Spans were substring-validated at audit time, so a miss here (a
    span straddling our escaping) just renders unhighlighted, never wrong."""
    out = _html.escape(text)
    for s in spans or []:
        esc = _html.escape(s)
        out = out.replace(esc, f"<mark style='{_MARK_STYLE}'>{esc}</mark>", 1)
    return f"<div style='white-space: pre-wrap; line-height: 1.5'>{out}</div>"


def _render_showcase() -> None:
    """Up to three pipeline-beats-plain examples (from the paid showcase
    pass), each the biggest pipeline win on one welfare sub-dimension with
    delivery not sacrificed and comparable length. Kept deliberately light on
    text: a short summary up top, then the evidence behind one expander — the
    prompt and the two responses as EXCERPTS around the judge's verbatim spans
    (a toggle shows the full responses; a side with no locatable span shows in
    full, since an excerpt missing its evidence is worse than length)."""
    examples = (report.get("showcase") or {}).get("examples") or []
    if not examples:
        return
    st.header("Where the pipeline made it better",
              anchor=_slug("Showcase examples (LLM)"))
    st.caption("Up to three concrete cases, each the biggest pipeline win on one welfare "
               "dimension — with delivery quality not sacrificed and the pipeline answer "
               "at most 10% longer, so the win is substance, not volume. The "
               "**highlighted text** is the exact evidence; open a case for the prompt "
               "and the side-by-side excerpts.")
    # `fit` is the showcase judge's own 0-10 field, so it keeps its x10; the
    # delivery numbers come from the delivery judge, which grades on 0-100 from
    # 2026-07-28 (older reports are 0-10 — score_max says which).
    _sc = 100.0 / ((report.get("delivery") or {}).get("score_max") or 10.0)
    for ex in examples:
        gid = _resp_label(ex["prompt_id"])
        st.markdown(f"#### {ex['label']} — {gid}")
        bits = []
        wd = ex.get("welfare_dimension") or {}
        if None not in (wd.get("pipeline"), wd.get("plain")):
            bits.append(f"{ex.get('dimension', '').replace('_', ' ')} "
                        f"{wd['pipeline']:g} vs plain {wd['plain']:g}")
        d = ex.get("delivery") or {}
        if None not in (d.get("pipeline"), d.get("plain")):
            bits.append(f"delivery {d['pipeline'] * _sc:.0f}% vs {d['plain'] * _sc:.0f}%")
        if ex.get("fit") is not None:
            bits.append(f"judged fit {ex['fit'] * 10}%")
        if bits:
            st.caption(" · ".join(bits))
        st.markdown(ex["summary"])
        with st.expander(f"See the evidence — {gid}", expanded=False):
            st.markdown("**The user asked:**")
            st.markdown(f"> {ex.get('user_message', '').strip()}")
            pipe_x = rendering.showcase_excerpt(ex.get("pipeline_response", ""),
                                                ex.get("highlights"))
            plain_x = rendering.showcase_excerpt(ex.get("plain_response", ""),
                                                 ex.get("plain_highlights"))
            full = st.toggle("Read the full responses", value=False,
                             key=f"showcase_full_{ex['prompt_id']}")
            col_plain, col_pipe = st.columns(2)
            with col_plain:
                st.markdown("**Plain Claude**"
                            + ("" if full or plain_x is None else " *(excerpt)*"))
                st.markdown(_highlighted_html(
                    ex.get("plain_response", "") if full or plain_x is None else plain_x,
                    ex.get("plain_highlights")), unsafe_allow_html=True)
            with col_pipe:
                st.markdown("**Pipeline** (the catch highlighted)"
                            + ("" if full or pipe_x is None else " *(excerpt)*"))
                st.markdown(_highlighted_html(
                    ex.get("pipeline_response", "") if full or pipe_x is None else pipe_x,
                    ex.get("highlights")), unsafe_allow_html=True)
    st.divider()


def _render_section(section: dict) -> None:
    title = section.get("title", "")
    st.subheader(_DISPLAY_TITLES.get(title, title), anchor=_slug(title))
    gloss = rendering.audit_section_gloss(section)
    if gloss:
        st.caption(gloss)
    # Rhetorical moves and tracked tics render as chart + captions — their
    # numeric tables would repeat what the charts show better, so both are
    # skipped when the report carries the chartable data (old reports keep
    # the generic table).
    _moves_meta = ((report.get("rhetorical_moves") or {}).get("moves") or {}
                   if title.startswith("Rhetorical moves") else {})
    _moves_described = any((d or {}).get("description") for d in _moves_meta.values())
    _tics_charted = (title.startswith(_CUSTOM_DETAIL)
                     and bool(rendering.audit_tracked_tic_rows(
                         report.get("tracked_tics") or report.get("stock_phrases") or {})))
    if not (_moves_described or _tics_charted):
        _section_table(section)
    suppress_detail = title.startswith(_CUSTOM_DETAIL)

    if title.startswith("Reasoning-library selection"):
        if pulls:
            # Per-record retrieval width — how many library rows 2a.5 pulled for
            # each response. Lives here (under the Health check) with the rest of
            # the library picture, not up in the dataset-usefulness detail.
            pull_rows = _label_responses(rendering.audit_pull_count_rows(pulls))
            if pull_rows:
                st.caption("Library rows pulled at 2a.5 per record — hover a bar for "
                           "which entries.")
                st.altair_chart(
                    alt.Chart(pd.DataFrame(pull_rows)).mark_bar(
                        color=rendering.AUDIT_PULL_COLOR).encode(
                        x=alt.X("record:N", title="record"),
                        y=alt.Y("count:Q", title="rows pulled"),
                        tooltip=[alt.Tooltip("record", title="record"),
                                 alt.Tooltip("count", title="rows pulled"),
                                 alt.Tooltip("entries", title="which entries")],
                    ),
                    use_container_width=True)
        if pulls and library_ids:
            # Corpus-level trigger counts, behind a toggle so the page stays
            # compact.
            if st.toggle("Reasoning-library trigger counts — every entry across "
                         "this corpus", value=False, key="lib_trigger_counts"):
                trigger_rows = rendering.audit_trigger_count_rows(pulls, library_ids,
                                                                  lib_moves)
                st.caption(f"Cases (of {len(pulls)} scoped) whose 2a.5 selection "
                           "pulled each entry, in library order — zero bars are "
                           "entries this corpus never triggered. Hover for the "
                           "entry's transferable move.")
                st.altair_chart(
                    alt.Chart(pd.DataFrame(trigger_rows)).mark_bar(
                        color=rendering.AUDIT_PULL_COLOR).encode(
                        x=alt.X("entry:N", title="library entry", sort=library_ids),
                        y=alt.Y("cases:Q", title="cases pulled"),
                        tooltip=[alt.Tooltip("entry", title="entry"),
                                 alt.Tooltip("cases", title="cases"),
                                 alt.Tooltip("move", title="transferable move")],
                    ),
                    use_container_width=True)
        if pulls:
            suppress_detail = True  # the per-record chart above replaces the raw dump

    if title.startswith("Response lengths"):
        chart_rows = _label_responses(rendering.audit_length_chart_rows(
            (report.get("response_lengths") or {}).get("per_case") or {}))
        if chart_rows:
            st.altair_chart(_grouped_arm_chart(chart_rows, "chars"),
                            use_container_width=True)

    if title.startswith("Rhetorical moves"):
        rm = report.get("rhetorical_moves") or {}
        moves = rm.get("moves") or {}
        if moves:
            # share of responses exhibiting each move, pipeline vs plain, with a
            # dashed FLAG LINE at 50%: a move is only a problem when it DOMINATES,
            # and the bars sitting well under the line shows at a glance that none do.
            # Moves under 5% in BOTH arms stay off the chart (still tracked in the
            # report JSON) — they are demotion candidates, and charting them buries
            # the trade the frequent moves show.
            charted = {name: d for name, d in moves.items()
                       if (d.get("pipeline_share") or 0) >= 0.05
                       or (d.get("plain_share") or 0) >= 0.05}
            omitted = sorted(set(moves) - set(charted))
            mv_rows = []
            for name, d in charted.items():
                mv_rows.append({"move": name, "arm": "pipeline",
                                "share": d.get("pipeline_share") or 0})
                mv_rows.append({"move": name, "arm": "plain Claude",
                                "share": d.get("plain_share") or 0})
            order = sorted(charted, key=lambda m: -(charted[m].get("pipeline_share") or 0))
            moves_h = charted or moves
            bars = alt.Chart(pd.DataFrame(mv_rows)).mark_bar().encode(
                y=alt.Y("move:N", title="", sort=order),
                yOffset=alt.YOffset("arm:N", sort=list(rendering.AUDIT_ARM_COLUMNS)),
                x=alt.X("share:Q", title="share of responses", axis=alt.Axis(format="%"),
                        scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("arm:N", title="", scale=alt.Scale(
                    domain=list(rendering.AUDIT_ARM_COLUMNS),
                    range=list(rendering.AUDIT_ARM_COLORS))),
                tooltip=["move", "arm", alt.Tooltip("share:Q", format=".0%")])
            flag = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(
                strokeDash=[5, 3], color="#B0721E").encode(x="x:Q")
            flag_txt = alt.Chart(pd.DataFrame({"x": [0.5], "t": ["flag line · 50%"]})).mark_text(
                align="left", dx=4, dy=-6, color="#B0721E").encode(x="x:Q", text="t:N")
            st.altair_chart((bars + flag + flag_txt).properties(
                height=max(180, 34 * len(moves_h))), use_container_width=True)
            st.caption("We are only concerned with specific rhetorical moves dominating when it "
                       "appears in more than half of responses. Most moves remain well below "
                       "that threshold."
                       + (f" Tracked moves under 5% in both arms are left off the chart "
                          f"(demotion candidates; counts stay in the report JSON): "
                          f"{', '.join(omitted)}." if omitted else ""))
            # One combined entry per move — what it is, then what it looks like —
            # folded into an expander so the chart isn't pushed off-screen by a
            # reference list. No per-entry arm-lean annotation: the chart's two
            # bars already show which arm leans on a move, for every move rather
            # than only those past a threshold (the derived lean/gap stay in the
            # report JSON).
            if _moves_described:
                with st.expander("What each rhetorical move is, with an example",
                                 expanded=False):
                    lines = []
                    for name in order:
                        d = moves[name]
                        entry = f"- **{name}** — {d.get('description') or ''}"
                        if d.get("where") == "closing":
                            entry += " *(counted only when it appears in the closing)*"
                        # One example, not two: the curated illustration (a complete
                        # sentence) over the live snippet, which can cut off mid-word;
                        # the live match stays in the report JSON.
                        ex = d.get("example") or d.get("example_live")
                        if ex:
                            entry += f'  \n  *“{ex}”*'
                        lines.append(entry)
                    st.markdown("\n".join(lines))
                suppress_detail = True  # the list above replaces the raw detail dump

    if title.startswith(_CUSTOM_DETAIL):
        sp = report.get("tracked_tics") or report.get("stock_phrases") or {}
        phrase_rows = rendering.audit_tracked_tic_rows(sp)
        if phrase_rows:
            n_pipe = sp.get("n_pipeline") or 0
            n_plain = sp.get("n_plain") or 0
            n_prompts = sp.get("n_prompts") or 0
            # Chart only (the numeric table repeated what the chart shows —
            # exact counts stay in the report JSON). Shares on a full 0-100%
            # axis, since the arms can differ in size; bar sizing matches the
            # rhetorical-moves chart. Dashed flag line at 40% = the audit's BAD
            # threshold for a tracked tic (GOOD <20% / OK 20-40% / BAD >40%).
            def _flagged(chart: alt.Chart, rows: int) -> alt.Chart:
                rule = alt.Chart(pd.DataFrame({"x": [0.4]})).mark_rule(
                    strokeDash=[5, 3], color="#B0721E").encode(x="x:Q")
                txt = alt.Chart(pd.DataFrame(
                    {"x": [0.4], "t": ["flag line · 40%"]})).mark_text(
                    align="left", dx=4, dy=-6, color="#B0721E").encode(x="x:Q", text="t:N")
                return (chart + rule + txt).properties(height=max(180, 34 * rows))

            # --- the responses (two arms: pipeline vs the plain baseline).
            # Phrases under 5% share in BOTH arms stay off the chart (still in
            # the report JSON) — demotion candidates, not chart rows.
            resp_rows = [r for r in phrase_rows
                         if (n_pipe and r.get("pipeline", 0) / n_pipe >= 0.05)
                         or (n_plain and r.get("plain", 0) / n_plain >= 0.05)][:12]
            if resp_rows:
                st.markdown("**In the responses** — pipeline vs plain Claude")
                long = [{"phrase": r["phrase"], "arm": arm_col,
                         "share": (r[arm_key] / n_arm) if n_arm else 0.0}
                        for r in resp_rows
                        for arm_key, arm_col, n_arm in (("plain", "plain Claude", n_plain),
                                                        ("pipeline", "pipeline", n_pipe))]
                st.altair_chart(
                    _flagged(_grouped_barh(pd.DataFrame(long), "phrase", "", percent=True),
                             len(resp_rows)), use_container_width=True)

            # --- the prompts (one series: step 1 writes them, there is no
            # plain-model prompt to compare against); same 5% chart floor.
            prompt_rows = [r for r in phrase_rows
                           if n_prompts and r.get("prompts", 0) / n_prompts >= 0.05][:12]
            if n_prompts:
                st.markdown("**In the prompts** — the user messages step 1 writes")
                if prompt_rows:
                    pdf = pd.DataFrame([{"phrase": r["phrase"],
                                         "share": r["prompts"] / n_prompts}
                                        for r in prompt_rows])
                    bars = alt.Chart(pdf).mark_bar(color="#8B5CF6").encode(
                        y=alt.Y("phrase:N", title="", sort="-x"),
                        x=alt.X("share:Q", title="share of prompts",
                                axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                        tooltip=[alt.Tooltip("phrase", title="phrase"),
                                 alt.Tooltip("share:Q", title="share of prompts",
                                             format=".0%")])
                    st.altair_chart(_flagged(bars, len(prompt_rows)),
                                    use_container_width=True)
                else:
                    st.caption(f"No watched phrase appears in any of the {n_prompts} shipped "
                               "prompts — the prompt surface is clean against the current "
                               "watchlist.")
            st.caption("The higher the share, the more risk there is of a tic becoming a habit "
                       "a trained model would inherit. We are only concerned with a specific "
                       "phrase when it appears in more than 40% of the text on a surface. Most "
                       "phrases remain well below that threshold. Watched phrases under 5% on "
                       "every charted surface are left off the charts (demotion candidates; "
                       "counts stay in the report JSON).")

    if title.startswith("Tic candidates"):
        # The discovery queue behind the watchlist above — render the actual
        # candidates as a table (the generic view showed only the counts, with
        # the phrases buried in gray captions). Folded into an expander: it's
        # triage work, not a headline.
        tc = report.get("tic_candidates") or {}
        _arm_labels = {"response": "pipeline responses", "plain": "plain responses",
                       "prompt": "prompts"}
        cand_rows = [{"phrase": c.get("phrase", ""),
                      "found in": _arm_labels.get(arm, arm),
                      "hits": f"{c.get('df', '?')}/{c.get('of', '?')}",
                      "other arm": (f"{c['ref_df']}/{c['ref_of']}"
                                    if c.get("ref_of") else "—"),
                      "example": c.get("example", "")}
                     for arm in ("response", "plain", "prompt")
                     for c in (tc.get(arm) or [])]
        if cand_rows:
            with st.expander(f"Pending candidates ({len(cand_rows)})", expanded=False):
                st.caption("Phrases rare in general English but recurring in one arm "
                           "(and not already watched or dismissed). Promote or dismiss "
                           "with `python evals/review_tics.py list` — recurrence across "
                           "runs, not one run's counts, is what earns a promotion.")
                st.dataframe(pd.DataFrame(cand_rows), width="stretch", hide_index=True)
            suppress_detail = True  # the table replaces the gray caption dump

    if title.startswith("Lexical diversity — prompts"):
        st.caption("Measures how varied the WORDING of the prompts is — the phrases the corpus "
                   "over-uses, scored over character n-grams. This is about how the prompts are "
                   "written, not what they are about; subject matter is measured under Meanings "
                   "and topics in the Composition and Diversity Analysis block.")
        ld = report.get("lexical_diversity") or {}
        if ld.get("cloud"):
            st.markdown(f"**Surface-form layout** — near-dup>0.90 (char n-gram) "
                        f"{ld.get('over_0.90', 0):.0%} · style Vendi ratio "
                        f"{ld.get('style_vendi_ratio', 0):.2f}")
            st.caption("Same charts as the semantic section, but in char-n-gram (writing form) "
                       "space: nearest-neighbour redundancy (dashed line = >0.90) · document "
                       "cloud (2-D PCA of surface features; hover for the record). The over-used "
                       "phrase list is demoted (mostly common English) — see the **Style "
                       "fingerprint** section for curated tic/move reuse.")
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(_nn_hist(ld.get("nn_sims") or [], 0.90,
                                         "nearest-neighbour surface cosine"),
                                use_container_width=True)
            with c2:
                st.altair_chart(_cloud_scatter(ld["cloud"]), use_container_width=True)

    if title.startswith("Style fingerprint"):
        fp = (report.get("style_fingerprint") or {}).get("pipeline") or {}
        if fp.get("points"):
            st.caption("Each dot is one response in curated-feature space (tracked tics + "
                       "rhetorical moves — no common words); dots that overlap share a "
                       "tic/move fingerprint. Dashed line on the histogram = near-twin >0.95.")
            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(_nn_hist([p["nn"] for p in fp["points"]], 0.95,
                                         "nearest-neighbour fingerprint cosine"),
                                use_container_width=True)
            with c2:
                st.altair_chart(_cloud_scatter(
                    [{"id": p["id"], "x": p["x"], "y": p["y"],
                      "snippet": ", ".join(p["features"]) or "(no tics/moves)"}
                     for p in fp["points"]]), use_container_width=True)

    if not suppress_detail:
        for line in section.get("detail", []):
            st.caption(line)


# --- Diversity — one bigger section holding the diversity measures: the
# rhetorical moves and tracked tics the answers reuse (promoted here from the
# Health check) and semantic diversity (what they're about). All subsections
# under this header. ---
diversity = loader.load_diversity(run.run_dir)
# Rhetorical moves + tracked tics are promoted into Diversity (and skipped in the
# Health-check group loop below). The tic-candidates review queue stays DOWN in
# the Health check — it's triage work, not the headline diversity story — and
# keeps its custom expander rendering wherever it renders. "Stock phrases" is
# the legacy tics title.
_DIVERSITY_PROMOTED = ("Rhetorical moves", "Tracked tics", "Stock phrases")
promoted = [s for pfx in _DIVERSITY_PROMOTED for s in sections
            if s.get("title", "").startswith(pfx)]

# The two judges' dimension diagnostics — headline signals, rendered right
# after the Pareto headline (before the diversity analysis), not down in the
# Health check. Welfare first: it is the axis the pipeline exists to move.
_render_welfare()
_render_delivery()

# Corpus length — promoted right under Delivery quality (the per-record chart
# + the how-much-longer line; skipped in the Health-check loop below).
_render_lengths()

# Showcase — the three concrete pipeline-beats-plain examples, above the
# diversity analysis.
_render_showcase()

if promoted or diversity is not None:
    st.header("Composition and Diversity Analysis")
    st.caption("How varied the responses are across several dimensions: the **rhetorical "
               "moves** they make (classified by an LLM), the **wording and phrases** they "
               "repeat (detected automatically), and the **meanings or topics** they cover "
               "(measured using embedding similarity).")

# Rhetorical moves + tracked tics — promoted from the Health check.
for section in promoted:
    _render_section(section)

# --- Semantic diversity (embeddings) — a separate report file, rendered when
# evals/diversity.py has run on this run dir. A subsection of Diversity, above
# the Health check tail. The lexical sections point here for topic/meaning
# diversity.
if diversity is None:
    st.caption("No semantic diversity report yet — generate it (embedding cents) with:")
    st.code(f"python evals/diversity.py --input {run.run_dir} --ideas", language="bash")
else:
    st.subheader("Meanings and topics (prompts and responses)")
    st.caption("Measures how varied the semantic diversity of the prompt and response pairs are. "
               "Similarity is measured with embeddings, so two records count as alike when they "
               "cover the same subject even in completely different words. Embedding model used "
               f"this run: `{diversity.get('embed_model')}`.")

    scopes = diversity.get("scopes") or {}
    # Only the combined (prompt + response) record is shown — the separate
    # prompt-only and response-only breakdowns were dropped as low-value.
    shown = [("combined", "Combined (prompt + response)")] if "combined" in scopes else []
    if shown:
        st.caption("Each record shown two ways. **Redundancy** is how close each "
                   "record sits to its nearest neighbour, where bars past the dashed line are "
                   "near-duplicates and lower is more varied. **Topic spread** groups the records "
                   "into meaning clusters, where many even bars mean many distinct topics and one "
                   "tall bar means they clump onto a single one.")
        for key, label in shown:
            blk = scopes[key]
            c = blk.get("clusters") or {}
            n = blk.get("n") or 0
            over = blk.get("over") or {}
            k_clusters = c.get("k") or len(c.get("sizes") or [])
            vr = blk.get("vendi_ratio", 0)
            st.markdown(f"**{label}**")
            col1, col2 = st.columns(2)
            with col1:
                st.altair_chart(_nn_hist(blk.get("nn_sims") or [], 0.90,
                                         "nearest-neighbour cosine"),
                                use_container_width=True)
                st.caption(f"**Redundancy** — {over.get('0.90', 0):.0%} near-duplicate (>0.90), "
                           f"{over.get('0.80', 0):.0%} similar (>0.80). Lower is more varied.")
            with col2:
                st.altair_chart(_cluster_bars(c.get("sizes") or []),
                                use_container_width=True)
                st.caption(f"**Topic spread** — evenness {c.get('evenness', 0):.3f} across "
                           f"{k_clusters} clusters, largest holding "
                           f"{c.get('largest_share', 0):.0%} of records. Higher evenness is "
                           "more distinct topics.")
            # The Vendi effective-count kept as a text stat (its 2-D cloud chart
            # was removed — unlabelable PCA axes confused readers).
            st.caption(f"**{vr * n:.1f} of {n} records effectively distinct** in meaning "
                       f"(Vendi ratio {vr:.2f}). Higher is more varied.")
            # What each topic-spread bar IS: k-means clusters are unlabeled, so
            # each is shown by its most central member — same styled-list
            # treatment as the moves legend. Reports written before
            # clusters.detail existed just don't get the expander.
            c_detail = c.get("detail") or []
            if c_detail:
                with st.expander(f"What each cluster is — {label} "
                                 f"({len(c_detail)} clusters)", expanded=False):
                    st.caption("Clusters are unlabelled groups of records with similar meaning, "
                               "numbered to match the topic-spread bars (largest first). Each is "
                               "shown by its most central record, which is a typical member "
                               "rather than a name for the group.")
                    st.markdown("\n".join(
                        f"- **cluster {i + 1} ({d['size']} records)** — most central: "
                        f"{d.get('rep_id', '?')}  \n  *“{d.get('rep', '')}”*  \n"
                        f"  members: {', '.join(d.get('ids', [])[:10])}"
                        + (f" +{len(d['ids']) - 10} more" if len(d.get('ids', [])) > 10 else "")
                        for i, d in enumerate(c_detail)))

        with st.expander("Full diversity tables (corpus totals + per-scope)", expanded=False):
            for section in diversity.get("sections") or []:
                st.markdown(f"**{section.get('title', '')}**")
                _section_table(section)
                for line in section.get("detail", []):
                    st.caption(line)

    ideas = diversity.get("ideas") or {}
    if ideas.get("nn_sims"):
        st.markdown(f"**Idea-level diversity** — {ideas['n']} one-line scenario summaries; "
                    f"{ideas.get('over_0.95', 0):.0%} share their core idea with another "
                    "(dashed line = the >0.95 re-skinned-idea threshold)")
        st.altair_chart(_nn_hist(ideas["nn_sims"], 0.95,
                                 "nearest-neighbour similarity of idea summaries"),
                        use_container_width=True)
    elif not ideas:
        st.caption("Idea-level pass not run — add `--ideas` for re-skinned-scenario detection.")

# --- Health check (everything else): the overview table, batch totals, then
# the bucketed prompt/response/library checks.
# These catch drift; they are not the dataset's usefulness story above. ---
st.header("Health check")
st.caption("An honest accounting of the **stylistic footprint** this data would leave on a "
           "model trained on it — its length, recurring phrases, and rhetorical habits — so you "
           "can judge that it won't harm the model. Most of what shows up here is benign; where "
           "the pipeline leans on something, it is usually **trading one habit for another** (a "
           "plain-Claude tic for a pipeline one), not adding a new risk. Each check below says "
           "what it measures and why. Read them for regressions across runs, not as targets to "
           "chase.")
_render_health_overview()

# _NOT_DISPLAYED sections are deliberately hidden (still measured — report JSON
# and terminal keep them); _RETIRED_SECTIONS are old reports' considerations-era
# paid sections, whose data stays in the JSON but is never rendered.
# Insider-vocabulary leak is bucketed "response" but rendered LAST (after the
# group loop) — scaffolding-bleed is the closing note of the health check.
_RENDER_LAST = ("Insider-vocabulary leak",)
_SKIP_SECTIONS = (("Response lengths",)  # lengths promoted under Delivery quality
                  + _RETIRED_SECTIONS + _DIVERSITY_PROMOTED
                  + _PAID_COMPANIONS + _NOT_DISPLAYED + _RENDER_LAST)
_GROUP_HEADERS = {
    "prompt": "Prompt side — the shipped user messages",
    "response": "Response side — final replies vs the plain-Claude control",
    "library": "Reasoning library — selection & coverage",
    "paid": "Paid LLM checks",
    "other": "Other checks",
}

_by_group: dict = {}
for section in sections:
    if not section.get("title", "").startswith(_SKIP_SECTIONS):
        _by_group.setdefault(rendering.audit_section_group(section), []).append(section)

for group in rendering.AUDIT_GROUP_ORDER:
    group_sections = _by_group.get(group) or []
    if not group_sections:
        continue
    st.header(_GROUP_HEADERS[group])
    for section in group_sections:
        _render_section(section)

# Rendered last: scaffolding-vocabulary bleed — the closing check.
for section in sections:
    if section.get("title", "").startswith(_RENDER_LAST):
        _render_section(section)

common.json_block(report, f"audit_{run.run_id}", "Raw report JSON")
