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
    _judge_model = _short_model((report.get("moral_patient_reasons") or {}).get("judge_model")
                                or (report.get("moral_patient_reasons") or {}).get("model")
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
        f"stages were generated with **`{_pipe_model}`**; the audit's reasoning judge used "
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
            "into moralizing. The result is the shipped assistant turn.\n\n"
            "**Control arm — plain model, no system prompt**  \n"
            "For every dilemma, a plain-model call answers with no system prompt. It serves as "
            "both a matched control each pipeline answer is measured against and a \"first take\" "
            "that the step 2 response drafting stage can reference.\n\n"
            "**This audit** runs offline checks (repeated phrasing/tics, length, locale "
            "plausibility), the Composition and Diversity Analysis above (the kinds of reasoning "
            "and rhetorical moves the responses use, the phrases they repeat, and the meanings and "
            f"topics they cover), and a paid reasoning judge (`{_judge_model}`) that extracts the "
            "valuable welfare considerations each arm raises and scores, item by item, which of "
            "plain Claude's survive into the pipeline answer and what the pipeline adds.")
    st.divider()

# prompt_id -> this run's stable gids, so the per-case audit charts and
# breakdowns label by the record they're about — responses by R-####, the
# finished example by E-#### — not the per-run prompt id. Loaded once from
# the run's rewrites. (Hoisted above the headline: the survival chart up
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

# --- Headline: valuable welfare considerations (the dataset's usefulness, up top) ---
# ONE measure — valuable welfare considerations per answer — from a single unified
# extraction. The top-line story is deliberately immediate: one bar per arm,
# green (pipeline) reads higher than terracotta (plain), higher is better. The
# reasoning/alternative split is a SEPARATE labelled chart below, not opacity
# shades on the same bar (which read as two different constructs).
_ic = (report.get("valuable_welfare_considerations")
       or report.get("important_considerations") or {})
if _ic.get("available"):
    st.header("Valuable welfare considerations")
    st.caption("The main metric we optimize for — it measures the important welfare-relevant "
               "substance each response brings. An LLM judge reads each answer and pulls out "
               "every consideration that clears the bar: a **distinct** point that either weighs "
               "a being's interests (welfare reasoning) or proposes a concrete lower-harm action "
               "(a humane alternative). Paraphrases and restatements of the same point are merged "
               "into one, and anything with no moral patient behind it — pure cost, logistics, or "
               "legal risk — does not count. The judge does the same for the plain model "
               "answering with no system prompt, so the pipeline is always read against that "
               "baseline. Higher is better.")
    if _ic.get("parent", {}).get("plain"):
        _lift = (_ic["parent"]["pipeline"] / _ic["parent"]["plain"] - 1) * 100
        st.markdown(f"Overall, the pipeline adds **{_lift:.0f}% more** valuable welfare "
                    "considerations "
                    "than plain.")
    # GRAPH 1 — top-line: one plain bar per arm (total considerations/answer), labelled.
    _totals = pd.DataFrame([{"arm": "plain Claude", "total": _ic["parent"]["plain"]},
                            {"arm": "pipeline", "total": _ic["parent"]["pipeline"]}])
    _arms = ["plain Claude", "pipeline"]
    _top = alt.Chart(_totals).mark_bar().encode(
        y=alt.Y("arm:N", title="", sort=_arms),
        x=alt.X("total:Q", title="valuable welfare considerations per answer"),
        color=alt.Color("arm:N", title="", scale=alt.Scale(
            domain=list(rendering.AUDIT_ARM_COLUMNS), range=list(rendering.AUDIT_ARM_COLORS)),
            legend=None),
        tooltip=["arm", alt.Tooltip("total:Q", title="per answer", format=".2f")])
    _top_labels = alt.Chart(_totals).mark_text(align="left", dx=5, fontWeight="bold").encode(
        y=alt.Y("arm:N", sort=_arms), x=alt.X("total:Q"), text=alt.Text("total:Q", format=".1f"))
    st.altair_chart((_top + _top_labels).properties(height=110), use_container_width=True)

    # Breakdown, as its OWN labelled chart (Oliver: two clearly-labeled facets,
    # not dual-opacity). Grouped bars, one facet per row, arm as hue.
    _subs = _ic["subsets"]
    st.markdown("Valuable welfare considerations split into:  \n"
                "1. **welfare reasoning**: a point weighing a being's interests  \n"
                "2. **humane alternatives**: a concrete lower-harm action the user could take")
    _brk = [{"arm": arm_col, "facet": s["name"], "value": s[arm_key]}
            for s in _subs
            for arm_key, arm_col in (("plain", "plain Claude"), ("pipeline", "pipeline"))]
    _facets = [s["name"] for s in _subs]
    _brk_chart = alt.Chart(pd.DataFrame(_brk)).mark_bar().encode(
        y=alt.Y("facet:N", title="", sort=_facets),
        yOffset=alt.YOffset("arm:N", sort=_arms),
        x=alt.X("value:Q", title="per answer"),
        color=alt.Color("arm:N", title="arm", scale=alt.Scale(
            domain=list(rendering.AUDIT_ARM_COLUMNS), range=list(rendering.AUDIT_ARM_COLORS))),
        tooltip=["facet", "arm", alt.Tooltip("value:Q", title="per answer", format=".2f")])
    st.altair_chart(_brk_chart.properties(height=140), use_container_width=True)

    if _ic.get("length_ratio"):
        _len = ("On average, the pipeline response length is "
                f"**{(_ic['length_ratio'] - 1) * 100:.0f}% longer** than plain")
        if _ic.get("retained_share") is not None:
            _len += f", keeps **{_ic['retained_share']:.0%}** of the considerations plain raised"
            if _ic.get("added_share") is not None:
                _len += f" and adds **{_ic['added_share']:.0%}** more"
        st.markdown(_len + ".")

    _hd_per_case = (report.get("moral_patient_reasons") or {}).get("per_case") or {}

    # The per-record pipeline-vs-plain pair chart is NOT here — it lives down in
    # the Health check (lower-value detail). This headline keeps the retention
    # story below.
    # GRAPH — the per-record fate of plain's considerations (kept / weakened /
    # dropped, + what the pipeline added).
    _hd_surv = _label_responses(rendering.audit_survival_chart_rows(_hd_per_case))
    if _hd_surv:
        st.markdown("**Comparison of plain's considerations:**  \n"
                    "*Dropped* = a consideration plain Claude raised that this pipeline answer "
                    "didn't echo  \n"
                    "*Added* = new considerations the pipeline brought")
        # Stacked survival chart — hover a segment to see WHICH considerations
        # sit in it. Bottom three segments sum to the plain arm's count.
        st.altair_chart(
            alt.Chart(pd.DataFrame(_hd_surv)).mark_bar().encode(
                x=alt.X("record:N", title="record"),
                y=alt.Y("count:Q", title="considerations"),
                color=alt.Color("category:N", title="", scale=alt.Scale(
                    domain=list(rendering.AUDIT_SURVIVAL_CATEGORIES),
                    range=list(rendering.AUDIT_SURVIVAL_COLORS)),
                    sort=list(rendering.AUDIT_SURVIVAL_CATEGORIES)),
                order=alt.Order("stack_order:Q", sort="ascending"),
                tooltip=[alt.Tooltip("record", title="record"),
                         alt.Tooltip("category", title="fate"),
                         alt.Tooltip("count", title="count"),
                         alt.Tooltip("reasons", title="which considerations")],
            ),
            use_container_width=True)

    # The worked examples behind the chart: per-response kept / weakened /
    # dropped / added lists, one drop-down with a picker inside it, labelled by
    # stable ids (response R-#### · example E-####).
    _hd_pids = sorted(_hd_per_case)
    if _hd_pids:
        with st.expander("What considerations were kept, weakened, dropped and added for "
                         f"each response ({len(_hd_pids)})", expanded=False):
            choice = st.selectbox("Response", _hd_pids, format_func=_resp_label,
                                  key="reasons_percase_pick")
            st.caption(f"{_resp_label(choice)} — considerations kept / weakened / dropped / "
                       "added (plain vs pipeline)")
            common.show_reason_comparison(_hd_per_case[choice])
            entry_ids = pulls.get(choice) or []
            # Folded behind its own toggle: the pulled rows are context, not
            # the comparison the drop-down is opened for.
            if entry_ids and st.toggle(
                    f"Library entries pulled at 2a.5 ({len(entry_ids)}) — "
                    "id + transferable move",
                    value=False, key=f"lib_pulls_{choice}"):
                for eid in entry_ids:
                    move = lib_moves.get(eid, "")
                    st.markdown(f"- **{eid}**{' — ' + move if move else ''}")
    st.divider()
elif _ic.get("available") is False:
    st.info("Run the audit with `--reasons` to populate the valuable-welfare-considerations summary.")

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
    "Reasoning-composition diversity": "Kinds of welfare reasoning (responses)",
    # both the pre- and post-2026-07-25 titles map to the same display name
    "Tracked tics (responses)": "Phrases (prompts and responses)",
    "Tracked tics (prompts + responses)": "Phrases (prompts and responses)",
    # legacy pre-rename titles → the current name, so old reports read the same
    "Important considerations (LLM)": "Valuable welfare considerations (LLM)",
    "Important considerations": "Valuable welfare considerations",
}
# Alternatives + stance ride the same paid pass as the reasons section; they
# render right after it so the judge's outputs read together.
_PAID_COMPANIONS = ("Humane alternatives", "Delivery quality", "Response stance",
                    "Reasoning-composition", "Showcase examples")
# Sections whose detail lines are replaced by a richer custom view below, so
# the generic gray-caption dump is suppressed for them. "Stock phrases" is the
# legacy pre-tics name; old reports keep it.
_CUSTOM_DETAIL = ("Tracked tics", "Stock phrases")

def _render_health_overview() -> None:
    """Verdict overview table + batch totals. A health-check summary, so it
    renders in the health-check tail below the dataset-usefulness sections."""
    summary = rendering.audit_verdict_summary(report)
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

def _render_considerations_table(section: dict) -> None:
    """The valuable-welfare-considerations per-record pair chart + numeric rows — the
    lower-value detail (the headline charts tell the story), so it renders in the
    Health check tail. Carries the section anchor the verdict-summary table links
    to, plus a one-line paid-pass provenance caption."""
    title = section.get("title", "")
    st.subheader(_DISPLAY_TITLES.get(title, title), anchor=_slug(title))
    mpr = report.get("moral_patient_reasons") or {}
    per_case = mpr.get("per_case") or {}
    n_pipe = (mpr.get("pipeline") or {}).get("n")
    n_plain = (mpr.get("plain") or {}).get("n")
    bits = []
    if n_pipe is not None and n_plain is not None:
        bits.append(f"means over pipeline {n_pipe} / plain {n_plain} answers extracted")
    if mpr.get("cost_usd") is not None:
        bits.append(f"paid pass ${mpr['cost_usd']:.4f} · model `{mpr.get('model') or '?'}`")
    if bits:
        st.caption(" · ".join(bits) + ".")

    # Per response: one pipeline/plain pair per record — spot an answer that runs
    # lean or an extraction gap where a bar is missing.
    chart_rows = _label_responses(rendering.audit_reason_chart_rows(per_case))
    if chart_rows:
        st.markdown("**Per response** — pipeline vs plain, one pair per record.")
        fail = mpr.get("failures") or 0
        if fail:
            missing = [f"{(_gids_by_pid.get(pid) or {}).get('response') or pid} ({arm})"
                       for pid, e in per_case.items() for arm in ("plain", "pipeline")
                       if (e.get(arm) or {}).get("reasons") is None]
            st.caption(f"⚠️ {fail} extraction failure(s) excluded from the means (a missing bar "
                       "is a gap, not a zero)"
                       + (f": {', '.join(sorted(missing))}" if missing else "") + ".")
        st.altair_chart(_grouped_arm_chart(chart_rows, "considerations"),
                        use_container_width=True)

    _section_table(section)
    for line in section.get("detail", []):
        st.caption(line)


def _render_corpus_dumps() -> None:
    """The full corpus-level distinct-consideration lists per arm, behind
    expanders. Low-signal, so they sit at the very bottom of the Health check."""
    mpr = report.get("moral_patient_reasons") or {}
    for arm, arm_title in (("plain", "Plain Claude"), ("pipeline", "Pipeline")):
        corpus = (mpr.get(arm) or {}).get("corpus_reasons") or []
        if corpus:
            with st.expander(f"Corpus-level distinct considerations — {arm_title} ({len(corpus)})"):
                for reason in corpus:
                    st.markdown(f"- {reason}")


def _render_delivery() -> None:
    """Delivery quality: the 0-10 per-response manner score + the Pareto scatter
    against valuable welfare considerations. Rendered near the top (right after
    the considerations headline) — it's a headline signal, not a health check."""
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
        line = f"Mean delivery quality: pipeline **{pm * 10:.0f}%**"
        if bm is not None:
            line += f" vs plain **{bm * 10:.0f}%**"
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
                    f"{dims[arm].get(k) * 10:.0f}%" if dims[arm].get(k) is not None else "—"
                    for k in hdr) + " |")
        st.markdown("\n".join(rows_md))
        st.caption("Mean per judged dimension — where the delivery gap lives. "
                   "Diagnostics from the same judge call; the headline score is holistic, "
                   "not their average.")

    st.subheader("Welfare considerations ↔ delivery quality",
                 anchor=_slug("Considerations vs delivery"))
    mpr = report.get("moral_patient_reasons") or {}
    tradeoff = ("The aim is to add more substantive welfare reasoning without trading off "
                "too much helpful, unobtrusive delivery.")
    p_mean = (mpr.get("pipeline") or {}).get("mean_unique")
    b_mean = (mpr.get("plain") or {}).get("mean_unique")
    if None not in (pm, bm, p_mean, b_mean) and b_mean:
        tradeoff += (f" In this run, vs plain: **{p_mean / b_mean - 1:+.0%}** considerations, "
                     f"**{pm / bm - 1:+.0%}** delivery quality.")
    st.markdown(tradeoff)

    mpr_pc = mpr.get("per_case") or {}
    rows = _label_responses(rendering.audit_delivery_pareto_rows(per_case, mpr_pc))
    if rows:
        st.caption("Each dot is one response — **x = delivery quality** (manner), "
                   "**y = valuable welfare considerations** (substance). Up-and-to-the-right is "
                   "the goal: more substance without losing delivery. Pipeline (green) vs plain "
                   "Claude (terracotta); the large diamonds mark each arm's corpus mean. Hover "
                   "for the record and the judge's one-line reason.")
        df = pd.DataFrame(rows)
        arm_color = alt.Color("arm:N", title="arm", scale=alt.Scale(
            domain=list(rendering.AUDIT_ARM_COLUMNS), range=list(rendering.AUDIT_ARM_COLORS)))
        # Whole-point scores: pin the axis to integer ticks so the scale never
        # renders 0.4-style gradations for a grade that cannot take them.
        x_axis = alt.X("delivery:Q", title="delivery quality (0–10)",
                       scale=alt.Scale(domain=[0, 10]),
                       axis=alt.Axis(values=list(range(0, 11))))
        scatter = alt.Chart(df).mark_circle(size=90, opacity=0.7).encode(
            x=x_axis,
            y=alt.Y("considerations:Q", title="valuable welfare considerations per answer"),
            color=arm_color,
            tooltip=[alt.Tooltip("record", title="record"), "arm",
                     alt.Tooltip("considerations", title="considerations"),
                     alt.Tooltip("delivery", title="delivery (0–10)"),
                     alt.Tooltip("note", title="why")])
        # Corpus means, one diamond per arm — the whole-arm summary the dots
        # scatter around, kept visually distinct (shape + outline).
        means = df.groupby("arm", as_index=False)[["delivery", "considerations"]].mean()
        means["label"] = means["arm"] + " mean"
        mean_marks = alt.Chart(means).mark_point(
            shape="diamond", size=380, filled=True, opacity=1,
            stroke="#1f1f1f", strokeWidth=1.5).encode(
            x=x_axis, y="considerations:Q", color=arm_color,
            tooltip=[alt.Tooltip("label", title=""),
                     alt.Tooltip("delivery", title="mean delivery", format=".1f"),
                     alt.Tooltip("considerations", title="mean considerations", format=".1f")])
        st.altair_chart((scatter + mean_marks).properties(height=360),
                        use_container_width=True)

    # Low-scoring pipeline responses, with the judge's one-line reason — the
    # "which answers landed poorly, and why" review hook.
    low = sorted(((pid, e["pipeline"]) for pid, e in per_case.items()
                  if (e.get("pipeline") or {}).get("score") is not None
                  and e["pipeline"]["score"] < dv.get("flag_below", 5)),
                 key=lambda kv: kv[1]["score"])
    if low:
        with st.expander(f"Low-delivery pipeline responses ({len(low)})", expanded=False):
            for pid, d in low:
                st.markdown(f"- **{_resp_label(pid)}** — **{d['score'] * 10}%**"
                            + (f": *{d['note']}*" if d.get("note") else ""))
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
    """Three concrete pipeline-beats-plain examples (from the paid showcase
    pass): summary + the exact improved spans highlighted, with the full
    side-by-side comparison behind an expander. Sentence-level highlights by
    design — a text diff marks whole reflowed paragraphs, which buries the one
    sentence that actually changed."""
    examples = (report.get("showcase") or {}).get("examples") or []
    if not examples:
        return
    st.header("Where the pipeline made it better",
              anchor=_slug("Showcase examples (LLM)"))
    st.caption("Three concrete cases, one per kind of improvement, selected by an LLM "
               "judge from the retention and delivery data. The **highlighted text** is "
               "the exact place the improvement lives; expand each case to compare the "
               "full responses side by side.")
    for ex in examples:
        gid = _resp_label(ex["prompt_id"])
        d = ex.get("delivery") or {}
        dnote = (f" · delivery {d['pipeline'] * 10}% vs plain {d['plain'] * 10}%"
                 if d.get("pipeline") is not None and d.get("plain") is not None else "")
        st.markdown(f"#### {ex['label']} — {gid}")
        st.caption(f"why this example: judged fit {ex.get('fit') * 10}%{dnote}")
        st.markdown(ex["summary"])
        with st.expander(f"Compare the full responses — {gid}", expanded=False):
            st.markdown("**The user asked:**")
            st.markdown(f"> {ex.get('user_message', '').strip()}")
            col_plain, col_pipe = st.columns(2)
            with col_plain:
                st.markdown("**Plain Claude**")
                st.markdown(_highlighted_html(ex.get("plain_response", ""), []),
                            unsafe_allow_html=True)
            with col_pipe:
                st.markdown("**Pipeline** (improvements highlighted)")
                st.markdown(_highlighted_html(ex.get("pipeline_response", ""),
                                              ex.get("highlights")),
                            unsafe_allow_html=True)
    st.divider()


def _render_alternatives_section(section: dict) -> None:
    """Humane alternatives (LLM): per-record chart + collapsed per-response
    citations (mirrors the reasons breakdown)."""
    title = section.get("title", "")
    st.subheader(title, anchor=_slug(title))
    gloss = rendering.audit_section_gloss(section)
    if gloss:
        st.caption(gloss)
    _section_table(section)
    moves_pc = (report.get("moves") or {}).get("per_case") or {}
    alt_rows = _label_responses(rendering.audit_alternative_chart_rows(moves_pc))
    if alt_rows:
        st.caption("Concrete lower-harm alternatives each arm proposes, per response "
                   "(actions, not considerations). The pipeline-over-plain gap is the "
                   "\"how, not whether\" signal.")
        st.altair_chart(_grouped_arm_chart(alt_rows, "alternatives"),
                        use_container_width=True)
    # Per-response citations under one collapsed drop-down (mirrors the
    # reasons breakdown): which alternatives each arm actually offered.
    pids = sorted(moves_pc)
    if pids:
        with st.expander(f"Per-response alternatives ({len(pids)})", expanded=False):
            choice = st.selectbox("Response", pids, format_func=_resp_label,
                                  key="alts_percase_pick")
            st.caption(f"{_resp_label(choice)} — plain Claude's alternatives judged against "
                       "the pipeline's response, plus what the pipeline added.")
            groups = rendering.audit_alternative_groups(
                (moves_pc[choice] or {}).get("alternatives") or {})
            for gtitle, items in groups or []:
                st.markdown(f"**{gtitle}**")
                if items:
                    st.markdown("\n".join(f"- {a}" for a in items))
                else:
                    st.caption("none")
    for line in section.get("detail", []):
        st.caption(line)


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
    # Reasoning-composition likewise: the share bars + similarity cloud carry it,
    # so the numeric rows are dropped (they stay in the report JSON/terminal).
    _composition_charted = (title.startswith("Reasoning-composition")
                            and bool(((report.get("reason_composition") or {}).get("pipeline")
                                      or {}).get("points")))
    if not (_moves_described or _tics_charted or _composition_charted):
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

    if title.startswith("Reasoning-library coverage"):
        # Retrieval width vs added reasoning — the correlation view: does
        # pulling more library rows at 2a.5 come with more pipeline-added
        # reasons? (Needs the paid --reasons survival data.)
        per_case = (report.get("moral_patient_reasons") or {}).get("per_case") or {}
        scatter_rows = _label_responses(rendering.audit_pull_scatter_rows(per_case, pulls))
        if scatter_rows:
            df = pd.DataFrame(scatter_rows)
            r = (df["pulled"].corr(df["added"])
                 if len(df) >= 3 and df["pulled"].nunique() > 1 else None)
            st.caption("Each point is one record: library rows pulled at 2a.5 (x) vs "
                       "reasons the pipeline added beyond plain Claude (y, from the "
                       "survival judge). Hover for the record and which entries."
                       + (f" Pearson r = {r:.2f} over {len(df)} records."
                          if r is not None and not pd.isna(r) else ""))
            points = alt.Chart(df).mark_circle(
                color=rendering.AUDIT_PULL_COLOR, size=70, opacity=0.7).encode(
                x=alt.X("pulled:Q", title="library rows pulled (2a.5)"),
                y=alt.Y("added:Q", title="pipeline-added reasons"),
                tooltip=[alt.Tooltip("record", title="record"),
                         alt.Tooltip("pulled", title="rows pulled"),
                         alt.Tooltip("added", title="added reasons"),
                         alt.Tooltip("entries", title="which entries")],
            )
            trend = points.transform_regression("pulled", "added").mark_line(
                color=rendering.AUDIT_PULL_COLOR, strokeDash=[4, 3])
            st.altair_chart((points + trend).properties(height=260),
                            use_container_width=True)

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
            mv_rows = []
            for name, d in moves.items():
                mv_rows.append({"move": name, "arm": "pipeline",
                                "share": d.get("pipeline_share") or 0})
                mv_rows.append({"move": name, "arm": "plain Claude",
                                "share": d.get("plain_share") or 0})
            order = sorted(moves, key=lambda m: -(moves[m].get("pipeline_share") or 0))
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
            st.altair_chart((bars + flag + flag_txt).properties(height=max(180, 34 * len(moves))),
                            use_container_width=True)
            st.caption("We are only concerned with specific rhetorical moves dominating when it "
                       "appears in more than half of responses. Most moves remain well below "
                       "that threshold.")
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

            # --- the responses (two arms: pipeline vs the plain baseline)
            resp_rows = [r for r in phrase_rows if r.get("pipeline") or r.get("plain")][:12]
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
            # plain-model prompt to compare against)
            prompt_rows = [r for r in phrase_rows if r.get("prompts")][:12]
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
                       "phrases remain well below that threshold.")

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

    if title.startswith("Reasoning-composition"):
        rc = report.get("reason_composition") or {}
        pt = (rc.get("pipeline") or {})
        if pt.get("points"):
            # Just the share bars now — the 2-D similarity cloud was removed
            # (its PCA axes have no nameable meaning and confused readers more
            # than the layout informed them).
            share = (pt.get("mean_share") or {})
            bar_rows = [{"reason type": t, "mean share": share[t]}
                        for t in (rc.get("types") or []) if share.get(t)]
            if bar_rows:
                st.altair_chart(alt.Chart(pd.DataFrame(bar_rows)).mark_bar(
                    color="#4C78A8").encode(
                    x=alt.X("mean share:Q", title=None, axis=alt.Axis(format="%")),
                    # labelOverlap=False forces a label on EVERY bar (Vega
                    # was auto-hiding every other one).
                    y=alt.Y("reason type:N", sort="-x", title="",
                            axis=alt.Axis(labelOverlap=False)),
                    tooltip=["reason type", alt.Tooltip("mean share:Q", format=".0%")],
                ).properties(height=240), use_container_width=True)
                st.caption("Share of each welfare reasoning type in the overall corpus")
            # The type legend, styled like the moves list (bold name — meaning)
            # and folded into an expander for the same reason: it's reference
            # material, not something to scroll past to reach the charts. New
            # reports carry type_gloss; older ones fall back to parsing this
            # section's "type: gloss" detail lines.
            gloss_map = rc.get("type_gloss") or {}
            if not gloss_map:
                for line in section.get("detail", []):
                    name, sep, g = line.partition(": ")
                    if sep and name and " " not in name:
                        gloss_map[name] = g
            shown = sorted((t for t in (rc.get("types") or [])
                            if share.get(t) and gloss_map.get(t)),
                           key=lambda t: -share[t])
            if shown:
                with st.expander("What each welfare reasoning type means", expanded=False):
                    st.markdown("\n".join(f"- **{t}** — {gloss_map[t]}" for t in shown))
                suppress_detail = True  # the styled list replaces the caption dump

    if not suppress_detail:
        for line in section.get("detail", []):
            st.caption(line)


# --- Dataset usefulness cluster (top): Valuable welfare considerations (above) → its
# detailed view, measured on the RESPONSES (final assistant replies): the
# unified considerations pass, then the reasoning-composition mix. The paid
# section is titled "Valuable welfare considerations (LLM)" now; the legacy titles are
# matched too so old reports (separate reasons + alternatives sections) still
# render richly through the same renderer. ---
_REASONS_TITLES = ("Valuable welfare considerations (LLM)", "Important considerations (LLM)",
                   "Welfare reasoning", "Welfare considerations", "Moral-patient reasons")

# Resolved here so it's available everywhere, but the per-response detail (charts
# + table) is NOT rendered here — it renders DOWN in the Health check, since the
# headline above is the high-value view and this is the drill-down.
reasons_section = next((s for s in sections
                        if s.get("title", "").startswith(_REASONS_TITLES)), None)

# --- Diversity — one bigger section holding the diversity measures: the
# reasoning-composition mix (how the answers reason), the rhetorical moves and
# tracked tics they reuse (promoted here from the Health check), and semantic
# diversity (what they're about). All subsections under this header. ---
composition_section = next((s for s in sections
                            if s.get("title", "").startswith("Reasoning-composition")), None)
diversity = loader.load_diversity(run.run_dir)
# Rhetorical moves + tracked tics are promoted into Diversity (and skipped in the
# Health-check group loop below). The tic-candidates review queue stays DOWN in
# the Health check — it's triage work, not the headline diversity story — and
# keeps its custom expander rendering wherever it renders. "Stock phrases" is
# the legacy tics title.
_DIVERSITY_PROMOTED = ("Rhetorical moves", "Tracked tics", "Stock phrases")
promoted = [s for pfx in _DIVERSITY_PROMOTED for s in sections
            if s.get("title", "").startswith(pfx)]

# Delivery quality — a headline signal, rendered right after the considerations
# section (before the diversity analysis), not down in the Health check.
_render_delivery()

# Showcase — the three concrete pipeline-beats-plain examples, right under the
# Pareto view and above the diversity analysis.
_render_showcase()

if composition_section or promoted or diversity is not None:
    st.header("Composition and Diversity Analysis")
    st.caption("How varied the responses are across several dimensions: the **kinds of "
               "reasoning** they use (classified by an LLM), the **rhetorical moves** they "
               "make (classified by an LLM), the **wording and phrases** they repeat "
               "(detected automatically), and the **meanings or topics** they cover "
               "(measured using embedding similarity).")

if composition_section:
    _render_section(composition_section)

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

# --- Health check (everything else): the overview table, batch totals, the
# stance/moralizing tripwire, then the bucketed prompt/response/library checks.
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

# Valuable welfare considerations — only the numeric detail-rows table lands here now
# (every chart moved UP to the headline it substantiates). Low-value reference.
if reasons_section:
    _render_considerations_table(reasons_section)
elif "moral_patient_reasons" not in report:
    st.caption("Welfare-consideration extraction hasn't run for this report — add it "
               "(costs API calls) with:")
    st.code(f"{cmd} --reasons", language="bash")
for section in sections:
    if section.get("title", "").startswith("Humane alternatives"):
        _render_alternatives_section(section)

# Legacy stance section (pre-delivery reports) still renders here if present;
# current reports carry Delivery quality instead, rendered near the top.
for section in sections:
    if section.get("title", "").startswith("Response stance"):
        _render_section(section)

# _NOT_DISPLAYED sections are deliberately hidden (still measured — report JSON
# and terminal keep them). The usefulness cluster + stance are rendered above.
# Insider-vocabulary leak is bucketed "response" but rendered LAST (after the
# group loop) — scaffolding-bleed is the closing note of the health check.
_RENDER_LAST = ("Insider-vocabulary leak",)
_SKIP_SECTIONS = (("Valuable welfare considerations", "Important considerations")
                  + _REASONS_TITLES + _DIVERSITY_PROMOTED
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

# Very bottom of the Health check: the full corpus-distinct consideration lists
# (low-signal reference dumps, so they close out the page).
if reasons_section:
    _render_corpus_dumps()

common.json_block(report, f"audit_{run.run_id}", "Raw report JSON")
