"""Compare two runs: run facts + matched outputs + (at the bottom) template diffs.

DAD matches examples by a *content* key (default: the user message), so a
side-by-side pair is guaranteed to be the same case. A
comparison holds one dimension fixed (the key) and diffs the rest: fix the
prompt to tune responses, or fix the scenario to tune the prompts themselves.
Each stage shows both the prompt each run sent AND the output it got back
(the lineage page's shape), diffed across the two runs.
"""

import difflib
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dad_pipeline import step2_responses
from viewer import loader, rendering
from viewer.ui_pages import common

st.title("Compare runs")

runs = loader.list_runs()
if len(runs) < 2:
    st.info("Need at least two runs to compare.")
    st.stop()

pipelines = [p for p in loader.PIPELINES if any(r.pipeline == p for r in runs)]
pipeline = st.sidebar.selectbox("Pipeline", pipelines)
pipeline_runs = [r for r in runs if r.pipeline == pipeline]
if len(pipeline_runs) < 2:
    st.info(f"Only one {pipeline} run exists — nothing to compare against.")
    st.stop()

ids = [r.run_id for r in pipeline_runs]
run_a_id = st.sidebar.selectbox("Run A (baseline)", ids, index=min(1, len(ids) - 1))
run_b_id = st.sidebar.selectbox("Run B", ids, index=0)
run_a = next(r for r in pipeline_runs if r.run_id == run_a_id)
run_b = next(r for r in pipeline_runs if r.run_id == run_b_id)
if run_a_id == run_b_id:
    st.warning("Pick two different runs.")
    st.stop()


def _run_facts(run: loader.RunInfo, side: str) -> None:
    """The run-list facts for one run, so you know what you're comparing."""
    st.markdown(f"**{side}: {run.label or run.run_id}**")
    st.caption(
        f"`{run.run_id}` · {(run.created_at or '').replace('T', ' ')[:16]}\n\n"
        f"model **{run.model or '?'}** · **{run.counts.get('final', 0)}** records · "
        f"${run.total_cost:.2f} · git `{run.git_commit}`"
        + (" · ⚠ dirty tree at run time" if run.git_dirty else "")
    )


fa, fb = st.columns(2)
with fa:
    _run_facts(run_a, "A")
with fb:
    _run_facts(run_b, "B")

if not (run_a.has_snapshot and run_b.has_snapshot):
    st.caption(":material/history: One or both runs predate prompt snapshots — "
               "their templates are reconstructed from git and may not match what actually ran.")

st.header("Matched outputs")

if pipeline != "dad":
    # --- SDF: positional/name matching, final content side by side ---
    pairs = loader.match_outputs(run_a.run_dir, run_b.run_dir, pipeline)
    if not pairs:
        st.info("No matching outputs between the two runs.")
    else:
        note = {"positional": " · positional match", "group": " · grouped by principle"}
        for pair in pairs:
            with st.expander(f"{pair.key}{note.get(pair.quality, '')}"):
                for i in range(min(len(pair.a), len(pair.b))):
                    ca, cb = st.columns(2)
                    with ca:
                        st.markdown(f"**A: {run_a_id}**")
                        st.code(pair.a[i].get("content", ""), language=None, wrap_lines=True)
                    with cb:
                        st.markdown(f"**B: {run_b_id}**")
                        st.code(pair.b[i].get("content", ""), language=None, wrap_lines=True)
                if len(pair.a) != len(pair.b):
                    st.caption(f"Unpaired extras: A has {len(pair.a)}, B has {len(pair.b)}.")
else:
    # --- DAD: content-aware matching ---
    KEY_LABELS = {
        "user_message": "user message  (same prompt → compare responses)",
        "scenario_id": "scenario id  (same scenario → compare prompts too)",
        "prompt_id": "prompt key  (content-keyed P-#### gid; positional on legacy runs)",
    }
    have_scen = loader.run_has_scenario_ids(run_a.run_dir) and loader.run_has_scenario_ids(run_b.run_dir)
    key_options = [k for k in loader.DAD_MATCH_KEYS if k != "scenario_id" or have_scen]
    # Match under every key up front, so the selector shows how many examples
    # each key pairs — no picking a key blind and hoping it matches. Default to
    # the key that pairs the most.
    results = {k: loader.match_dad(run_a.run_dir, run_b.run_dir, k) for k in key_options}
    counts = {k: len(results[k][0]) for k in key_options}
    default_key = max(key_options, key=lambda k: counts[k])
    key_by = st.sidebar.radio(
        "Match examples by", key_options,
        index=key_options.index(default_key),
        format_func=lambda k: f"{KEY_LABELS[k]}  —  {counts[k]} matched",
        help="How two examples are decided to be 'the same' for side-by-side comparison. "
             "The count is how many examples each key pairs across the two runs.",
    )
    if not have_scen:
        st.sidebar.caption(":material/info: *scenario id* is unavailable — one or both runs' "
                           "dilemmas didn't record a scenario_id (older/reused prompt sets).")

    matched, only_a, only_b = results[key_by]
    if not matched:
        st.info("No examples matched on this key. Try a different match key, or the two runs "
                "share no common examples.")
    if only_a or only_b:
        st.caption(f":material/join_inner: Matched **{len(matched)}** · only in A **{len(only_a)}** · "
                   f"only in B **{len(only_b)}** (unmatched examples aren't shown).")

    if matched:
        matched.sort(key=lambda mm: mm.label.lower())
        idx = st.selectbox(
            "Example", range(len(matched)),
            format_func=lambda i: ("⚠ " if not matched[i].same_prompt else "") + matched[i].label,
        )
        m = matched[idx]
        a, b = m.a, m.b
        # Stable prompt gids (P-####) as the visible identity; the per-run
        # the bare prompt key only differs from prompt_gid on pre-gid runs.
        st.caption(f"A `{a.prompt_gid or a.prompt_id}`  ·  B `{b.prompt_gid or b.prompt_id}`  ·  "
                   + ("✓ same user prompt" if m.same_prompt
                      else "⚠ **different user prompt** — matched on "
                           f"{key_by.replace('_', ' ')}, but the prompts differ"))

        # Prompt: always available for reference; low-emphasis when identical.
        st.markdown("#### User prompt")
        if m.same_prompt:
            if st.toggle("Show user prompt (identical in A and B)", key=f"um_{idx}"):
                st.code(a.user_message, language=None, wrap_lines=True)
        else:
            common.show_diff(a.user_message, b.user_message,
                             f"A: {run_a_id}", f"B: {run_b_id}", key=f"um_{idx}")

        manifest_a, manifest_b = loader.load_manifest(run_a.run_dir), loader.load_manifest(run_b.run_dir)
        lin_a = loader.dad_lineage_by_prompt(run_a.run_dir, a.prompt_id)
        lin_b = loader.dad_lineage_by_prompt(run_b.run_dir, b.prompt_id)

        # Response: the payload in response-tuning mode.
        st.markdown("#### Response")
        if not (a.has_final and b.has_final):
            st.caption(":material/warning: One or both responses fell back to the pre-final draft "
                       "(step-3 rewrite produced no final record).")
        common.show_diff(a.response, b.response, f"A: {run_a_id}", f"B: {run_b_id}", key=f"resp_{idx}")

        def _s3_len_delta(lin: dict) -> str | None:
            """One side's step-3 length effect: 2b draft → final, in chars."""
            aud = lin.get("rewrite") or {}
            if aud.get("draft_response") and aud.get("rewritten_response"):
                d, f = len(aud["draft_response"]), len(aud["rewritten_response"])
                return f"{d:,} → {f:,} ({f - d:+,})"
            return None

        delta_a, delta_b = _s3_len_delta(lin_a), _s3_len_delta(lin_b)
        if delta_a or delta_b:
            st.caption(f"Step-3 rewrite length (chars, 2b draft → final):  "
                       f"A {delta_a or '—'}  ·  B {delta_b or '—'}")

        def _stage_output(stage: str, lin: dict) -> tuple[str | None, str | None]:
            """(text, note) one side produced at a stage; text None = not reached."""
            d = lin.get("dilemma") or {}
            if stage == "step1_dilemmas":
                return d.get("draft_user_message") or d.get("user_message"), None
            if stage == "step1_gate":
                # 1c gate: pass/fail on the 1b draft, never a rewrite. The text
                # at this stage is the judged draft (pre-refine on composed
                # runs); the verdict rides the note.
                gate = lin.get("gate")
                if gate is None:
                    return None, "1c gate did not run for this record"
                passed = gate.get("passed")
                verdict = ("gate: passed" if passed is True
                           else "gate: FAILED, shipped anyway" if passed is False
                           else "gate: unusable reply, shipped")
                reasons = "; ".join(gate.get("failures") or [])
                judged = d.get("draft_user_message") or d.get("user_message")
                return judged, verdict + (f" — {reasons}" if reasons else "")
            if stage == "step1_refine":
                # 1d refine: the rewrite (also the sole 1c stage on older runs
                # that predate the gate). The gate does not short-circuit this
                # view — on composed runs both stages ran and both must show.
                if d.get("refine_failed"):
                    return d.get("user_message"), "every refine attempt was unusable — the 1b draft shipped"
                if d.get("draft_user_message") is None:
                    return None, "refine did not run for this record"
                note = f"notes: {d['refine_notes']}" if d.get("refine_notes") else None
                return d.get("user_message"), note
            if stage == "step2_scope":
                sc = (lin.get("scope") or {}).get("scope")
                return (step2_responses.format_scope(sc) if sc else None), None
            if stage == "step2_select":
                rec = lin.get("scope") or {}
                if rec.get("entry_ids") is None:
                    return None, None
                note = ("selection unusable — the whole library was injected"
                        if rec.get("selection_fallback") else None)
                return "\n".join(rec["entry_ids"]), note
            if stage == "step2_respond":
                return (lin.get("response") or {}).get("assistant_response"), None
            return None, None

        # Stage by stage: the system + user prompt each run actually sent
        # (split-aware render_prompt) AND the output it got back — the lineage
        # page's prompt-then-output shape, diffed across the two runs.
        st.markdown("#### Stage by stage — inputs and outputs")
        stages = [("step1_dilemmas", "Step 1b — first attempt (draft)"),
                  ("step1_gate", "Step 1c — quality gate (pass/fail)"),
                  ("step1_refine", "Step 1d — refine (review & rewrite)"),
                  ("step2_scope", "Step 2a — scope"),
                  ("step2_select", "Step 2a.5 — library selection"),
                  ("step2_respond", "Step 2b — response draft"),
                  ("step3_rewrite", "Step 3 — constitution rewrite")]
        for stage, title in stages:
            ra = rendering.render_prompt("dad", stage, run_a.run_dir, manifest_a, lin_a)
            rb = rendering.render_prompt("dad", stage, run_b.run_dir, manifest_b, lin_b)
            if not (ra.is_llm_call or rb.is_llm_call):
                continue
            with st.expander(title):
                # System collapsed by default so the user prompt is reachable
                # without scrolling the (often huge) system diff.
                if st.toggle("System prompt (diff)", value=False, key=f"{stage}_systog_{idx}"):
                    common.show_diff(ra.system or "", rb.system or "",
                                     f"A: {run_a_id}", f"B: {run_b_id}", key=f"{stage}_sys_{idx}")
                st.markdown("**User prompt**")
                common.show_diff(ra.user or "", rb.user or "",
                                 f"A: {run_a_id}", f"B: {run_b_id}", key=f"{stage}_usr_{idx}")

                st.markdown("**Output at this stage**")
                if stage == "step3_rewrite":
                    # The cross-run diff of the finals is the Response section
                    # above; here show what each run's rewrite DID to its draft.
                    st.caption("Cross-run diff of the finals is the **Response** section at the top; "
                               "below is each run's own draft → rewritten diff.")
                    for side, side_lin, rid in (("A", lin_a, run_a_id), ("B", lin_b, run_b_id)):
                        aud = side_lin.get("rewrite") or {}
                        if not (aud.get("draft_response") and aud.get("rewritten_response")):
                            st.caption(f"{side}: not reached")
                            continue
                        if st.toggle(f"{side} ({rid}): draft → rewritten", value=False,
                                     key=f"s3_inner_{side}_{idx}"):
                            common.show_diff(aud["draft_response"], aud["rewritten_response"],
                                             "2b draft", "step-3 rewritten",
                                             key=f"s3_innerdiff_{side}_{idx}")
                else:
                    (out_a, note_a), (out_b, note_b) = _stage_output(stage, lin_a), _stage_output(stage, lin_b)
                    for side, note in (("A", note_a), ("B", note_b)):
                        if note:
                            st.caption(f"{side}: {note}")
                    if out_a is None and out_b is None:
                        st.caption("not reached in either run")
                    else:
                        common.show_diff(out_a or "", out_b or "",
                                         f"A: {run_a_id}", f"B: {run_b_id}", key=f"{stage}_out_{idx}")

    if only_a or only_b:
        with st.expander(f"Unmatched — only in A ({len(only_a)}) / only in B ({len(only_b)})"):
            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"**Only in A: {run_a_id}**")
                for ex in sorted(only_a, key=lambda e: e.goal.lower()):
                    st.caption(f"`{ex.prompt_gid or ex.prompt_id}` — {ex.goal}")
            with cb:
                st.markdown(f"**Only in B: {run_b_id}**")
                for ex in sorted(only_b, key=lambda e: e.goal.lower()):
                    st.caption(f"`{ex.prompt_gid or ex.prompt_id}` — {ex.goal}")

# --- Template-level differences (bottom): the authored prompt/constitution FILES,
# before any per-example substitution. Distinct from per-example "Inputs by stage"
# above (which shows the rendered prompt for one case). Collapsed by default. ---
st.divider()
st.header("Template differences")
templates_a = {t.name: t for t in rendering.list_templates(run_a.run_dir, run_a.git_commit, pipeline)}
templates_b = {t.name: t for t in rendering.list_templates(run_b.run_dir, run_b.git_commit, pipeline)}
changed = []
for name in sorted(set(templates_a) | set(templates_b)):
    ta, tb = templates_a.get(name), templates_b.get(name)
    text_a = ta.text if ta else None
    text_b = tb.text if tb else None
    if text_a != text_b:
        changed.append((name, text_a, text_b))

st.caption("What changed in the prompt/constitution *files* between the two runs — run-level, "
           "before any per-example substitution. The filled-in prompts for a specific example "
           "are under that example's **Stage by stage** section above.")
if not changed:
    st.success("No prompt or constitution differences — output differences come from sampling alone.")
elif st.toggle(f"Show {len(changed)} changed template(s)", value=False):
    for name, text_a, text_b in changed:
        label = name + (" (only in B)" if text_a is None else " (only in A)" if text_b is None else "")
        with st.expander(label, expanded=False):
            diff = "\n".join(difflib.unified_diff(
                (text_a or "").splitlines(), (text_b or "").splitlines(),
                fromfile=f"A/{name}", tofile=f"B/{name}", lineterm="",
            ))
            st.code(diff, language="diff", wrap_lines=True)
