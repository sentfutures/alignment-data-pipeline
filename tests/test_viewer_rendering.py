"""Viewer prompt reconstruction: the system/user split is honored.

rendering.py is streamlit-free, so _format_split is testable directly. It must
mirror shared.utils.load_split_prompt: cut on the ===USER=== marker, or treat a
marker-less template as user-only (so pre-split run snapshots reconstruct as
they actually ran)."""

from viewer import rendering


def _mk(text):
    tpl = rendering.Template("t.txt", text, "snapshot")
    r = rendering.RenderedPrompt(stage="x", is_llm_call=True)
    return tpl, r


def test_format_split_cuts_on_marker_and_formats_each_half():
    tpl, r = _mk("SYS {a}\n===USER===\nUSR {b}")
    system, user = rendering._format_split(tpl, {"a": "A", "b": "B"}, r)
    assert system == "SYS A"
    assert user == "USR B"


def test_format_split_no_marker_is_user_only():
    tpl, r = _mk("just the user prompt {a}")
    system, user = rendering._format_split(tpl, {"a": "A"}, r)
    assert system is None
    assert user == "just the user prompt A"


def test_format_split_missing_template_returns_none():
    tpl = rendering.Template("t.txt", None, "missing")
    r = rendering.RenderedPrompt(stage="x", is_llm_call=True)
    assert rendering._format_split(tpl, {}, r) == (None, None)
    assert r.warnings  # unavailable-template warning recorded


class TestInlineWordDiff:
    def test_additions_highlighted_and_equal_text_plain(self):
        html = rendering.inline_word_diff_html(
            "Keep the shed clean.", "Keep the shed clean and reduce insect harm.")
        # the unchanged prefix stays plain (outside any span)
        assert html.startswith("Keep the shed ")
        # the added words are wrapped in a highlight span; the prefix is not
        assert "background:rgba" in html
        highlighted = html.split("background:rgba", 1)[1]
        assert "reduce insect harm." in highlighted
        assert "Keep the" not in highlighted

    def test_removed_words_struck_through(self):
        html = rendering.inline_word_diff_html("an obviously wrong claim", "an claim")
        assert "line-through" in html
        struck = html.split("line-through", 1)[1]
        assert "obviously wrong" in struck

    def test_text_is_escaped_and_newlines_become_breaks(self):
        html = rendering.inline_word_diff_html("a <b> start", "a <b> start\n\nnew para")
        assert "&lt;b&gt;" in html and "<b>" not in html.replace("<br>", "")
        assert "<br><br>" in html


class TestAuditSectionTable:
    def test_verdicts_get_color_badges(self):
        sec = {"title": "T", "rows": [
            {"label": "a", "value": "1", "verdict": "GOOD", "note": ""},
            {"label": "b", "value": "2", "verdict": "BAD", "note": "look here"},
            {"label": "c", "value": "3", "verdict": None, "note": ""},
        ]}
        rows = rendering.audit_section_table(sec)
        assert rows[0]["verdict"] == "🟢 GOOD"
        assert rows[1]["verdict"] == "🔴 BAD"
        assert rows[1]["note"] == "look here"
        assert rows[2]["verdict"] == ""  # informational row keeps the column blank

    def test_columns_omitted_when_section_has_no_verdicts_or_notes(self):
        sec = {"rows": [{"label": "a", "value": "1", "verdict": None, "note": ""}]}
        assert rendering.audit_section_table(sec) == [{"check": "a", "value": "1"}]

    def test_empty_section_is_empty(self):
        assert rendering.audit_section_table({}) == []

    def test_arm_comparison_section_splits_into_pipeline_plain_columns(self):
        # a genuine pipeline-vs-plain section: value strings split into columns
        sec = {"rows": [
            {"label": "distinct shapes", "value": "pipeline 12/40 / plain 16/40",
             "verdict": None, "note": ""},
            {"label": "Self-BLEU", "value": "pipeline 0.71 · plain 0.55",
             "verdict": None, "note": ""},  # the lexical section's '·' separator
            {"label": "top shape share (pipeline)", "value": "28%",
             "verdict": "GOOD", "note": "(10+ paras)"},  # pipeline-only single row
        ]}
        rows = rendering.audit_section_table(sec)
        assert rows[0] == {"check": "distinct shapes", "pipeline": "12/40",
                           "plain": "16/40", "verdict": "", "note": ""}
        assert rows[1]["pipeline"] == "0.71" and rows[1]["plain"] == "0.55"
        # the single-value pipeline-only row lands in the pipeline column
        assert rows[2]["pipeline"] == "28%" and rows[2]["plain"] == ""
        assert rows[2]["verdict"] == "🟢 GOOD"

    def test_single_arm_section_keeps_one_value_column(self):
        # response-openings-style: no plain arm → no split, keep 'value'
        sec = {"rows": [
            {"label": "responses scanned", "value": "40", "verdict": None, "note": ""},
            {"label": "families", "value": "other 34, heres-the-x 1", "verdict": None, "note": ""},
        ]}
        rows = rendering.audit_section_table(sec)
        assert "pipeline" not in rows[0] and rows[0]["value"] == "40"

    def test_plain_only_single_row_lands_in_plain_column(self):
        sec = {"rows": [
            {"label": "check-back additions", "value": "pipeline 61 / plain 57",
             "verdict": None, "note": ""},
            {"label": "plain-baseline median chars", "value": "812",
             "verdict": None, "note": ""},
        ]}
        rows = rendering.audit_section_table(sec)
        assert rows[1]["plain"] == "812" and rows[1]["pipeline"] == ""


class TestSplitArmValue:
    def test_slash_separator(self):
        assert rendering._split_arm_value("pipeline 12/40 / plain 16/40") == ("12/40", "16/40")

    def test_middot_separator(self):
        assert rendering._split_arm_value("pipeline 0.71 · plain 0.55") == ("0.71", "0.55")

    def test_pipeline_only_when_no_baseline(self):
        assert rendering._split_arm_value("pipeline 40") == ("40", "")

    def test_non_arm_value_returns_none(self):
        assert rendering._split_arm_value("28%") is None
        assert rendering._split_arm_value("") is None


class TestAuditShapeChartRows:
    def test_long_form_rows_per_shape_and_arm(self):
        structure = {"pipeline": {"shapes": {"3-5 paras": 30, "1-2 paras": 10}},
                     "plain": {"shapes": {"3-5 paras": 25}}}
        rows = rendering.audit_shape_chart_rows(structure)
        assert {"shape": "3-5 paras", "arm": "pipeline", "count": 30} in rows
        assert {"shape": "3-5 paras", "arm": "plain Claude", "count": 25} in rows
        assert len(rows) == 3

    def test_empty_structure_is_empty(self):
        assert rendering.audit_shape_chart_rows({}) == []


class TestAuditTrackedTicRows:
    def test_watch_counts_sorted_by_pipeline_count(self):
        tt = {"n_pipeline": 40, "n_plain": 40,
              "watch": {"i want to be": {"origin": "pipeline-origin", "pipeline": 8, "plain": 3},
                        "here's the thing": {"origin": "plain-origin", "pipeline": 0, "plain": 5},
                        "never appears": {"origin": "pipeline-origin", "pipeline": 0, "plain": 0}}}
        rows = rendering.audit_tracked_tic_rows(tt)
        # phrases that never appear in either arm are dropped
        assert all(r["phrase"] != "never appears" for r in rows)
        # sorted by pipeline count desc: 'i want to be' (8) first
        assert rows[0]["phrase"] == "i want to be"

    def test_empty_is_empty(self):
        assert rendering.audit_tracked_tic_rows({}) == []


class TestAuditDeliveryPareto:
    def test_rows_join_considerations_and_delivery_per_arm(self):
        delivery_pc = {"AW-0001": {"pipeline": {"score": 8, "note": "clean"},
                                   "plain": {"score": 4, "note": "lectures"}}}
        reasons_pc = {"AW-0001": {"pipeline": {"reasons": ["a", "b", "c"]},
                                  "plain": {"reasons": ["a"]}}}
        rows = rendering.audit_delivery_pareto_rows(delivery_pc, reasons_pc)
        # one dot per arm: x = considerations count, y = delivery score
        by_arm = {r["arm"]: r for r in rows}
        assert by_arm["pipeline"]["considerations"] == 3
        assert by_arm["pipeline"]["delivery"] == 8
        assert by_arm["pipeline"]["note"] == "clean"
        assert by_arm["plain Claude"]["considerations"] == 1
        assert by_arm["plain Claude"]["delivery"] == 4

    def test_arm_missing_either_axis_is_skipped(self):
        # pipeline has a delivery score but no extracted considerations -> skipped
        delivery_pc = {"AW-0001": {"pipeline": {"score": 7, "note": ""}}}
        reasons_pc = {"AW-0001": {"pipeline": {"reasons": None}}}
        assert rendering.audit_delivery_pareto_rows(delivery_pc, reasons_pc) == []


class TestAuditLibraryAndAlternativeRows:
    def test_alternative_chart_rows_counts_from_anchored_and_added(self):
        mv = {"AW-0002": {"alternatives": {"anchored": [{"alternative": "a", "verdict": "kept"}],
                                           "added": ["b"]}},
              "AW-0001": {"alternatives": {"anchored": [], "added": ["x"]}}}
        rows = rendering.audit_alternative_chart_rows(mv)
        # plain = all anchored; pipeline = kept/weakened + added
        assert rows[0] == {"record": "AW-0001", "plain Claude": 0, "pipeline": 1}
        assert rows[1] == {"record": "AW-0002", "plain Claude": 1, "pipeline": 2}

    def test_alternative_groups_buckets_kept_weakened_dropped_added(self):
        alt = {"anchored": [{"alternative": "use farmed", "verdict": "kept"},
                            {"alternative": "vague plan", "verdict": "weakened"},
                            {"alternative": "ask vet", "verdict": "dropped"}],
               "added": ["reword the placard"]}
        groups = rendering.audit_alternative_groups(alt)
        assert [g[0].split(" (")[0] for g in groups] == [
            "✓ Kept by the pipeline", "〜 Weakened", "✗ Dropped", "➕ Added by the pipeline"]
        assert groups[0][1] == ["use farmed"] and groups[3][1] == ["reword the placard"]
        assert rendering.audit_alternative_groups({}) is None

    def test_trigger_count_rows_dedup_library_order_and_zeros(self):
        pulls = {"p1": ["C1", "C1", "C2"], "p2": ["C1"]}  # per-case dedup
        rows = rendering.audit_trigger_count_rows(pulls, ["C1", "C2", "C3"], {"C1": "move1"})
        assert rows == [{"entry": "C1", "cases": 2, "move": "move1"},
                        {"entry": "C2", "cases": 1, "move": ""},
                        {"entry": "C3", "cases": 0, "move": ""}]  # never-pulled stays, count 0

    def test_pull_scatter_rows_needs_both_survival_and_pull(self):
        per_case = {"p1": {"survival": {"added": ["r1", "r2"]}},
                    "p2": {"survival": {"added": []}},
                    "p3": {}}  # no survival -> excluded
        pulls = {"p1": ["C1", "C2", "C3"], "p2": ["C1"]}
        rows = rendering.audit_pull_scatter_rows(per_case, pulls)
        assert {r["record"] for r in rows} == {"p1", "p2"}
        r1 = next(r for r in rows if r["record"] == "p1")
        assert r1["pulled"] == 3 and r1["added"] == 2

    def test_pull_count_rows(self):
        assert rendering.audit_pull_count_rows({"p1": ["C1", "C2"]}) == \
            [{"record": "p1", "count": 2, "entries": "C1, C2"}]


class TestAuditChartRows:
    def test_length_rows_wide_form_keeps_missing_plain_as_none(self):
        per_case = {"AW-0002": {"pipeline": 500, "plain": 200},
                    "AW-0001": {"pipeline": 300, "plain": None}}
        rows = rendering.audit_length_chart_rows(per_case)
        # sorted by record; one row per record, one column per arm (colors are
        # pinned by column order — AUDIT_ARM_COLUMNS/AUDIT_ARM_COLORS)
        assert rows == [
            {"record": "AW-0001", "plain Claude": None, "pipeline": 300},
            {"record": "AW-0002", "plain Claude": 200, "pipeline": 500},
        ]

    def test_reason_rows_count_unique_reasons_per_arm(self):
        per_case = {"AW-0001": {
            "plain": {"reasons": ["a"], "chars": 100},
            "pipeline": {"reasons": ["a", "b", "c"], "chars": 400},
        }}
        rows = rendering.audit_reason_chart_rows(per_case)
        assert rows == [{"record": "AW-0001", "plain Claude": 1, "pipeline": 3}]

    def test_arm_columns_and_colors_stay_paired(self):
        assert len(rendering.AUDIT_ARM_COLUMNS) == len(rendering.AUDIT_ARM_COLORS)
        assert rendering.AUDIT_ARM_COLUMNS[0] == "plain Claude"

    def test_pull_count_rows_carry_counts_and_joined_entry_ids(self):
        pulls = {"AW-0002": ["C1", "M5", "T3"], "AW-0001": ["C3"]}
        rows = rendering.audit_pull_count_rows(pulls, {"AW-0001": "E-0042"})
        # sorted by prompt id, labeled like the other per-record chart rows
        assert rows == [
            {"record": "E-0042", "count": 1, "entries": "C3"},
            {"record": "AW-0002", "count": 3, "entries": "C1, M5, T3"},
        ]

    def test_pull_scatter_pairs_pull_width_with_added_reasons(self):
        per_case = {
            "AW-0001": {"survival": {"anchored": [], "added": ["n1", "n2"]}},
            "AW-0002": {"survival": {"anchored": [], "added": []}},
            "AW-0003": {"survival": {"anchored": [], "added": ["n3"]}},  # no pull record
            "AW-0004": {},  # no survival judgment
        }
        pulls = {"AW-0001": ["C1", "M5", "T3"], "AW-0002": ["C3"], "AW-0004": ["C1"]}
        rows = rendering.audit_pull_scatter_rows(per_case, pulls, {"AW-0001": "E-0042"})
        # only records with BOTH a survival judgment and a pull record plot
        assert rows == [
            {"record": "E-0042", "pulled": 3, "added": 2, "entries": "C1, M5, T3"},
            {"record": "AW-0002", "pulled": 1, "added": 0, "entries": "C3"},
        ]

    def test_trigger_counts_dedupe_per_case_and_keep_zero_entries(self):
        pulls = {"AW-0001": ["C1", "C1", "T3"], "AW-0002": ["C1"]}
        rows = rendering.audit_trigger_count_rows(
            pulls, ["C1", "M5", "T3"], {"C1": "move one", "M5": "move five"})
        # library order, per-case dedup (C1 fired in 2 CASES, not 3 pulls),
        # never-pulled entries stay in at zero
        assert rows == [
            {"entry": "C1", "cases": 2, "move": "move one"},
            {"entry": "M5", "cases": 0, "move": "move five"},
            {"entry": "T3", "cases": 1, "move": ""},
        ]

    def test_labels_map_records_to_example_gids_with_fallback(self):
        # per_case stays keyed by prompt_id; labels swap in the stable example
        # gid for display, and unmapped ids (pre-gid runs) pass through
        labels = {"AW-0001": "E-0042"}
        per_case = {"AW-0001": {"pipeline": 300, "plain": None},
                    "AW-0002": {"pipeline": 500, "plain": 200}}
        rows = rendering.audit_length_chart_rows(per_case, labels)
        assert [r["record"] for r in rows] == ["E-0042", "AW-0002"]
        reason_case = {"AW-0001": {"plain": {"reasons": ["a"]}, "pipeline": None}}
        assert rendering.audit_reason_chart_rows(reason_case, labels)[0]["record"] == "E-0042"
        surv_case = {"AW-0001": {"survival": {"anchored": [
            {"reason": "a", "verdict": "kept"}], "added": []}}}
        assert rendering.audit_survival_chart_rows(surv_case, labels)[0]["record"] == "E-0042"
        assert rendering.audit_record_label("AW-0009", labels) == "AW-0009"
        assert rendering.audit_record_label("AW-0009", None) == "AW-0009"


class TestAuditSectionMeta:
    # Every section title the current audit emits must resolve through the
    # title fallback (old reports carry no group/gloss fields).
    CURRENT_TITLES = [
        "Structural skeletons", "Openers & closers",
        "Unrealized dealt details (frontier frame)", "Locale / taxa plausibility",
        "Length-class realization", "Insider-vocabulary leak (responses)",
        "Response lengths (vs plain baseline)", "Stock phrases (responses)",
        "Lexical diversity (responses)", "Structural variation (responses)",
        "Response openings (drafts)", "Response openings (finals)",
        "Reasoning-library selection (2a.5)", "Reasoning-library coverage",
        "Welfare reasoning (LLM)",
    ]

    def test_field_wins_over_title_fallback(self):
        sec = {"title": "Response lengths (vs plain baseline)",
               "group": "paid", "gloss": "custom"}
        assert rendering.audit_section_group(sec) == "paid"
        assert rendering.audit_section_gloss(sec) == "custom"

    def test_title_fallback_covers_every_current_section(self):
        for title in self.CURRENT_TITLES:
            sec = {"title": title}
            assert rendering.audit_section_group(sec) in rendering.AUDIT_GROUP_ORDER[:-1], title
            assert rendering.audit_section_gloss(sec), title

    def test_unknown_sections_degrade_to_other(self):
        sec = {"title": "Some future check"}
        assert rendering.audit_section_group(sec) == "other"
        assert rendering.audit_section_gloss(sec) == ""


class TestAuditVerdictSummary:
    def test_worst_verdict_counts_and_report_order(self):
        report = {"sections": [
            {"title": "A", "group": "prompt", "rows": [
                {"verdict": "GOOD"}, {"verdict": "OK"}, {"verdict": None}]},
            {"title": "B", "group": "response", "rows": [
                {"verdict": "GOOD"}, {"verdict": "BAD"}, {"verdict": "OK"}]},
            {"title": "C", "group": "prompt", "rows": [{"verdict": None}]},
        ]}
        rows = rendering.audit_verdict_summary(report)
        assert [r["section"] for r in rows] == ["A", "B", "C"]
        assert rows[0]["worst"] == "OK" and rows[0]["counts"] == {"GOOD": 1, "OK": 1, "BAD": 0}
        assert rows[1]["worst"] == "BAD"
        assert rows[2]["worst"] is None  # purely informational section

    def test_skipped_sections_are_flagged(self):
        report = {"sections": [{"title": "A", "rows": [{"verdict": None}]}],
                  "skipped_sections": [{"section": "A", "reason": "bare-file input"}]}
        rows = rendering.audit_verdict_summary(report)
        assert rows[0]["skipped"] is True

    def test_empty_report_gives_empty_summary(self):
        assert rendering.audit_verdict_summary({}) == []


class TestAuditSurvivalGroups:
    def test_reasons_bucketed_by_verdict_plus_added(self):
        case = {
            "plain": {"reasons": ["a", "b", "c"]},
            "pipeline": {"reasons": ["x", "y"]},
            "survival": {"anchored": [
                {"reason": "a", "verdict": "kept"},
                {"reason": "b", "verdict": "weakened"},
                {"reason": "c", "verdict": "dropped"},
            ], "added": ["y"]},
        }
        groups = rendering.audit_survival_groups(case)
        assert [(t.split(" (")[0], rs) for t, rs in groups] == [
            ("✓ Kept by the pipeline", ["a"]),
            ("〜 Weakened", ["b"]),
            ("✗ Dropped", ["c"]),
            ("➕ Added by the pipeline", ["y"]),
        ]
        # counts live in the titles
        assert groups[0][0].endswith("(1)") and groups[3][0].endswith("(1)")

    def test_no_survival_data_returns_none_for_fallback(self):
        assert rendering.audit_survival_groups({"plain": {"reasons": ["a"]}}) is None


class TestAuditBatchTotals:
    def test_totals_with_absolute_and_percent_deltas(self):
        report = {
            "response_lengths": {"per_case": {
                "AW-0001": {"pipeline": 400, "plain": 200},
                "AW-0002": {"pipeline": 600, "plain": 300},
                "AW-0003": {"pipeline": 999, "plain": None},  # unpaired: excluded
            }},
            "moral_patient_reasons": {"per_case": {
                "AW-0001": {"pipeline": {"reasons": ["a", "b", "c"]},
                            "plain": {"reasons": ["a", "b"]}},
            }},
        }
        rows = rendering.audit_batch_totals(report)
        assert rows == [
            {"metric": "total characters", "plain Claude": "500", "pipeline": "1,000",
             "Δ absolute": "+500", "Δ %": "+100.0%"},
            {"metric": "total considerations", "plain Claude": "2", "pipeline": "3",
             "Δ absolute": "+1", "Δ %": "+50.0%"},
        ]

    def test_empty_report_gives_no_rows(self):
        assert rendering.audit_batch_totals({}) == []


class TestAuditSurvivalChartRows:
    def test_rows_bucket_counts_and_carry_reason_texts(self):
        per_case = {"AW-0001": {
            "survival": {"anchored": [
                {"reason": "a", "verdict": "kept"},
                {"reason": "b", "verdict": "kept"},
                {"reason": "c", "verdict": "dropped"},
            ], "added": ["n1"]},
        }, "AW-0002": {}}  # no survival -> no rows
        rows = rendering.audit_survival_chart_rows(per_case)
        assert rows == [
            {"record": "AW-0001", "category": "✓ kept", "stack_order": 0,
             "count": 2, "reasons": "a • b"},
            {"record": "AW-0001", "category": "✗ dropped", "stack_order": 2,
             "count": 1, "reasons": "c"},
            {"record": "AW-0001", "category": "➕ added", "stack_order": 3,
             "count": 1, "reasons": "n1"},
        ]

    def test_no_survival_anywhere_gives_empty(self):
        assert rendering.audit_survival_chart_rows({"AW-0001": {"plain": {}}}) == []
        assert len(rendering.AUDIT_SURVIVAL_CATEGORIES) == len(rendering.AUDIT_SURVIVAL_COLORS)


class TestComposedGateRefineRendering:
    """Composed 1c gate + 1d refine runs: BOTH stages must be renderable —
    the gate must never short-circuit the refine view (the stage split's whole
    point is that both calls are real, paid, and reviewable)."""

    @staticmethod
    def _composed_run(tmp_path):
        import random
        import shutil

        from dad_pipeline import compose_scenarios as cs

        run = tmp_path / "run"
        (run / "inputs" / "prompts").mkdir(parents=True)
        for name in ("step1c_gate.txt", "step1d_refine.txt"):
            shutil.copy(rendering.REPO_ROOT / "prompts" / "dad" / name
                        if hasattr(rendering, "REPO_ROOT")
                        else f"prompts/dad/{name}",
                        run / "inputs" / "prompts" / name)
        scenario = cs.deal_scenarios(1, random.Random(3))[0]
        scenario["scenario_description"] = "A designed situation."
        lineage = {
            "scenario": scenario,
            "gate": {"passed": True, "failures": [], "attempt": 1},
            "dilemma": {
                "scenario_id": scenario["scenario_id"],
                "user_message": "Refined final text.",
                "draft_user_message": "Judged draft text.",
                "annotation": {"visibility": "explicit", "leverage": "their personal choices"},
            },
        }
        manifest = {"manifest_version": 2, "git_commit": None}
        return run, manifest, lineage

    def test_gate_stage_renders_the_judged_draft(self, tmp_path):
        run, manifest, lineage = self._composed_run(tmp_path)
        r = rendering.render_prompt("dad", "step1_gate", run, manifest, lineage)
        assert r.is_llm_call
        # the gate judged the PRE-refine draft, not the shipped rewrite
        assert "Judged draft text." in (r.user or "")
        assert any("PASS" in w for w in r.warnings)

    def test_refine_stage_renders_even_when_the_gate_ran(self, tmp_path):
        # regression: the old single-stage view short-circuited to the gate
        # whenever gate.jsonl had a record, hiding the refine call entirely
        run, manifest, lineage = self._composed_run(tmp_path)
        r = rendering.render_prompt("dad", "step1_refine", run, manifest, lineage)
        assert r.is_llm_call
        assert "Judged draft text." in (r.user or "")     # the draft under review
        assert "<draft_prompt>" in (r.user or "")

    def test_gate_stage_marks_not_run_on_pre_gate_lineage(self, tmp_path):
        run, manifest, lineage = self._composed_run(tmp_path)
        lineage["gate"] = None
        r = rendering.render_prompt("dad", "step1_gate", run, manifest, lineage)
        assert not r.is_llm_call
        assert any("did not use the 1c gate" in w for w in r.warnings)
