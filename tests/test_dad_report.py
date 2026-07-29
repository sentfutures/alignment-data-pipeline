"""Tests for report/build_report.py — the standalone DAD story report.

Three things carry real risk here and get most of the coverage:

  * **Degradation.** No committed run has the paid delivery/showcase keys, so the
    generator must render a complete report from a partial audit and say what is
    missing rather than quietly omitting it.
  * **Self-containment.** The artefact's whole format exists so it can be opened
    offline from a filesystem. One external asset reference breaks that.
  * **Candour.** The weaknesses section is derived from the data, not written, so
    the failing checks are asserted to survive into the HTML.

Fully offline — the generator touches no network and no API, so no stubs beyond
the suite's autouse guards are needed.
"""

import json
import re

import pytest

from report import build_report as B
from report import render as R

# --- fixtures, shaped like the real audit JSON --------------------------------

PER_CASE = {
    "AW-0001": {
        "pipeline": {"reasons": ["a", "b", "c"], "chars": 4000,
                     "type_hist": {"direct": 2, "sentience": 1}},
        "plain": {"reasons": ["a", "b"], "chars": 2500, "type_hist": {"direct": 2}},
        "survival": {"anchored": [{"reason": "a", "verdict": "kept"},
                                  {"reason": "b", "verdict": "dropped"}],
                     "added": ["c"]},
        "response_gid": "R-0201", "example_gid": "E-0172",
    },
    "AW-0002": {
        "pipeline": {"reasons": ["d", "e"], "chars": 3800, "type_hist": {"direct": 2}},
        "plain": {"reasons": ["d"], "chars": 2400, "type_hist": {"direct": 1}},
        "survival": {"anchored": [{"reason": "d", "verdict": "weakened"}], "added": ["e"]},
        "response_gid": "R-0202", "example_gid": "E-0173",
    },
}

AUDIT_FULL = {
    "n_prompts": 2,
    "gid_map": {"AW-0001": {"response": "R-0201", "example": "E-0172"},
                "AW-0002": {"response": "R-0202", "example": "E-0173"}},
    "sections": [
        {"title": "Response stance (LLM)", "group": "paid",
         "rows": [{"label": "moralizes", "value": "pipeline 40% / plain 0%",
                   "verdict": "BAD", "note": "(fault — lower is better)"},
                  {"label": "defers", "value": "100%", "verdict": "GOOD", "note": ""}]},
        {"title": "Locale / taxa plausibility", "group": "prompt",
         "rows": [{"label": "implausible", "value": "0", "verdict": "GOOD", "note": ""}]},
    ],
    "moral_patient_reasons": {
        "n": 2, "failures": 1, "model": "claude-sonnet-5",
        "pipeline": {"n": 2, "mean_unique": 2.5},
        "plain": {"n": 2, "mean_unique": 1.5},
        "survival": {"kept": 1, "weakened": 1, "dropped": 1, "added_total": 2},
        "per_case": PER_CASE,
    },
    "moves": {
        "alternatives": {"pipeline_mean": 3.0, "plain_mean": 2.0},
        "stance": {"pipeline": {"defers": 1.0, "calibrated": 0.97, "moralizes": 0.4},
                   "plain": {"defers": 1.0, "calibrated": 1.0, "moralizes": 0.0}},
    },
    "delivery": {
        "model": "claude-opus-5", "pipeline_mean": 0.82, "plain_mean": 0.79,
        "dimensions": {"pipeline": {"tone": 0.8, "calibration": 0.9},
                       "plain": {"tone": 0.85, "calibration": 0.8}},
        "per_case": {"AW-0001": {"pipeline": {"score": 8}, "plain": {"score": 7}},
                     "AW-0002": {"pipeline": {"score": 9}, "plain": {"score": 8}}},
    },
    "showcase": {
        "model": "claude-opus-5",
        "examples": [{"prompt_id": "AW-0001", "label": "Welfare reasoning added",
                      "summary": "The pipeline **surfaced** a point plain missed.",
                      "user_message": "Should I do the thing?",
                      "plain_response": "Maybe.", "pipeline_response": "Consider the animals here.",
                      "highlights": ["the animals"], "fit": 9}],
    },
    "response_lengths": {"pipeline_mean": 4659.0, "plain_mean": 2988.0, "mean_ratio": 1.56,
                         "per_case": {}},
    "tracked_tics": {"n_pipeline": 2, "n_plain": 2,
                     "watch": {"cuts both ways": {"origin": "pipeline-origin",
                                                  "pipeline": 1, "plain": 0}}},
    "rhetorical_moves": {"moves": {"unbundling": {"description": "splits a bundled choice",
                                                  "pipeline_share": 0.28, "plain_share": 0.28}}},
    "structure": {"pipeline": {"effective_shapes": 9.44},
                  "plain": {"effective_shapes": 13.88}},
}

DIVERSITY = {"n_records": 2, "embed_model": "gemini-embedding-001",
             "vendi": {"score": 5.15, "ratio": 0.132},
             "nn": {"over_0.90": 0.0, "over_0.80": 0.33},
             "scopes": {"combined": {"clusters": {"evenness": 0.875, "largest_share": 0.33}}}}

MANIFEST = {"run_id": "2026-07-20_20-51_bedrock-40", "git_commit": "abc12345", "git_dirty": True,
            "config": {"backend": "bedrock", "model": "claude-sonnet-5",
                       "dad": {"scenario_model": "claude-opus-4-8",
                               "constitution_rewrite_model": "claude-opus-4-8"}}}

COSTS = [{"stage": "prompt_draft", "cost_usd": 0.5, "model": "claude-opus-4-8"},
         {"stage": "constitution_rewrite", "cost_usd": 1.5, "model": "claude-opus-4-8"}]

BASELINE = [{"prompt_id": "AW-0001", "user_message": "Should I do the thing?",
             "baseline_response": "Maybe."}]
REWRITES = [{"prompt_id": "AW-0001", "user_message": "Should I do the thing?",
             "draft_response": "Consider the animals.",
             "rewritten_response": "Consider the animals here."}]

CONTENT = {k: f"Prose for {k}." for k in B.CONTENT_IDS}
CONTENT["title"] = "Test report"
CONTENT["subtitle"] = "A {{n}}-example run."
CONTENT["example_pick"] = "auto"


def content(**overrides):
    return {**CONTENT, **overrides}


def strip_tags(html):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))


def make_run_dir(tmp_path, audit=None, diversity=DIVERSITY, manifest=MANIFEST,
                 costs=COSTS, content_text=None):
    run_dir = tmp_path / "runs" / "2026-07-20_20-51_bedrock-40"
    (run_dir / "audit").mkdir(parents=True)
    (run_dir / "final").mkdir()
    (run_dir / "baseline").mkdir()
    (run_dir / "step3").mkdir()
    (run_dir / "audit" / "audit_report.json").write_text(
        json.dumps(audit if audit is not None else AUDIT_FULL), encoding="utf-8")
    if diversity is not None:
        (run_dir / "audit" / "diversity_report.json").write_text(json.dumps(diversity),
                                                                 encoding="utf-8")
    if manifest is not None:
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if costs is not None:
        (run_dir / "cost_log.jsonl").write_text(
            "\n".join(json.dumps(c) for c in costs), encoding="utf-8")
    (run_dir / "baseline" / "baseline_responses.jsonl").write_text(
        "\n".join(json.dumps(r) for r in BASELINE), encoding="utf-8")
    (run_dir / "step3" / "rewrites.jsonl").write_text(
        "\n".join(json.dumps(r) for r in REWRITES), encoding="utf-8")
    (run_dir / "final" / "dad_corpus.jsonl").write_text(
        json.dumps({"record_id": "AW-0001", "messages": []}), encoding="utf-8")
    content_file = tmp_path / "content.md"
    content_file.write_text(
        content_text if content_text is not None
        else "".join(f"<!-- id: {k} -->\n{v}\n\n" for k, v in CONTENT.items()),
        encoding="utf-8")
    return run_dir, content_file


class TestParseContent:
    def test_round_trips_sections(self):
        text = "".join(f"<!-- id: {k} -->\n{k.upper()} body\n\n" for k in B.CONTENT_IDS)
        parsed = B.parse_content(text)
        assert parsed["title"] == "TITLE body"
        assert parsed["problem"] == "PROBLEM body"
        assert set(parsed) == set(B.CONTENT_IDS)

    def test_unknown_id_raises(self):
        text = "".join(f"<!-- id: {k} -->\nx\n\n" for k in B.CONTENT_IDS)
        with pytest.raises(ValueError, match="unknown section id"):
            B.parse_content(text + "<!-- id: nonsense -->\nx")

    def test_missing_id_raises(self):
        ids = [k for k in B.CONTENT_IDS if k != "problem"]
        with pytest.raises(ValueError, match="missing section id"):
            B.parse_content("".join(f"<!-- id: {k} -->\nx\n\n" for k in ids))

    def test_no_markers_raises(self):
        with pytest.raises(ValueError, match="no '<!-- id"):
            B.parse_content("just prose")


class TestFacts:
    def test_unknown_placeholder_is_a_build_error(self):
        with pytest.raises(KeyError, match="unknown fact"):
            B.fill("{{not_a_fact}}", {"n": 2})

    def test_reconstructs_considerations_from_legacy_schema(self):
        cons = B._considerations(AUDIT_FULL)
        assert cons["source"] == "reconstructed"
        assert cons["pipeline"] == pytest.approx(5.5)  # 2.5 reasoning + 3.0 alternatives
        assert cons["plain"] == pytest.approx(3.5)

    def test_prefers_modern_schema_when_present(self):
        audit = dict(AUDIT_FULL, valuable_welfare_considerations={
            "available": True, "parent": {"pipeline": 9.0, "plain": 6.0},
            "subsets": [{"name": "welfare reasoning", "pipeline": 5.0, "plain": 4.0}]})
        cons = B._considerations(audit)
        assert cons["source"] == "modern"
        assert cons["pipeline"] == 9.0

    def test_facts_are_read_from_the_data_not_hardcoded(self):
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["response_lengths"]["mean_ratio"] = 2.5
        assert B.facts(audit)["length_pct"] == "150%"


class TestBuildReport:
    def test_builds_every_section(self):
        html = B.build_report(audit=AUDIT_FULL, content=content(), diversity=DIVERSITY,
                              manifest=MANIFEST, costs=COSTS, baseline=BASELINE,
                              rewrites=REWRITES, run_id="run-x")
        for sid, _ in B.TOC:
            assert f"id='{sid}'" in html

    def test_is_self_contained(self):
        """No external CSS, JS, fonts or images — the file must open offline."""
        html = B.build_report(audit=AUDIT_FULL, content=content(), diversity=DIVERSITY,
                              manifest=MANIFEST)
        assert not re.search(r"<(img|link|iframe)\b", html)
        assert not re.search(r"<script[^>]*\ssrc=", html)
        assert "@import" not in html and "url(" not in html

    def test_prose_hyperlinks_are_allowed(self):
        html = B.build_report(audit=AUDIT_FULL,
                              content=content(problem="See [the post](https://example.com/x)."))
        assert "href='https://example.com/x'" in html

    def test_both_dark_mode_declarations_survive(self):
        html = B.build_report(audit=AUDIT_FULL, content=content())
        assert "@media (prefers-color-scheme:dark)" in html
        assert "[data-theme=dark]" in html

    def test_placeholders_are_resolved(self):
        html = B.build_report(audit=AUDIT_FULL, content=content())
        assert "{{" not in html
        assert "A 2-example run." in html

    def test_escapes_hostile_corpus_text(self):
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["showcase"]["examples"][0]["pipeline_response"] = "<script>alert(1)</script>"
        html = B.build_report(audit=audit, content=content(), baseline=BASELINE)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


class TestDegradation:
    """No committed run has the paid keys, so partial input is the normal case."""

    def test_offline_only_audit_still_builds(self):
        audit = {k: v for k, v in AUDIT_FULL.items()
                 if k not in ("delivery", "showcase", "moves", "moral_patient_reasons")}
        html = B.build_report(audit=audit, content=content())
        assert "id='results'" in html
        assert "None" not in strip_tags(html)

    def test_missing_delivery_says_so_rather_than_omitting(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "delivery"}
        html = B.build_report(audit=audit, content=content())
        text = strip_tags(html)
        assert "not measured on this run" in text
        assert "--reasons" in text

    def test_delivery_present_renders_the_pareto(self):
        html = B.build_report(audit=AUDIT_FULL, content=content())
        assert "delivery quality (0-10)" in html

    def test_bare_audit_still_carries_the_narrative(self):
        html = B.build_report(audit={"n_prompts": 3}, content=content())
        assert "Prose for problem." in html
        assert "Prose for reproduce." in html

    def test_missing_manifest_diversity_and_costs(self):
        html = B.build_report(audit=AUDIT_FULL, content=content(), manifest=None,
                              diversity=None, costs=None)
        assert "id='summary'" in html

    def test_missing_gid_map_falls_back_to_prompt_ids(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "gid_map"}
        html = B.build_report(audit=audit, content=content())
        assert "AW-0001" in html


class TestWorkedExample:
    def test_showcase_example_is_used_and_highlighted(self):
        html = B.build_report(audit=AUDIT_FULL, content=content(), baseline=BASELINE)
        assert "Should I do the thing?" in html
        assert "<mark>the animals</mark>" in html
        assert "showcase judge" in strip_tags(html)

    def test_non_locating_highlight_leaves_text_whole(self):
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["showcase"]["examples"][0]["highlights"] = ["text that is not present"]
        html = B.build_report(audit=audit, content=content(), baseline=BASELINE)
        assert "Consider the animals here." in html
        assert "<mark>" not in html

    def test_pinned_example_overrides_the_showcase(self):
        html = B.build_report(audit=AUDIT_FULL, content=content(example_pick="AW-0001"),
                              baseline=BASELINE, rewrites=REWRITES)
        assert "pinned in the prose file" in strip_tags(html)

    def test_falls_back_to_the_most_added_record(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "showcase"}
        html = B.build_report(audit=audit, content=content(), baseline=BASELINE,
                              rewrites=REWRITES)
        assert "selected mechanically" in strip_tags(html)

    def test_example_text_comes_from_the_run_files(self):
        """The generator pulls response text off disk; nothing is retyped."""
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "showcase"}
        html = B.build_report(audit=audit, content=content(), baseline=BASELINE,
                              rewrites=REWRITES)
        assert "Consider the animals here." in html

    def test_rewrite_diff_renders_when_both_texts_exist(self):
        html = B.build_report(audit=AUDIT_FULL, content=content(example_pick="AW-0001"),
                              baseline=BASELINE, rewrites=REWRITES)
        assert "<ins>" in html

    def test_no_example_data_is_reported_not_crashed(self):
        html = B.build_report(audit={"n_prompts": 1}, content=content())
        assert "No worked example" in strip_tags(html)


class TestCandour:
    """The weaknesses floor is derived from the run, so it cannot be edited away."""

    def test_bad_verdicts_reach_the_report(self):
        html = B.build_report(audit=AUDIT_FULL, content=content(), manifest=MANIFEST)
        text = strip_tags(html)
        assert "Response stance" in text
        assert "BAD" in text

    def test_moralizing_regression_is_shown_in_both_arms(self):
        html = B.build_report(audit=AUDIT_FULL, content=content())
        text = strip_tags(html)
        assert "40%" in text and "0%" in text

    def test_non_faithful_backend_is_flagged(self):
        warnings = B.derived_warnings(AUDIT_FULL, MANIFEST, B.facts(AUDIT_FULL, MANIFEST))
        assert any("bedrock" in w for _, w in warnings)

    def test_api_backend_removes_that_warning(self):
        manifest = json.loads(json.dumps(MANIFEST))
        manifest["config"]["backend"] = "api"
        warnings = B.derived_warnings(AUDIT_FULL, manifest, B.facts(AUDIT_FULL, manifest))
        assert not any("faithful mode" in w for _, w in warnings)

    def test_dirty_git_tree_is_surfaced(self):
        warnings = B.derived_warnings(AUDIT_FULL, MANIFEST, B.facts(AUDIT_FULL, MANIFEST))
        assert any("dirty" in w for _, w in warnings)

    def test_extraction_failures_produce_an_asymmetry_note(self):
        html = B.build_report(audit=AUDIT_FULL, content=content())
        assert "not a fully matched comparison" in strip_tags(html)

    def test_delivery_regression_leads_the_weaknesses(self):
        """The substance/manner trade this method exists to avoid, going the
        wrong way, must surface as BAD and first — the bedrock-40 case."""
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["delivery"]["pipeline_mean"] = 0.70
        audit["delivery"]["plain_mean"] = 0.79
        warnings = B.derived_warnings(audit, MANIFEST, B.facts(audit, MANIFEST))
        severities = [sev for sev, _ in warnings]
        assert severities == sorted(severities, key=lambda s: s != "BAD")  # BADs first
        assert any(sev == "BAD" and "wrong way" in w for sev, w in warnings)
        html = B.build_report(audit=audit, content=content())
        assert "the pipeline is worse here" in html

    def test_delivery_gain_is_not_flagged(self):
        warnings = B.derived_warnings(AUDIT_FULL, MANIFEST, B.facts(AUDIT_FULL, MANIFEST))
        assert not any("wrong way" in w for _, w in warnings)

    def test_missing_delivery_is_a_derived_weakness(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "delivery"}
        warnings = B.derived_warnings(audit, MANIFEST, B.facts(audit, MANIFEST))
        assert any(sev == "BAD" and "showcase" in w for sev, w in warnings)

    def test_weaknesses_render_without_any_editorial_prose(self):
        html = B.build_report(audit=AUDIT_FULL, content=content(weaknesses_intro=""),
                              manifest=MANIFEST)
        section = html[html.find("id='weaknesses'"):html.find("id='checks'")]
        assert "BAD" in section

    def test_every_check_is_listed(self):
        html = B.build_report(audit=AUDIT_FULL, content=content())
        section = html[html.find("id='checks'"):]
        assert "Locale / taxa plausibility" in section


class TestRenderPrimitives:
    def test_charts_emit_parseable_svg(self):
        import xml.etree.ElementTree as ET

        ET.fromstring(R.hbar([("a", 1), ("b<script>", 2)]))
        ET.fromstring(R.grouped_hbar([{"label": "x", "p": 1, "q": 2}],
                                     series=[("p", "red"), ("q", "blue")]).split("<div")[0])
        ET.fromstring(R.stacked_bar([{"label": "r", "segments": {"kept": 2}}],
                                    categories=[("kept", "red")]).split("<div")[0])
        ET.fromstring(R.scatter([{"x": 1, "y": 2, "color": "red", "tip": "t"}]))

    def test_empty_data_is_a_note_not_a_broken_chart(self):
        assert "no" in R.hbar([]).lower()
        assert "<svg" not in R.grouped_hbar([], series=[("a", "red")])

    def test_zero_values_do_not_divide_by_zero(self):
        assert "<svg" in R.hbar([("a", 0), ("b", 0)])

    def test_table_escapes_cells_but_passes_raw_through(self):
        html = R.table(["h"], [("<b>x</b>",), (R.Raw("<b>y</b>"),)])
        assert "&lt;b&gt;x&lt;/b&gt;" in html
        assert "<b>y</b>" in html

    def test_inline_md_escapes_before_formatting(self):
        assert R.inline_md("**a** <b>") == "<b>a</b> &lt;b&gt;"

    def test_highlight_is_fail_open(self):
        assert R.highlight("hello", ["nope"]) == "<div class='resp'>hello</div>"
        assert "<mark>ell</mark>" in R.highlight("hello", ["ell"])


class TestMain:
    def test_writes_the_file(self, tmp_path, monkeypatch):
        run_dir, content_file = make_run_dir(tmp_path)
        out = tmp_path / "report.html"
        monkeypatch.setattr("sys.argv", ["build_report.py", "--run", str(run_dir),
                                         "--content", str(content_file), "--out", str(out)])
        B.main()
        assert out.exists()
        assert "id='summary'" in out.read_text(encoding="utf-8")

    def test_rebuild_overwrites_cleanly(self, tmp_path, monkeypatch):
        run_dir, content_file = make_run_dir(tmp_path)
        out = tmp_path / "report.html"
        argv = ["build_report.py", "--run", str(run_dir), "--content", str(content_file),
                "--out", str(out)]
        monkeypatch.setattr("sys.argv", argv)
        B.main()
        first = out.read_text(encoding="utf-8")
        B.main()
        assert out.read_text(encoding="utf-8") == first

    def test_missing_audit_report_exits_with_guidance(self, tmp_path):
        run_dir, content_file = make_run_dir(tmp_path)
        (run_dir / "audit" / "audit_report.json").unlink()
        with pytest.raises(SystemExit, match="audit_dad.py"):
            B.load_inputs(run_dir, content_file)

    def test_loads_real_run_shaped_inputs(self, tmp_path):
        run_dir, content_file = make_run_dir(tmp_path)
        kwargs = B.load_inputs(run_dir, content_file)
        assert kwargs["audit"]["n_prompts"] == 2
        assert kwargs["baseline"][0]["prompt_id"] == "AW-0001"
        assert kwargs["diversity"]["vendi"]["score"] == 5.15
