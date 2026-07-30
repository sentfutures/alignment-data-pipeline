"""Tests for report/dad.py — the standalone DAD report page.

Four things carry real risk here and get most of the coverage:

  * **Degradation.** Not every committed run has the paid delivery/showcase keys, so
    the generator must render a complete page from a partial audit and say what is
    missing rather than quietly omitting it — including in the lede.
  * **Self-containment.** The artefact's whole format exists so it can be opened
    offline from a filesystem. One external asset reference breaks that.
  * **Candour.** The weaknesses section is derived from the data, not written, so the
    failing checks are asserted to survive into the HTML; the view may collapse rows
    but only with a visible count.
  * **Colour integrity.** Arm colours must follow the arm rather than the row order,
    and a series hue must never double as the page's "good".

Fully offline — the generator touches no network and no API, so no stubs beyond the
suite's autouse guards are needed.
"""

import json
import re

import pytest

from report import dad as D
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
        "n": 2, "failures": 1, "model": "claude-sonnet-5", "judge_model": "claude-opus-5",
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
        "pipeline_mean": 8.2, "plain_mean": 7.9, "n_pipeline": 2, "n_plain": 2, "failures": 0,
        "dimensions": {"pipeline": {"tone": 8.0, "calibration": 9.0},
                       "plain": {"tone": 8.5, "calibration": 8.0}},
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
    "response_lengths": {"n": 2, "pipeline_mean": 4659.0, "plain_mean": 2988.0,
                         "mean_ratio": 1.56, "per_case": {}},
    "tracked_tics": {"n_pipeline": 2, "n_plain": 2,
                     "watch": {"cuts both ways": {"origin": "pipeline-origin",
                                                  "pipeline": 1, "plain": 0}}},
    "rhetorical_moves": {"moves": {"unbundling": {"description": "splits a bundled choice",
                                                  "pipeline_share": 0.28, "plain_share": 0.28},
                                   "autonomy-coda": {"description": "hands the call back",
                                                     "pipeline_share": 0.38, "plain_share": 0.0}}},
    "structure": {"pipeline": {"effective_shapes": 9.44},
                  "plain": {"effective_shapes": 13.88}},
    "library_coverage": {"n_cases": 2, "library_size": 44, "used": 37},
    "reason_composition": {"type_gloss": {"direct": "the animal's own experience"}},
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

CONTENT = {k: f"Prose for {k}." for k in D.CONTENT_IDS}
CONTENT["title"] = "Test report"
CONTENT["lede"] = "A {{n}}-example run, {{delivery_clause}}."
CONTENT["example_pick"] = "auto"


def content(**overrides):
    return {**CONTENT, **overrides}


def build(**kwargs):
    kwargs.setdefault("content", content())
    return D.build(**kwargs)


def strip_tags(html):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))


def make_run_dir(tmp_path, audit=None, diversity=DIVERSITY, manifest=MANIFEST, costs=COSTS):
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
    content_file = tmp_path / "content_dad.md"
    content_file.write_text("".join(f"<!-- id: {k} -->\n{v}\n\n" for k, v in CONTENT.items()),
                            encoding="utf-8")
    return run_dir, content_file


class TestFacts:
    def test_reconstructs_considerations_from_legacy_schema(self):
        cons = D._considerations(AUDIT_FULL)
        assert cons["source"] == "reconstructed"
        assert cons["pipeline"] == pytest.approx(5.5)  # 2.5 reasoning + 3.0 alternatives
        assert cons["plain"] == pytest.approx(3.5)

    def test_prefers_modern_schema_when_present(self):
        audit = dict(AUDIT_FULL, valuable_welfare_considerations={
            "available": True, "parent": {"pipeline": 9.0, "plain": 6.0},
            "subsets": [{"name": "welfare reasoning", "pipeline": 5.0, "plain": 4.0}]})
        cons = D._considerations(audit)
        assert cons["source"] == "modern"
        assert cons["pipeline"] == 9.0

    def test_facts_are_read_from_the_data_not_hardcoded(self):
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["response_lengths"]["mean_ratio"] = 2.5
        assert D.facts(audit)["length_pct"] == "150%"

    def test_dealt_and_measured_counts_are_distinguished(self):
        """40 dilemmas dealt and 39 measured is the normal case, and reporting the
        first as if it were the second is the kind of thing a reader spots in
        thirty seconds."""
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["n_prompts"] = 40
        f = D.facts(audit)
        assert f["n"] == 40 and f["n_measured"] == 2

    def test_the_extractor_is_not_credited_as_the_judge(self):
        f = D.facts(AUDIT_FULL)
        assert f["extract_model"] == "claude-sonnet-5"
        assert f["judge_model"] == "claude-opus-5"

    def test_delivery_is_reported_out_of_ten_not_as_a_percentage(self):
        f = D.facts(AUDIT_FULL)
        assert f["delivery_pipeline"] == "8.2"
        assert "%" not in f["delivery_clause"]
        assert "out of 10" in f["delivery_clause"]

    def test_footprint_regressions_are_derived(self):
        """The prose used to assert 'one of these is an outright regression' about a
        section whose every block is conditional."""
        assert "structural variety" in D.facts(AUDIT_FULL)["footprint_regressions"]
        clean = json.loads(json.dumps(AUDIT_FULL))
        clean["structure"]["pipeline"]["effective_shapes"] = 20.0
        clean["response_lengths"]["mean_ratio"] = 1.0
        clean["moves"]["stance"]["pipeline"]["moralizes"] = 0.0
        assert D.facts(clean)["footprint_regressions"].startswith("None of these")


class TestBuildPage:
    def test_builds_every_section(self):
        html = build(audit=AUDIT_FULL, diversity=DIVERSITY, manifest=MANIFEST, costs=COSTS,
                     baseline=BASELINE, rewrites=REWRITES, run_id="run-x")
        for sid, _ in D.TOC:
            assert f"id='{sid}'" in html

    def test_is_self_contained(self):
        """No external CSS, JS, fonts or images — the file must open offline."""
        html = build(audit=AUDIT_FULL, diversity=DIVERSITY, manifest=MANIFEST)
        assert not re.search(r"<(img|link|iframe)\b", html)
        assert not re.search(r"<script[^>]*\ssrc=", html)
        assert "@import" not in html and "url(" not in html

    def test_prose_hyperlinks_are_allowed(self):
        html = build(audit=AUDIT_FULL, content=content(gap="See [the post](https://x.test/y)."))
        assert "href='https://x.test/y'" in html

    def test_is_light_mode_only(self):
        """One theme, deliberately — see render.py's docstring. This replaces
        test_both_dark_mode_declarations_survive: the spec changed, not the code's
        luck. `only light` also opts out of mobile browsers' auto-darkening, which
        prefers-color-scheme does not cover."""
        html = build(audit=AUDIT_FULL)
        assert "color-scheme:only light" in html
        assert "content='only light'" in html
        assert "prefers-color-scheme" not in html
        assert "data-theme" not in html

    def test_placeholders_are_resolved(self):
        html = build(audit=AUDIT_FULL)
        assert "{{" not in html
        assert "A 2-example run" in html

    def test_escapes_hostile_corpus_text(self):
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["showcase"]["examples"][0]["pipeline_response"] = "<script>alert(1)</script>"
        html = build(audit=audit, baseline=BASELINE)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

    def test_the_rail_carries_the_section_numbers_not_the_headings(self):
        """Numbered headings plus parallel What/How/Where phrasing was the page's
        loudest structural tell; the rail's counter() supplies the numbers now."""
        html = build(audit=AUDIT_FULL)
        assert "<h2>The gap</h2>" in html
        headings = re.findall(r"<h2>([^<]*)</h2>", html)
        assert headings and not any(h[0].isdigit() for h in headings)
        assert "counter(sec)" in html  # the rail supplies them instead

    def test_the_rail_opens_with_the_back_link(self):
        """The back-link is the one bit of cross-page navigation, and it sits at the top
        of the rail rather than in the footer so it stays reachable once a reader is
        deep in the page."""
        html = build(audit=AUDIT_FULL, sibling=("index.html", "Overview"))
        rail = re.search(r"<nav class='rail'.*?</nav>", html, re.S).group(0)
        assert "href='index.html'" in rail
        assert rail.index("index.html") < rail.index("href='#gap'")
        assert ">Contents<" in rail

    def test_no_sibling_renders_no_dead_link(self):
        assert "index.html" not in build(audit=AUDIT_FULL)

    def test_no_eyebrow(self):
        """The uppercase kicker over the title read as generated; it is gone, and the
        page's only uppercase treatment is now the chip."""
        assert "eyebrow" not in build(audit=AUDIT_FULL)

    def test_anchored_sections_land_with_headroom(self):
        """A rail link used to drop the heading flush against the top of the viewport,
        and under the sticky rail's own offset."""
        html = build(audit=AUDIT_FULL)
        assert "scroll-behavior:smooth" in html
        assert re.search(r"section\{[^}]*scroll-margin-top:[\d.]+rem", html)
        reduced = html[html.find("@media (prefers-reduced-motion:reduce)"):][:120]
        assert "scroll-behavior:auto" in reduced


class TestScoreboard:
    def test_reports_the_measures_that_undercut_the_headline(self):
        """Density and structural variety both move the wrong way while the headline
        moves the right way. They belong next to it, not in a footnote."""
        html = D.scoreboard(AUDIT_FULL, D.facts(AUDIT_FULL), D._considerations(AUDIT_FULL))
        text = strip_tags(html)
        assert "considerations per 1,000 characters" in text
        assert "structural variety" in text
        assert "answer length" in text

    def test_an_unmeasured_row_says_so_rather_than_vanishing(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "moves"}
        html = D.scoreboard(audit, D.facts(audit), D._considerations(audit))
        assert "not measured" in strip_tags(html)

    def test_a_worse_number_gets_the_bad_chip(self):
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["delivery"]["pipeline_mean"] = 7.0
        audit["delivery"]["plain_mean"] = 7.9
        html = D.scoreboard(audit, D.facts(audit), D._considerations(audit))
        assert "chip bad'>worse" in html


class TestDegradation:
    def test_offline_only_audit_still_builds(self):
        audit = {k: v for k, v in AUDIT_FULL.items()
                 if k not in ("delivery", "showcase", "moves", "moral_patient_reasons")}
        html = build(audit=audit)
        assert "id='results'" in html
        assert "None" not in strip_tags(html)

    def test_missing_delivery_says_so_rather_than_omitting(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "delivery"}
        text = strip_tags(build(audit=audit))
        assert "not measured on this run" in text
        assert "--reasons" in text

    def test_lede_degrades_without_the_paid_pass(self):
        """The lede carries the run's two headline numbers, so it has to survive a run
        that never measured them — with the caveat in their place."""
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "delivery"}
        html = build(audit=audit)
        lede = re.search(r"<p class='lede'>(.*?)</p>", html, re.S).group(1)
        assert "Judged delivery was not measured on this run" in lede
        assert "{{" not in html

    def test_delivery_present_renders_the_pareto(self):
        html = build(audit=AUDIT_FULL)
        assert "Substance against manner" in html
        assert "<circle" in html

    def test_bare_audit_still_carries_the_narrative(self):
        html = build(audit={"n_prompts": 3})
        assert "Prose for gap." in html
        assert "Prose for reproduce." in html

    def test_missing_manifest_diversity_and_costs(self):
        html = build(audit=AUDIT_FULL, manifest=None, diversity=None, costs=None)
        assert "id='results'" in html

    def test_missing_gid_map_falls_back_to_prompt_ids(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "gid_map"}
        assert "AW-0001" in build(audit=audit)


class TestWorkedExample:
    def test_showcase_example_is_used_and_highlighted(self):
        html = build(audit=AUDIT_FULL, baseline=BASELINE)
        assert "Should I do the thing?" in html
        assert "<mark>the animals</mark>" in html
        assert "showcase judge" in strip_tags(html)

    def test_both_answers_are_shown_in_full(self):
        """The two answers are the artefact. They stay inline and verbatim; only the
        word-level diff moved to the appendix."""
        html = build(audit=AUDIT_FULL, baseline=BASELINE)
        pair = html[html.find("<div class='pair'>"):]
        pair = pair[:pair.find("</div></div></div>") + len("</div></div></div>")]
        assert "<details" not in pair  # neither answer is behind a drawer
        text = strip_tags(pair)
        assert "Maybe." in text and "Consider the animals here." in text

    def test_non_locating_highlight_leaves_text_whole(self):
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["showcase"]["examples"][0]["highlights"] = ["text that is not present"]
        html = build(audit=audit, baseline=BASELINE)
        assert "Consider the animals here." in html
        assert "<mark>" not in html

    def test_pinned_example_overrides_the_showcase(self):
        html = build(audit=AUDIT_FULL, content=content(example_pick="AW-0001"),
                     baseline=BASELINE, rewrites=REWRITES)
        assert "pinned in the prose file" in strip_tags(html)

    def test_falls_back_to_the_most_added_record(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "showcase"}
        html = build(audit=audit, baseline=BASELINE, rewrites=REWRITES)
        assert "selected mechanically" in strip_tags(html)

    def test_example_text_comes_from_the_run_files(self):
        """The generator pulls response text off disk; nothing is retyped."""
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "showcase"}
        html = build(audit=audit, baseline=BASELINE, rewrites=REWRITES)
        assert "Consider the animals here." in html

    def test_rewrite_diff_shows_hunks_inline_and_the_whole_thing_in_the_appendix(self):
        html = build(audit=AUDIT_FULL, content=content(example_pick="AW-0001"),
                     baseline=BASELINE, rewrites=REWRITES)
        assert "<ins>" in html
        assert "3 largest changes" in html
        appendix = html[html.find("id='appendix'"):]
        assert "full stage-3 rewrite diff" in appendix.lower()

    def test_diff_summary_reports_how_much_changed(self):
        assert "%" in D._diff_summary("a b c d", "a b c e")

    def test_no_example_data_is_reported_not_crashed(self):
        assert "No worked example" in strip_tags(build(audit={"n_prompts": 1}))


class TestColourIntegrity:
    def test_status_colors_are_not_series_colors(self):
        """--good used to be byte-identical to --series-3, the pipeline's own hue, so
        the palette quietly editorialised 'pipeline = good'."""
        series = set(re.findall(r"--series-\d:(#[0-9a-f]{6})", R.CSS))
        status = set(re.findall(r"--(?:good|warn|bad):(#[0-9a-f]{6})", R.CSS))
        assert not (series & status)

    def test_arm_colors_follow_the_arm_not_the_row_order(self):
        """hbar(color=None) falls back to PAL[i], which painted the headline chart's
        pipeline bar in the control's colour while every other chart used green."""
        html = build(audit=AUDIT_FULL, diversity=DIVERSITY)
        chart = html[html.find("Valuable welfare considerations per answer"):]
        chart = chart[:chart.find("</svg>")]
        fills = re.findall(r"fill='(var\(--series-\d\))'", chart)
        assert fills == [R.PLAIN, R.PIPELINE]

    def test_every_chart_carries_an_accessible_name(self):
        html = build(audit=AUDIT_FULL, diversity=DIVERSITY, manifest=MANIFEST, costs=COSTS,
                     baseline=BASELINE, rewrites=REWRITES)
        for svg in re.findall(r"<svg\b.*?</svg>", html, flags=re.S):
            assert "<title>" in svg


class TestCandour:
    """The weaknesses floor is derived from the run, so it cannot be edited away."""

    def test_bad_verdicts_reach_the_report(self):
        text = strip_tags(build(audit=AUDIT_FULL, manifest=MANIFEST))
        assert "Response stance" in text
        assert "BAD" in text

    def test_moralizing_regression_is_shown_in_both_arms(self):
        text = strip_tags(build(audit=AUDIT_FULL))
        assert "40%" in text and "0%" in text

    def test_non_faithful_backend_is_flagged(self):
        warnings = D.derived_warnings(AUDIT_FULL, MANIFEST, D.facts(AUDIT_FULL, MANIFEST))
        assert any("bedrock" in w for _, w in warnings)

    def test_api_backend_removes_that_warning(self):
        manifest = json.loads(json.dumps(MANIFEST))
        manifest["config"]["backend"] = "api"
        warnings = D.derived_warnings(AUDIT_FULL, manifest, D.facts(AUDIT_FULL, manifest))
        assert not any("faithful mode" in w for _, w in warnings)

    def test_dirty_git_tree_is_surfaced(self):
        warnings = D.derived_warnings(AUDIT_FULL, MANIFEST, D.facts(AUDIT_FULL, MANIFEST))
        assert any("uncommitted" in w for _, w in warnings)

    def test_extraction_failures_produce_an_asymmetry_note(self):
        assert "not a fully matched comparison" in strip_tags(build(audit=AUDIT_FULL))

    def test_delivery_arm_asymmetry_is_disclosed(self):
        """The bedrock-40 case: the one BAD headline was a mean over 33 pipeline
        answers against 26 different control answers, with 19 judgements dropped, and
        the page said nothing. The retention rule reads its own failures, not
        delivery's, so this needed its own rule."""
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["delivery"].update(n_pipeline=33, n_plain=26, failures=19)
        warnings = D.derived_warnings(audit, MANIFEST, D.facts(audit, MANIFEST))
        assert any("not a matched comparison" in w and "19" in w for _, w in warnings)
        assert "different sets of records" in strip_tags(build(audit=audit, manifest=MANIFEST))

    def test_matched_arms_are_not_flagged(self):
        warnings = D.derived_warnings(AUDIT_FULL, MANIFEST, D.facts(AUDIT_FULL, MANIFEST))
        assert not any("not a matched comparison" in w for _, w in warnings)

    def test_delivery_regression_leads_the_weaknesses(self):
        """The substance/manner trade this method exists to avoid, going the wrong
        way, must surface as BAD and first — the bedrock-40 case."""
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["delivery"]["pipeline_mean"] = 7.0
        audit["delivery"]["plain_mean"] = 7.9
        warnings = D.derived_warnings(audit, MANIFEST, D.facts(audit, MANIFEST))
        severities = [sev for sev, _ in warnings]
        assert severities == sorted(severities, key=lambda s: s != "BAD")  # BADs first
        assert any(sev == "BAD" and "wrong way" in w for sev, w in warnings)
        html = build(audit=audit, manifest=MANIFEST)
        assert "chip bad'>regression" in html  # and in the hero, where a skimmer lands

    def test_delivery_gain_is_not_flagged(self):
        warnings = D.derived_warnings(AUDIT_FULL, MANIFEST, D.facts(AUDIT_FULL, MANIFEST))
        assert not any("wrong way" in w for _, w in warnings)

    def test_missing_delivery_is_a_derived_weakness(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "delivery"}
        warnings = D.derived_warnings(audit, MANIFEST, D.facts(audit, MANIFEST))
        assert any(sev == "BAD" and "showcase" in w for sev, w in warnings)

    def test_weaknesses_render_without_any_editorial_prose(self):
        html = build(audit=AUDIT_FULL, content=content(weaknesses_intro=""), manifest=MANIFEST)
        section = html[html.find("id='weaknesses'"):html.find("id='reproduce'")]
        assert "BAD" in section

    def test_every_check_is_listed_in_the_appendix(self):
        """The 24-row table moved out of the main flow, but it did not leave the page:
        'nothing was left out' has to stay a claim a reader can check."""
        html = build(audit=AUDIT_FULL)
        appendix = html[html.find("id='appendix'"):]
        assert "Locale / taxa plausibility" in appendix
        assert "Response stance (LLM)" in appendix


class TestRenderPrimitives:
    def test_charts_emit_parseable_svg(self):
        import xml.etree.ElementTree as ET

        ET.fromstring(R.hbar([("a", 1), ("b<script>", 2)]))
        ET.fromstring(R.grouped_hbar([{"label": "x", "p": 1, "q": 2}],
                                     series=[("p", "red"), ("q", "blue")]).split("<div")[0])
        ET.fromstring(R.stacked_bar([{"label": "r", "segments": {"kept": 2}}],
                                    categories=[("kept", "red")]).split("<div")[0])
        ET.fromstring(R.scatter([{"x": 1, "y": 2, "color": "red", "tip": "t"}]))
        ET.fromstring(R.segbar([("kept", 2, "red"), ("added", 1, "blue")]).split("<div")[0])
        ET.fromstring(R.histogram([("7", 2), ("8", 5)]))

    def test_empty_data_is_a_note_not_a_broken_chart(self):
        assert "no" in R.hbar([]).lower()
        assert "<svg" not in R.grouped_hbar([], series=[("a", "red")])
        assert "<svg" not in R.segbar([("kept", 0, "red")])

    def test_segbar_labels_live_in_the_legend_not_on_the_fill(self):
        """Surface-coloured text on the arm fills was 2.5:1 on the green — a fail on
        cream, and already a fail on white."""
        html = R.segbar([("kept", 439, R.PLAIN), ("added", 260, R.PIPELINE)])
        svg = html[:html.find("</svg>")]
        assert "<text" not in svg
        assert "kept · 439" in html and "added · 260" in html

    def test_zero_values_do_not_divide_by_zero(self):
        assert "<svg" in R.hbar([("a", 0), ("b", 0)])

    def test_hbar_takes_one_color_or_a_sequence(self):
        assert R.hbar([("a", 1), ("b", 2)], color="red").count("fill='red'") == 2
        both = R.hbar([("a", 1), ("b", 2)], color=("red", "blue"))
        assert "fill='red'" in both and "fill='blue'" in both

    def test_table_escapes_cells_but_passes_raw_through(self):
        html = R.table(["h"], [("<b>x</b>",), (R.Raw("<b>y</b>"),)])
        assert "&lt;b&gt;x&lt;/b&gt;" in html
        assert "<b>y</b>" in html

    def test_table_right_aligns_the_columns_it_is_told_to(self):
        html = R.table(["a", "n"], [("x", "1")], align="lr")
        assert "<td class='num'>1</td>" in html
        assert "<td>x</td>" in html

    def test_inline_md_escapes_before_formatting(self):
        assert R.inline_md("**a** <b>") == "<b>a</b> &lt;b&gt;"

    def test_deks_come_from_a_prose_convention(self):
        assert R.paragraphs("> the finding\n\nbody") == \
            "<p class='dek'>the finding</p><p>body</p>"

    def test_highlight_is_fail_open(self):
        assert R.highlight("hello", ["nope"]) == "<div class='resp'>hello</div>"
        assert "<mark>ell</mark>" in R.highlight("hello", ["ell"])

    def test_figure_names_the_chart_and_states_the_finding(self):
        html = R.figure(title="T", chart="<svg viewBox='0 0 1 1'></svg>", caption="**F.**")
        assert "<title>T</title>" in html
        assert "<figcaption class='fig-c'><b>F.</b></figcaption>" in html

    def test_at_most_one_hero_tile(self):
        with pytest.raises(ValueError, match="one hero"):
            R.tiles([R.stat("1", "a", tone="hero"), R.stat("2", "b", tone="hero")])

    def test_a_tile_carries_direction_as_a_chip_not_a_colored_numeral(self):
        html = R.stat("7.0", "delivery", flag="regression", tone="bad")
        assert "chip bad'>regression" in html
        assert "class='tile-v'>7.0" in html

    def test_drawer_summaries_name_their_payload_size(self):
        assert "1,010 words" in R.details("Full answer", "x", meta="1,010 words")

    def test_print_rules_keep_figures_and_rows_whole(self):
        block = R.CSS[R.CSS.find("@media print"):]
        assert "figure" in block and "break-inside:avoid-page" in block
        assert "thead{display:table-header-group}" in block


class TestCLI:
    def _argv(self, run_dir, content_file, out_dir):
        return ["build_report.py", "--page", "dad", "--dad-run", str(run_dir),
                "--content", str(content_file), "--out-dir", str(out_dir)]

    def test_writes_the_file(self, tmp_path, monkeypatch):
        from report import build_report as B
        run_dir, content_file = make_run_dir(tmp_path)
        monkeypatch.setattr("sys.argv", self._argv(run_dir, content_file, tmp_path))
        B.main()
        out = tmp_path / "dad.html"
        assert out.exists()
        assert "id='results'" in out.read_text(encoding="utf-8")

    def test_rebuild_overwrites_cleanly(self, tmp_path, monkeypatch):
        from report import build_report as B
        run_dir, content_file = make_run_dir(tmp_path)
        monkeypatch.setattr("sys.argv", self._argv(run_dir, content_file, tmp_path))
        B.main()
        first = (tmp_path / "dad.html").read_text(encoding="utf-8")
        B.main()
        assert (tmp_path / "dad.html").read_text(encoding="utf-8") == first

    def test_missing_audit_report_exits_with_guidance(self, tmp_path):
        run_dir, content_file = make_run_dir(tmp_path)
        (run_dir / "audit" / "audit_report.json").unlink()
        with pytest.raises(SystemExit, match="audit_dad.py"):
            D.load_inputs(run_dir, [content_file])

    def test_loads_real_run_shaped_inputs(self, tmp_path):
        run_dir, content_file = make_run_dir(tmp_path)
        kwargs = D.load_inputs(run_dir, [content_file])
        assert kwargs["audit"]["n_prompts"] == 2
        assert kwargs["baseline"][0]["prompt_id"] == "AW-0001"
        assert kwargs["diversity"]["vendi"]["score"] == 5.15
