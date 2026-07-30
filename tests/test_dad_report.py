"""Tests for report/dad.py — the dilemma corpus's section of the handoff page.

The section never renders alone any more, so every test here builds the whole page
around it (report/page.py owns the shell) and asserts on the ``#dad`` beats.

Five things carry real risk here and get most of the coverage:

  * **Degradation.** Not every committed run has the paid delivery/showcase keys, so
    the generator must render a complete section from a partial audit and say what is
    missing rather than quietly omitting it.
  * **Self-containment.** The artefact's whole format exists so it can be opened
    offline from a filesystem. One external asset reference breaks that.
  * **Candour.** The weaknesses beat is derived from the data, not written, so the
    failing checks are asserted to survive into the HTML; the view may collapse rows
    but only with a visible count.
  * **Saying it once.** The delivery regression is stated in prose exactly once. It
    used to be stated four times, which reads as hedging.
  * **Colour integrity.** Arm colours must follow the arm rather than the row order,
    and a series hue must never double as the page's "good".

Fully offline — the generator touches no network and no API, so no stubs beyond the
suite's autouse guards are needed.
"""

import json
import re

import pytest

from report import dad as D
from report import page as P
from report import render as R
from report import sdf as S

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

MANIFEST = {"run_id": "2026-07-20_20-51_bedrock-40", "created_at": "2026-07-20T20:51:58",
            "git_commit": "abc12345", "git_dirty": True,
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
DEALS = [{"domain": ["public policy / law"], "taxa_category": "farmed animals",
          "cultural_setting": "Brazil, written in Portuguese"}]

CONTENT = {k: f"Prose for {k}." for k in P.CONTENT_IDS + D.CONTENT_IDS + S.CONTENT_IDS}
CONTENT["title"] = "Test report"
CONTENT["example_pick"] = "auto"
CONTENT["dad_what"] = "A {{n}}-example run, {{near_dup_pct}} near-duplicated."


def content(**overrides):
    return {**CONTENT, **overrides}


def build(**kwargs):
    """Build the whole page around this DAD run. The section never renders alone."""
    kwargs.setdefault("audit", AUDIT_FULL)
    page_content = kwargs.pop("content", None) or content()
    example = kwargs.pop("example", None)
    return P.build(content=page_content, dad_inputs=kwargs, example=example)


def dad_section(html):
    """Just the #dad panel, for assertions that must not be satisfied elsewhere. It is
    the last section on the page: synthetic documents comes first throughout."""
    return html[html.index("<section id='dad'"):]


def strip_tags(html):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))


def make_run_dir(tmp_path, audit=None, diversity=DIVERSITY, manifest=MANIFEST, costs=COSTS):
    run_dir = tmp_path / "runs" / "2026-07-20_20-51_bedrock-40"
    (run_dir / "audit").mkdir(parents=True)
    (run_dir / "final").mkdir()
    (run_dir / "baseline").mkdir()
    (run_dir / "step1").mkdir()
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
    (run_dir / "step1" / "scenario_deals.jsonl").write_text(
        "\n".join(json.dumps(d) for d in DEALS), encoding="utf-8")
    (run_dir / "final" / "dad_corpus.jsonl").write_text(
        json.dumps({"record_id": "AW-0001", "messages": []}), encoding="utf-8")
    content_file = tmp_path / "content_all.md"
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

    def test_the_delivery_comparison_is_not_available_to_prose(self):
        """It is written once, by _delivery_statement(). A clause in facts() is an
        invitation to write it a second time in a prose file."""
        assert "delivery_clause" not in D.facts(AUDIT_FULL)
        assert "substance_clause" not in D.facts(AUDIT_FULL)

    def test_footprint_regressions_are_derived(self):
        """The prose used to assert 'one of these is an outright regression' about a
        section whose every block is conditional."""
        assert "structural variety" in D.facts(AUDIT_FULL)["footprint_regressions"]
        clean = json.loads(json.dumps(AUDIT_FULL))
        clean["structure"]["pipeline"]["effective_shapes"] = 20.0
        clean["response_lengths"]["mean_ratio"] = 1.0
        clean["moves"]["stance"]["pipeline"]["moralizes"] = 0.0
        assert D.facts(clean)["footprint_regressions"].startswith("None of these")

    def test_spread_counts_the_dealt_axes(self):
        """Counted off step 1's deals, where the spread is engineered — and a list
        value (a scenario with two domains) counts as two, not one."""
        deals = [{"domain": ["a", "b"], "taxa_category": "x"}, {"domain": ["b"]}]
        assert D.spread(deals) == "2 domains · 1 taxa groups"
        assert D.spread([]) == ""


class TestBuildSection:
    def test_builds_every_beat(self):
        html = build(diversity=DIVERSITY, manifest=MANIFEST, costs=COSTS,
                     baseline=BASELINE, rewrites=REWRITES, run_id="run-x")
        for anchor, label in D.BEATS:
            assert f"<h3 id='{anchor}'>{label}</h3>" in html

    def test_the_beats_are_flat_children_of_one_section(self):
        """A figure has to be a direct child of the section for the CSS grid to bleed
        it past the text measure, so no beat may wrap itself in a container — and the
        panel IS that section rather than a wrapper around one."""
        section = dad_section(build(diversity=DIVERSITY))
        assert section.count("<section") == 1
        assert "class='panel'" in section.split(">", 1)[0]

    def test_is_self_contained(self):
        html = build(diversity=DIVERSITY, manifest=MANIFEST)
        assert not re.search(r"<(link|iframe)\b", html)
        assert not re.search(r"<script[^>]*\ssrc=", html)
        assert "@import" not in html and "url(" not in html
        refs = re.findall(r"(?:src|href)='([^']+)'", html)
        assert refs and all(r.startswith(("data:", "#", "https://")) for r in refs)

    def test_prose_hyperlinks_are_allowed(self):
        html = build(content=content(dad_what="See [the post](https://x.test/y)."))
        assert "href='https://x.test/y'" in html

    def test_is_light_mode_only(self):
        html = build()
        assert "color-scheme:only light" in html
        assert "content='only light'" in html
        assert "prefers-color-scheme" not in html
        assert "data-theme" not in html

    def test_placeholders_are_resolved(self):
        html = build()
        assert "{{" not in html
        assert "A 2-example run" in html

    def test_escapes_hostile_corpus_text(self):
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["showcase"]["examples"][0]["pipeline_response"] = "<script>alert(1)</script>"
        html = build(audit=audit, baseline=BASELINE)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

    def test_the_report_is_titled_for_what_it_teaches(self):
        """"The dilemma corpus" told a reader nothing they could act on, and "corpora"
        was the wrong register for the whole page."""
        html = build()
        assert f"<h2>{D.SECTION_TITLE}</h2>" in html
        assert D.SECTION_TITLE == "Difficult advice"
        headings = re.findall(r"<h2>([^<]*)</h2>", html)
        assert headings and not any(h[0].isdigit() for h in headings)
        assert "corpus" not in strip_tags(html).lower().replace("dad_corpus.jsonl", "")

    def test_no_eyebrow(self):
        """The uppercase kicker over the title read as generated; it is gone, and the
        page's only uppercase treatment is now the chip."""
        assert "eyebrow" not in build()

    def test_anchored_beats_land_with_headroom(self):
        """A link used to drop the heading flush against the top of the viewport.
        Sub-beats are link targets too — the chooser opens a panel from #dad-weak."""
        html = build()
        assert "scroll-behavior:smooth" in html
        assert re.search(r"section\{[^}]*scroll-margin-top:[\d.]+rem", html)
        assert re.search(r"h3\[id\]\{[^}]*scroll-margin-top:[\d.]+rem", html)
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


class TestSayingItOnce:
    """The delivery regression was stated in four places, which reads as hedging."""

    @staticmethod
    def _regressed():
        audit = json.loads(json.dumps(AUDIT_FULL))
        audit["delivery"]["pipeline_mean"] = 7.0
        audit["delivery"]["plain_mean"] = 7.8
        return audit

    def test_prose_states_it_exactly_once(self):
        """Tables, tiles and the derived weakness carry the same number as DATA, which
        is not the same as saying it again."""
        html = build(audit=self._regressed(), manifest=MANIFEST)
        # Strip inline SVG first: path data is full of decimals, and a paragraph that
        # merely contains an icon is not a paragraph that states a finding. (\b on the
        # p as well, or the regex matches <path> too.)
        text = re.sub(r"<svg\b.*?</svg>", " ", html, flags=re.S)
        prose = re.findall(r"<p\b[^>]*>(.*?)</p>", text, re.S)
        said = [p for p in prose if "7.0" in p and "7.8" in p]
        assert len(said) == 1, said
        assert "bad-note" in html[:html.index(said[0])].rsplit("<p", 1)[-1]

    def test_it_is_in_the_results_where_the_reader_lands(self):
        html = build(audit=self._regressed(), manifest=MANIFEST)
        section = dad_section(html)
        assert section.index("went the wrong way") < section.index("id='dad-example'")

    def test_the_number_still_reaches_the_tile_and_the_weaknesses(self):
        html = build(audit=self._regressed(), manifest=MANIFEST)
        assert "chip bad'>regression" in html
        assert "wrong way" in strip_tags(html[html.index("id='dad-weak'"):])


class TestChartBudget:
    """Two charts lead; every other one is in the appendix drawer."""

    def test_only_the_two_lead_charts_are_outside_the_appendix(self):
        html = build(diversity=DIVERSITY, manifest=MANIFEST, baseline=BASELINE)
        section = dad_section(html)
        lead = section[:section.index("id='dad-appendix'")]
        titles = re.findall(r"<figcaption class='fig-t'>([^<]*)</figcaption>", lead)
        assert titles == ["Valuable welfare considerations per answer",
                          "Substance against manner, one dot per answer"]

    def test_the_demoted_charts_are_still_on_the_page(self):
        """Moved, not dropped — and the drawer names how many it holds."""
        html = build(diversity=DIVERSITY, manifest=MANIFEST, baseline=BASELINE)
        appendix = dad_section(html)[dad_section(html).index("id='dad-appendix'"):]
        for title in ("Answer length", "Stance", "Structural variety",
                      "What happened to the control's considerations",
                      "Delivery quality, dimension by dimension"):
            assert title in appendix
        assert re.search(r"\d+ figures", appendix)

    def test_the_drawer_says_which_measures_went_the_wrong_way(self):
        html = build(diversity=DIVERSITY, manifest=MANIFEST)
        assert "figures · On this run" in html


class TestDegradation:
    def test_offline_only_audit_still_builds(self):
        audit = {k: v for k, v in AUDIT_FULL.items()
                 if k not in ("delivery", "showcase", "moves", "moral_patient_reasons")}
        html = build(audit=audit)
        assert "id='dad-what'" in html
        assert "None" not in strip_tags(html)

    def test_missing_delivery_says_so_rather_than_omitting(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "delivery"}
        text = strip_tags(build(audit=audit))
        assert "not measured on this run" in text
        assert "--reasons" in text

    def test_missing_delivery_drops_the_scatter_without_a_hole(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "delivery"}
        html = build(audit=audit)
        assert "Substance against manner" not in html
        assert "id='dad-example'" in html

    def test_delivery_present_renders_the_pareto(self):
        html = build()
        assert "Substance against manner" in html
        assert "<circle" in html

    def test_bare_audit_still_carries_the_narrative(self):
        html = build(audit={"n_prompts": 3})
        assert "Prose for method_intro." in html
        assert "Prose for reproduce." in html

    def test_missing_manifest_diversity_and_costs(self):
        html = build(manifest=None, diversity=None, costs=None)
        assert "id='dad-what'" in html

    def test_missing_gid_map_falls_back_to_prompt_ids(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "gid_map"}
        assert "AW-0001" in build(audit=audit)


class TestWorkedExample:
    def test_showcase_example_is_used_and_highlighted(self):
        html = build(baseline=BASELINE)
        assert "Should I do the thing?" in html
        assert "<mark>the animals</mark>" in html
        assert "showcase judge" in strip_tags(html)

    def test_both_answers_are_shown_in_full(self):
        """The two answers are the artefact. They stay inline and verbatim; only the
        word-level diff moved to the appendix."""
        html = build(baseline=BASELINE)
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
        html = build(content=content(example_pick="AW-0001"),
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
        html = build(content=content(example_pick="AW-0001"),
                     baseline=BASELINE, rewrites=REWRITES)
        assert "<ins>" in html
        assert "3 largest changes" in html
        appendix = html[html.find("id='dad-appendix'"):]
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
        html = build(diversity=DIVERSITY)
        chart = html[html.find("Valuable welfare considerations per answer"):]
        chart = chart[:chart.find("</svg>")]
        fills = re.findall(r"fill='(var\(--series-\d\))'", chart)
        assert fills == [R.PLAIN, R.PIPELINE]

    def test_every_chart_carries_an_accessible_name(self):
        """Charts are named; the button icons are decorative and marked aria-hidden,
        which is the correct treatment for a mark that repeats its own label."""
        html = build(diversity=DIVERSITY, manifest=MANIFEST, costs=COSTS,
                     baseline=BASELINE, rewrites=REWRITES)
        for svg in re.findall(r"<svg\b.*?</svg>", html, flags=re.S):
            assert "<title>" in svg or "aria-hidden='true'" in svg


class TestCandour:
    """The weaknesses floor is derived from the run, so it cannot be edited away."""

    def test_bad_verdicts_reach_the_report(self):
        text = strip_tags(build(manifest=MANIFEST))
        assert "Response stance" in text
        assert "BAD" in text

    def test_moralizing_regression_is_shown_in_both_arms(self):
        text = strip_tags(build())
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
        assert "not fully matched" in strip_tags(build())

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

    def test_delivery_gain_is_not_flagged(self):
        warnings = D.derived_warnings(AUDIT_FULL, MANIFEST, D.facts(AUDIT_FULL, MANIFEST))
        assert not any("wrong way" in w for _, w in warnings)

    def test_missing_delivery_is_a_derived_weakness(self):
        audit = {k: v for k, v in AUDIT_FULL.items() if k != "delivery"}
        warnings = D.derived_warnings(audit, MANIFEST, D.facts(audit, MANIFEST))
        assert any(sev == "BAD" and "showcase" in w for sev, w in warnings)

    def test_weaknesses_render_without_any_editorial_prose(self):
        html = build(content=content(weaknesses_intro=""), manifest=MANIFEST)
        section = html[html.find("id='dad-weak'"):html.find("id='dad-appendix'")]
        assert "BAD" in section

    def test_every_check_is_listed_in_the_appendix(self):
        """The 24-row table moved out of the main flow, but it did not leave the page:
        'nothing was left out' has to stay a claim a reader can check."""
        html = build()
        appendix = html[html.find("id='dad-appendix'"):]
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

    def test_a_subheading_is_a_deep_link_target(self):
        assert R.sub("dad-weak", "Where it is weak") == \
            "<h3 id='dad-weak'>Where it is weak</h3>"

    def test_the_illustration_is_a_data_uri_or_an_honest_hole(self):
        """What fills the slot has to travel inside the file, so the primitive takes a
        data URI and refuses anything else."""
        empty = R.illustration()
        assert "TODO" in empty and "src=" not in empty
        filled = R.illustration("data:image/png;base64,AAAA", alt="a butterfly")
        assert "<img src='data:image/png;base64,AAAA' alt='a butterfly'>" in filled
        assert "TODO" not in filled
        with pytest.raises(ValueError, match="data: URI"):
            R.illustration("../assets/hero.png")


    def test_drawer_summaries_name_their_payload_size(self):
        assert "1,010 words" in R.details("Full answer", "x", meta="1,010 words")

    def test_print_rules_keep_figures_and_rows_whole(self):
        block = R.CSS[R.CSS.find("@media print"):]
        assert "figure" in block and "break-inside:avoid-page" in block
        assert "thead{display:table-header-group}" in block


class TestCLI:
    def _argv(self, run_dir, content_file, out_dir, sdf_run=None):
        argv = ["build_report.py", "--dad-run", str(run_dir),
                "--content", str(content_file), "--out-dir", str(out_dir)]
        return argv + (["--sdf-run", str(sdf_run)] if sdf_run else [])

    def test_writes_one_file(self, tmp_path, monkeypatch):
        """One page, named index.html so it publishes to Pages as it stands."""
        from report import build_report as B
        run_dir, content_file = make_run_dir(tmp_path)
        monkeypatch.setattr("sys.argv", self._argv(run_dir, content_file, tmp_path))
        B.main()
        assert not (tmp_path / "dad.html").exists()
        out = tmp_path / "index.html"
        assert "<section id='dad' class='panel'" in out.read_text(encoding="utf-8")
        assert "<section id='sdf' class='panel'" in out.read_text(encoding="utf-8")

    def test_rebuild_overwrites_cleanly(self, tmp_path, monkeypatch):
        from report import build_report as B
        run_dir, content_file = make_run_dir(tmp_path)
        monkeypatch.setattr("sys.argv", self._argv(run_dir, content_file, tmp_path))
        B.main()
        first = (tmp_path / "index.html").read_text(encoding="utf-8")
        B.main()
        assert (tmp_path / "index.html").read_text(encoding="utf-8") == first

    def test_a_dad_run_alone_is_enough(self, tmp_path, monkeypatch):
        from report import build_report as B
        run_dir, content_file = make_run_dir(tmp_path)
        monkeypatch.setattr("sys.argv", self._argv(run_dir, content_file, tmp_path))
        B.main()
        assert "not published yet" in (tmp_path / "index.html").read_text(encoding="utf-8")

    def test_no_dad_run_exits_with_guidance(self, tmp_path, monkeypatch):
        from report import build_report as B
        monkeypatch.setattr("sys.argv", ["build_report.py", "--out-dir", str(tmp_path)])
        with pytest.raises(SystemExit, match="--dad-run"):
            B.main()

    def test_missing_audit_report_exits_with_guidance(self, tmp_path):
        run_dir, _ = make_run_dir(tmp_path)
        (run_dir / "audit" / "audit_report.json").unlink()
        with pytest.raises(SystemExit, match="audit_dad.py"):
            D.load_inputs(run_dir)

    def test_loads_real_run_shaped_inputs(self, tmp_path):
        run_dir, _ = make_run_dir(tmp_path)
        kwargs = D.load_inputs(run_dir)
        assert kwargs["audit"]["n_prompts"] == 2
        assert kwargs["baseline"][0]["prompt_id"] == "AW-0001"
        assert kwargs["diversity"]["vendi"]["score"] == 5.15
        assert kwargs["deals"][0]["taxa_category"] == "farmed animals"
        assert "content" not in kwargs  # the page owns one content namespace
