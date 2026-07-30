"""Tests for report/page.py — the handoff page that carries both corpora.

This replaces test_report_hub.py: there is no landing page and no second file any more,
so the risks that page had (a link to a report nobody built, a card whose numbers
disagree with the report it points at) are gone. What replaces them:

  * **The choice.** Neither report is open on load; ``#dad`` / ``#sdf`` in the URL
    opens one, which is what the dataset card's deep links depend on, and printing
    expands both.
  * **Degradation.** The page must build from a DAD run alone, with the synthetic
    documents' column and report saying so.
  * **Candour before evidence.** The five shared caveats, including the licence TODO,
    render above both report sections.
  * **Brevity.** The page exists because a reader has forty seconds. Deks are rationed
    and the prose has a ceiling, both asserted here.

Fully offline.
"""

import re

import pytest

from report import common as C
from report import dad as D
from report import page as P
from report import render as R
from report import sdf as S

CONTENT = {k: f"Prose for {k}." for k in P.CONTENT_IDS + D.CONTENT_IDS + S.CONTENT_IDS}
CONTENT["title"] = "Two corpora"
CONTENT["example_pick"] = "auto"

DAD_AUDIT = {
    "n_prompts": 40,
    "valuable_welfare_considerations": {"available": True,
                                        "parent": {"pipeline": 17.07, "plain": 12.54}},
    "moral_patient_reasons": {"n": 39, "model": "claude-sonnet-5", "judge_model": "claude-opus-5",
                              "pipeline": {"n": 39}, "plain": {"n": 39}},
    "delivery": {"pipeline_mean": 7.03, "plain_mean": 7.85,
                 "per_case": {"AW-0001": {"pipeline": {"score": 7}, "plain": {"score": 8}}}},
    "response_lengths": {"n": 39, "mean_ratio": 1.56},
}
DAD_MANIFEST = {"created_at": "2026-07-20T20:51:58", "git_commit": "326e4567", "git_dirty": True,
                "config": {"backend": "bedrock", "model": "claude-sonnet-5",
                           "dad": {"constitution_rewrite_model": "claude-opus-4-8"}}}
DAD_DEALS = [{"domain": ["public policy / law"], "taxa_category": "farmed animals",
              "cultural_setting": "Brazil, written in Portuguese"},
             {"domain": ["consumer choice", "public policy / law"], "taxa_category": "companion",
              "cultural_setting": "Japan, written in Japanese"}]
DAD_INPUTS = {"audit": DAD_AUDIT, "manifest": DAD_MANIFEST, "deals": DAD_DEALS,
              "corpus": [{"record_id": "AW-0001"}] * 39, "costs": [],
              "n_prompt_templates": 8, "run_id": "2026-07-20_20-51_bedrock-40"}

SDF_AUDIT = {"n_docs": 100,
             "composition": {"language": {"English": 29, "Mandarin Chinese": 12},
                             "n_types": 15, "top_type_share": 0.13},
             "length": {"truncated": 12, "truncated_frac": 0.12},
             "near_dups": {"0.9": 0.0},
             "openings": {"formulaic_frac": 0.0},
             "patterns": [{"pattern": "Refuse-then-alternative", "prevalence": 0.013,
                           "is_defect": True, "flagged": False}]}
SDF_MANIFEST = {"created_at": "2026-07-11T20:06:36", "git_commit": "18ede291", "git_dirty": True,
                "config": {"backend": "claude_code", "model": "claude-sonnet-5",
                           "sdf": {"rewrite_model": "claude-fable-5"}}}
SDF_DIVERSITY = {"n_records": 100, "vendi": {"score": 22.58}, "nn": {"over_0.90": 0.0}}
SDF_INPUTS = {"audit": SDF_AUDIT, "manifest": SDF_MANIFEST, "diversity": SDF_DIVERSITY,
              "costs": [], "n_prompt_templates": 4,
              "run_id": "2026-07-11_20-06_matrix100-cli"}


def content(**overrides):
    return {**CONTENT, **overrides}


def shipped_content():
    """The prose files this repository actually publishes.

    The brevity tests measure these rather than the fixtures, because prose growing
    back is the regression they exist to catch. Loading them also pins the id contract:
    a section renamed in a module and not in its prose file fails here.
    """
    from pathlib import Path
    report_dir = Path(__file__).resolve().parent.parent / "report"
    return C.load_content([report_dir / "content_page.md", report_dir / "content_dad.md"],
                          P.CONTENT_IDS + D.CONTENT_IDS + S.CONTENT_IDS)


def build(**kwargs):
    kwargs.setdefault("content", content())
    kwargs.setdefault("dad_inputs", DAD_INPUTS)
    return P.build(**kwargs)


def strip_tags(html):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))


class TestShape:
    def test_the_page_is_three_sections_and_two_reports(self):
        html = build(sdf_inputs=SDF_INPUTS)
        ids = re.findall(r"<section id='([^']+)'", html)
        assert ids == ["datasets", "explore", "sdf", "dad"]
        assert re.findall(r"<h2>([^<]*)</h2>", html) == [
            "Explore example results", S.SECTION_TITLE, D.SECTION_TITLE]
        # The comparison is titled by its own two mastheads, not by a heading over them.
        assert "<section id='datasets'><div class='cmp-wrap'>" in html

    def test_there_is_no_contents_rail(self):
        """Five links beside a page whose whole navigation is one choice is furniture."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert "class='rail'" not in html
        assert "Contents" not in strip_tags(html)
        assert "counter(sec)" not in html

    def test_both_reports_take_the_same_skeleton(self):
        """A reader learns the shape once. Synthetic documents only ships some of the
        beats while its report is in preparation, but the ones it has are the same
        beats under the same names."""
        html = build(sdf_inputs=SDF_INPUTS)
        beats = dict(re.findall(r"<h3 id='([^']+)'>([^<]*)</h3>", html))
        assert beats["dad-what"] == beats["sdf-what"] == "What it is"
        assert beats["dad-weak"] == beats["sdf-weak"] == "Where it is weak"
        for anchor, _ in D.BEATS:
            assert anchor in beats

    def test_the_hero_is_the_image_the_title_and_the_lines_that_follow(self):
        """Image, title, intro, centred, and nothing else. A lede, a meta line or a set
        of tiles in here is the masthead this page was rebuilt to get rid of, and an
        "Intro" heading over the paragraph only names what a reader can already see."""
        html = build(sdf_inputs=SDF_INPUTS)
        hero = re.search(r"<header class='hero'>.*?</header>", html, re.S).group(0)
        assert hero.index("class='illo") < hero.index("<h1>") < hero.index("id='intro'")
        assert "class='lede'" not in hero and "class='meta'" not in hero
        assert "class='tiles'" not in hero
        assert "<h2>Intro</h2>" not in html
        assert re.search(r"\.hero\{[^}]*text-align:center", html)
        assert "min-height" not in re.search(r"\.hero\{[^}]*\}", html).group(0)

    def test_the_hero_image_is_cropped_to_the_ink(self):
        """The artwork is 1536x1024 with its drawing in a 1318x425 band — a third of the
        file is transparent above it and a third below, so uncropped the hero spends
        ~340px on nothing and every gap under it looks wrong. Cropped in CSS, so the
        asset stays exactly as supplied."""
        rule = re.search(r"\.hero \.illo\.art img\{[^}]*\}", build()).group(0)
        assert "aspect-ratio:1318/425" in rule
        assert "object-fit:cover" in rule and "object-position:50% 48.5%" in rule

    def test_the_intro_stops_after_two_paragraphs(self):
        """The two datasets are named once, in the comparison's mastheads. Listing them
        here as well was the same two names twice within a screen."""
        html = build(content=shipped_content(), sdf_inputs=SDF_INPUTS)
        hero = re.search(r"<header class='hero'>.*?</header>", html, re.S).group(0)
        assert "<li>" not in hero and "<ul>" not in hero
        assert hero.count("<p>") == 2

    def test_the_footer_is_one_line_who_made_it_and_where_to_go(self):
        """No run ids, no commits, no build claim: the last line of the page is not
        where a reader goes looking for provenance."""
        html = build(sdf_inputs=SDF_INPUTS)
        foot = re.search(r"<footer class='foot'>.*?</footer>", html, re.S).group(0)
        assert f"A project by <a href='{P.MAKER_URL}'" in foot
        assert P.MAKER in foot
        for gone in ("2026-07-20_20-51_bedrock-40", "bedrock", "git ", "build time"):
            assert gone not in strip_tags(foot), gone

    def test_the_footer_links_are_links_not_buttons(self):
        """Two destinations, each with its mark and the outbound arrow, floated right."""
        html = build(sdf_inputs=SDF_INPUTS)
        foot = re.search(r"<footer class='foot'>.*?</footer>", html, re.S).group(0)
        assert foot.count("class='ilink'") == 2
        assert "class='lbtn'" not in foot          # the buttons live in the comparison
        assert ">Datasets</span>" in foot and ">Pipelines</span>" in foot
        assert foot.index("A project by") < foot.index("foot-links")
        assert re.search(r"footer\.foot\{[^}]*justify-content:space-between", html)

    def test_is_self_contained(self):
        """One file. Every reference in it either stays on the page (an anchor), is
        carried inside it (a data URI), or is prose pointing at the web — never a
        relative path to something that has to travel alongside."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert not re.search(r"<(link|iframe)\b", html)
        assert not re.search(r"<script[^>]*\ssrc=", html)
        assert "@import" not in html and "url(" not in html
        refs = re.findall(r"(?:src|href)='([^']+)'", html)
        assert refs and all(r.startswith(("data:", "#", "https://")) for r in refs)

    def test_the_hero_illustration_is_carried_inside_the_page(self):
        html = build(sdf_inputs=SDF_INPUTS, illustration="data:image/png;base64,AAAA")
        assert "<img src='data:image/png;base64,AAAA'" in html
        assert "TODO: hero illustration" not in html
        assert re.search(r"<img[^>]+alt='[^']+'", html)  # it is a picture, so it needs one

    def test_an_illustration_that_would_have_to_travel_is_refused(self):
        """A relative path renders fine in the repo and 404s the moment the file is
        emailed or published on its own, which is the whole failure this format avoids."""
        with pytest.raises(ValueError, match="data: URI"):
            build(illustration="assets/hero.png")

    def test_is_light_mode_only(self):
        html = build()
        assert "color-scheme:only light" in html
        assert "prefers-color-scheme" not in html

    def test_links_out_only_to_the_repo_and_the_dataset(self):
        html = build(sdf_inputs=SDF_INPUTS)
        origins = {re.match(r"https://[^/]+", u).group(0)
                   for u in re.findall(r"href='(https?://[^']+)'", html)}
        assert origins <= {"https://github.com", "https://huggingface.co",
                           "https://alignment.anthropic.com", P.MAKER_URL}
        assert P.HF_DAD in html and P.HF_SDF in html and P.REPO_URL in html

    def test_every_link_that_leaves_the_page_says_so(self):
        """The arrow is the only signal a reader gets that a click ends the page."""
        html = build(sdf_inputs=SDF_INPUTS)
        for m in re.finditer(r"<a [^>]*href='(https?://[^']+)'[^>]*>(.*?)</a>", html, re.S):
            assert "class='ext'" in m.group(2), m.group(1)
        for m in re.finditer(r"<a [^>]*href='#[^']+'[^>]*>(.*?)</a>", html, re.S):
            assert "class='ext'" not in m.group(1)

    def test_selected_text_takes_the_accent(self):
        """The page's one piece of interaction colour, rather than the browser's blue."""
        html = build()
        assert re.search(r"::selection\{background:var\(--accent\);color:var\(--surface-0\)\}",
                         html)

    def test_a_link_is_a_typographic_object_not_a_coloured_word(self):
        """Mono against the serif, bold, and underlined in the accent at full strength."""
        html = build(sdf_inputs=SDF_INPUTS)
        rule = re.search(r"\na\{[^}]*\}", html).group(0)
        for want in ("font-family:var(--mono)", "font-weight:600",
                     "color:var(--accent)", "text-decoration-thickness:2px"):
            assert want in rule, rule

    def test_the_buttons_are_not_dragged_along_with_the_links(self):
        """.lbtn, .choice and .cta are actions, and each sets its own font shorthand so
        a bare `a` rule cannot reach them. The outbound buttons are deliberately mono,
        matching the links; they just say so themselves."""
        html = build(sdf_inputs=SDF_INPUTS)
        for cls in ("\n.lbtn{", ".choice{", ".cta{"):
            start = html.index(cls)
            rule = html[start:html.index("}", start)]
            assert "font:" in rule, rule          # its own shorthand beats the `a` rule
        start = html.index("\n.lbtn{")
        assert "var(--mono)" in html[start:html.index("}", start)]

    def test_the_outbound_arrow_is_drawn_not_typed(self):
        """As a glyph U+2197 is a hairline in most faces and a different shape in every
        one, on a page that gets printed and screenshotted."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert "&#8599;" not in html and "\u2197" not in html
        assert "<svg class='ext'" in html
        arrow = re.search(r"<svg class='ext'.*?</svg>", html, re.S).group(0)
        assert "stroke-width='2'" in arrow and "currentColor" in arrow
        assert "aria-hidden='true'" in arrow

    def test_leaving_the_page_leaves_it_in_a_new_tab(self):
        """The chooser's state lives in the URL, so a reader who follows a link out and
        comes back with the back button lands on a page that has closed itself."""
        html = build(sdf_inputs=SDF_INPUTS)
        for m in re.finditer(r"<a [^>]*href='(https?://[^']+)'([^>]*)>", html):
            assert "target='_blank'" in m.group(2), m.group(1)
            assert "rel='noopener noreferrer'" in m.group(2), m.group(1)
        for m in re.finditer(r"<a href='#[^']+'([^>]*)>", html):
            assert "target=" not in m.group(1)

    def test_the_comparison_links_at_the_prompts_not_just_the_records(self):
        """The reader is here to run the pipeline, so each column links at the templates
        that column's pipeline runs, next to its dataset viewer."""
        table = re.search(r"<section id='datasets'>.*?</section>",
                          build(sdf_inputs=SDF_INPUTS), re.S).group(0)
        assert P.PROMPTS_SDF.endswith("/prompts/sdf") and P.PROMPTS_DAD.endswith("/prompts/dad")
        assert table.count(">Prompts</span>") == 2
        assert table.index(P.PROMPTS_SDF) < table.index(P.HF_SDF)  # templates, then data

    def test_the_buttons_are_accent_outlines_not_cream_panels(self):
        """One filled surface was doing duty as a button, a card and a code block at
        once. The controls are the accent now; the cream is just paper."""
        html = build(sdf_inputs=SDF_INPUTS)
        for cls in ("\n.lbtn{", ".choice{"):
            start = html.index(cls)
            rule = html[start:html.index("}", start)]
            assert "background:none" in rule and "var(--accent-edge)" in rule
            assert "border-radius:4px" in rule
            assert "var(--surface-1)" not in rule and "var(--surface-2)" not in rule

    def test_the_two_destinations_are_buttons_with_their_own_mark(self):
        html = build(sdf_inputs=SDF_INPUTS)
        table = re.search(r"<section id='datasets'>.*?</section>", html, re.S).group(0)
        assert table.count("class='lbtn'") == 4  # a dataset and a prompts link per column
        assert R.ICONS["github"][3][:40] in html   # the published silhouette
        assert R.ICONS["hf"][3][:40] in html        # and the real Hugging Face mark
        for svg in re.findall(r"<svg class='ico'.*?</svg>", html, re.S):
            assert "aria-hidden='true'" in svg and "currentColor" in svg


class TestChooser:
    """The page asks which dataset you want and shows that one. The risks are a panel
    that cannot be reached, and a panel that cannot be found."""

    def test_nothing_is_open_until_something_is_chosen(self):
        html = build(sdf_inputs=SDF_INPUTS)
        for pid in ("dad", "sdf"):
            panel = re.search(rf"<section id='{pid}' class='panel'[^>]*>", html).group(0)
            assert "hidden" in panel
            assert f"aria-labelledby='choose-{pid}'" in panel
        assert html.count("aria-selected='false'") == 2
        assert "aria-selected='true'" not in html

    def test_the_buttons_are_a_real_tab_control(self):
        html = build(sdf_inputs=SDF_INPUTS)
        choices = re.search(r"<div class='choices'[^>]*>.*?</div>", html, re.S).group(0)
        assert "role='tablist'" in html
        for pid, label in (("dad", D.SECTION_TITLE), ("sdf", S.SECTION_TITLE)):
            assert f"aria-controls='{pid}'" in choices
            assert f"id='choose-{pid}'" in choices
            assert label in choices

    def test_the_buttons_are_two_names_and_nothing_else(self):
        """What each dataset is and how big it is are in the comparison directly above.
        Repeating both under each button made them hard to read as buttons."""
        section = re.search(r"<section id='explore'>.*?</section>",
                            build(sdf_inputs=SDF_INPUTS), re.S).group(0)
        after_heading = strip_tags(section).split("results")[1]
        assert after_heading.split() == [*S.SECTION_TITLE.split(), "&darr;",
                                         *D.SECTION_TITLE.split(), "&darr;"]

    def test_the_choice_lines_up_with_what_is_being_chosen(self):
        """40rem centred is exactly the two dataset columns above (2 x 20rem), so each
        button sits under its own column instead of off to the left with the prose."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert re.search(r"\.choices\{[^}]*width:min\(100%,40rem\)", html)
        assert re.search(r"\.choices\{[^}]*margin:0 auto", html)
        assert re.search(r"#explore h2\{[^}]*text-align:center", html)

    def test_each_report_ends_by_offering_the_other(self):
        """The dataset a reader did not choose is one click from the end of the one
        they did, which is the whole reason a chooser is allowed to hide things."""
        html = build(sdf_inputs=SDF_INPUTS)
        sdf_panel = html[html.index("<section id='sdf'"):html.index("<section id='dad'")]
        assert "class='panel-cta'" in sdf_panel
        assert f"data-panel='dad'>{D.SECTION_TITLE} example" in sdf_panel
        dad_panel = html[html.index("<section id='dad'"):]
        assert f"data-panel='sdf'>{S.SECTION_TITLE} example" in dad_panel
        # A filled button, not a text link: it is the one thing the page asks for here.
        rule = html[html.index("\n.cta{"):html.index("}", html.index("\n.cta{"))]
        assert "background:var(--accent)" in rule and "color:var(--surface-0)" in rule

    def test_hiding_a_panel_actually_hides_it(self):
        """A panel is a <section>, and section{display:grid} beats the browser's own
        [hidden] rule, so the override has to be written down."""
        assert ".panel[hidden]{display:none}" in build()

    def test_a_printed_page_carries_both_reports(self):
        """Whichever is open on screen, a PDF of this page is the whole thing."""
        block = build()[build().find("@media print"):]
        assert ".panel[hidden]{display:block!important}" in block
        assert ".choices" in block  # the buttons themselves are not printed

    def test_the_chooser_reads_the_url_so_deep_links_survive(self):
        """The dataset card links to #dad and #sdf. Without this the link lands on a
        page with both reports closed."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert "hashchange" in html
        assert "closest('.panel')" in html  # #dad-weak opens the report it lives in

    def test_the_deep_link_waits_for_the_page_to_finish_laying_out(self):
        """The hero is a multi-megabyte data URI. Scrolling to a deep-linked beat at
        parse time put the reader ~2,200px away from it once the image claimed its
        space; measured in Chromium, and fixed by deferring to load."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert "window.addEventListener('load'" in html
        assert "readyState==='complete'" in html


class TestComparisonTable:
    def test_the_rows_are_what_a_lab_needs_to_run_it(self):
        """Six rows, in one pass. Dates, model ids and the composition spread went to
        the report that goes into them: a reader here is deciding whether to run the
        pipeline, not shopping for a dataset."""
        html = build(sdf_inputs=SDF_INPUTS)
        table = re.search(r"<section id='datasets'>.*?</section>", html, re.S).group(0)
        labels = re.findall(r"<th class='cmp-k' scope='row'>([^<]*)</th>", table)
        assert labels == ["what it is for", "one record is", "prompt templates",
                          "records"]  # the prompts before the data
        text = strip_tags(table)
        for gone in ("July 2026", "claude-", "domains", "taxa groups", "languages",
                     "licence"):
            assert gone not in text, gone

    def test_the_labels_are_one_line_each_flush_against_the_pair(self):
        """An index down the side of the comparison; one that wraps, or that floats away
        from the columns it indexes, stops reading as one. Both properties have to
        out-specify `.cmp th`, which sets the alignment and padding for every cell."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert re.search(r"\.cmp th\.cmp-k\{[^}]*text-align:right", html)
        assert re.search(r"\.cmp-k\{[^}]*white-space:nowrap", html)

    def test_each_figure_carries_the_way_to_the_thing_it_counts(self):
        """The prompt count is the datapoint for someone who wants to generate their own
        data — read off the run's own inputs/prompts snapshot — and it sits next to the
        link to those prompts. Same for the record counts and the published sample."""
        html = build(sdf_inputs=SDF_INPUTS)
        table = re.search(r"<section id='datasets'>.*?</section>", html, re.S).group(0)
        rows = dict(re.findall(r"<th class='cmp-k' scope='row'>([^<]*)</th>(.*?)</tr>",
                               table, re.S))
        assert strip_tags(rows["records"]).split()[0] == "100"
        assert strip_tags(rows["prompt templates"]).split()[0] == "4"
        assert P.PROMPTS_SDF in rows["prompt templates"] and P.HF_SDF in rows["records"]
        assert P.PROMPTS_DAD in rows["prompt templates"] and P.HF_DAD in rows["records"]
        assert "<tfoot>" not in table

    def test_a_run_that_kept_no_prompt_snapshot_says_so(self):
        html = build(dad_inputs={**DAD_INPUTS, "n_prompt_templates": None})
        table = re.search(r"<section id='datasets'>.*?</section>", html, re.S).group(0)
        assert "8 templates" not in table

    def test_synthetic_documents_comes_first_everywhere(self):
        """One order for the whole page: the comparison, the chooser and the panels."""
        html = build(content=shipped_content(), sdf_inputs=SDF_INPUTS)
        assert re.findall(r"<span class='cmp-name'>([^<]*)</span>", html) == [
            S.SECTION_TITLE, D.SECTION_TITLE]
        assert re.findall(r"data-panel='(\w+)' id='choose", html) == ["sdf", "dad"]
        assert html.index("<section id='sdf'") < html.index("<section id='dad'")

    def test_the_mastheads_carry_the_names_and_what_each_one_is(self):
        """The two items that used to sit in the intro live here now, and the config ids
        do not: a masthead is a name, not a filename."""
        html = build(content=shipped_content(), sdf_inputs=SDF_INPUTS)
        head = re.search(r"<thead>.*?</thead>", html, re.S).group(0)
        assert f"<span class='cmp-name'>{D.SECTION_TITLE}</span>" in head
        assert f"<span class='cmp-name'>{S.SECTION_TITLE}</span>" in head
        assert "Examples of an AI reasoning well" in head
        assert "<code>dad</code>" not in head and "<code>sdf</code>" not in head

    def test_the_two_columns_are_centred_on_the_page_not_on_the_prose(self):
        """The comparison centres itself with left:50% + translateX(-50%), which is
        50% OF ITS GRID AREA. In the default `section>*` track that area is the 38rem
        prose column, and the whole table lands ~5.75rem left of the hero centred above
        it; only the full-bleed track spans the main column, whose centre is the page's.
        This is the rule that was missing."""
        html = build(sdf_inputs=SDF_INPUTS)
        bleed = re.search(r"section>figure,[^{]*\{grid-column:text-start/full-end\}", html)
        assert bleed and "section>.cmp-wrap" in bleed.group(0)
        wrap = re.search(r"\n\.cmp-wrap\{[^}]*\}", html).group(0)  # the rule, not the selector list
        assert "left:50%" in wrap and "translateX(-50%)" in wrap

    def test_the_labels_hang_off_the_left_of_the_centred_pair(self):
        """What is centred is the PAIR, not the table: the table is pushed right by
        half the wrapper minus one column minus the labels, which puts the pair's
        midpoint on the wrapper's midpoint and leaves the labels outside it, to the
        left. Stated as arithmetic rather than left to flex free space or auto margins
        to arrive at — two earlier attempts at this centred the whole table instead."""
        html = build(sdf_inputs=SDF_INPUTS)
        rule = re.search(r"\n\.cmp\{[^}]*\}", html).group(0)
        assert "margin-left:calc(50% - var(--cmp-col) - var(--cmp-label))" in rule
        assert "width:calc(var(--cmp-label) + 2*var(--cmp-col))" in rule
        assert "margin-left:-" not in rule and "margin:0 auto" not in rule
        wrap = re.search(r"\n\.cmp-wrap\{[^}]*\}", html).group(0)
        assert "display:flex" not in wrap  # the pair's position must not depend on it

    def test_the_three_column_widths_are_set_where_fixed_layout_reads_them(self):
        """table-layout:fixed takes its widths from the first row only, so the corner
        cell has to carry the label width; .cmp-k sits in the body rows, which fixed
        layout never consults."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert re.search(r"\.cmp \.cmp-corner\{[^}]*width:var\(--cmp-label\)", html)
        assert re.search(r"\.cmp thead th\{[^}]*width:var\(--cmp-col\)", html)
        assert re.search(r"\.cmp-wrap\{--cmp-label:[\d.]+rem;--cmp-col:[\d.]+rem", html)


    def test_the_label_column_carries_no_rules(self):
        """The row rules belong to the two columns being compared; the labels are an
        index down the side, not a third column. `.cmp th` sets the border, so the
        override has to out-specify it rather than merely follow it."""
        html = build(sdf_inputs=SDF_INPUTS)
        assert re.search(r"\.cmp th\.cmp-k\{[^}]*border-bottom:0", html)

    def test_no_sdf_run_leaves_an_honest_column(self):
        """A cell that quietly shows nothing reads as a dataset with no properties."""
        html = build()
        table = re.search(r"<section id='datasets'>.*?</section>", html, re.S).group(0)
        assert "not published yet" in strip_tags(table)
        assert P.HF_SDF in table  # the viewer link still works

    def test_a_run_without_deals_says_nothing_rather_than_guessing(self):
        html = build(dad_inputs={**DAD_INPUTS, "deals": []})
        assert "domains" not in strip_tags(html)


class TestSdfPlaceholder:
    def test_headline_figures_come_from_the_audit(self):
        text = strip_tags(build(sdf_inputs=SDF_INPUTS))
        assert "100 documents" in text
        assert "23 effectively distinct documents" in text

    def test_its_weaknesses_are_derived_too(self):
        """audit_sdf.py prints its verdicts instead of recording them, so this report
        re-applies the eval's own thresholds. 12% truncated is BAD on them."""
        html = build(sdf_inputs=SDF_INPUTS)
        section = html[html.index("<section id='sdf'"):]
        assert "12% of documents are truncated" in strip_tags(section)
        assert "claude_code" in section  # the backend provenance rule
        assert "chip bad'>BAD" in section

    def test_a_clean_run_earns_no_rows(self):
        clean = {"audit": {"n_docs": 500, "length": {"truncated": 0, "truncated_frac": 0.0},
                           "composition": {"top_type_share": 0.1},
                           "near_dups": {"0.9": 0.0}, "openings": {"formulaic_frac": 0.0}},
                 "manifest": {"config": {"backend": "api"}}, "run_id": "r"}
        html = build(sdf_inputs=clean)
        section = html[html.index("<section id='sdf'"):html.index("<section id='dad'")]
        assert "Where it is weak" not in section

    def test_a_flagged_templating_pattern_is_a_bad_row(self):
        audit = {**SDF_AUDIT, "patterns": [{"pattern": "Refuse-then-alternative",
                                            "prevalence": 0.42, "is_defect": True,
                                            "flagged": True}]}
        warnings = S.derived_warnings(audit, SDF_MANIFEST, S.facts(audit))
        assert any(sev == "BAD" and "Refuse-then-alternative" in w for sev, w in warnings)

    def test_composition_is_read_from_the_field_names_the_audit_writes(self):
        """An earlier version read composition.languages/types, which no audit has ever
        written, so both figures rendered empty and nobody noticed."""
        f = S.facts(SDF_AUDIT, SDF_DIVERSITY, SDF_MANIFEST)
        assert f["n_languages"] == 2 and f["n_types"] == 15
        assert S.facts({"composition": {"languages": {"en": 1}, "types": {"a": 1}}}) \
            .get("n_languages") is None

    def test_without_a_run_the_section_says_so(self):
        html = build()
        section = html[html.index("<section id='sdf'"):]
        assert "No audit output was supplied" in strip_tags(section)
        assert P.HF_SDF in section


class TestBrevity:
    """The page's whole brief is a reader with forty seconds, so these read the
    SHIPPED prose files rather than the fixtures."""

    def page(self):
        return build(content=shipped_content(), sdf_inputs=SDF_INPUTS)

    def test_at_most_two_deks(self):
        """Every aphoristic two-beat line under a heading came out. Two is the
        allowance, so adding a third means taking one away."""
        assert self.page().count("class='dek'") <= 2

    def test_the_prose_has_a_ceiling(self):
        """The two pages this replaced carried ~3,400 words of authored prose between
        them. A regression here is prose growing back, which is the failure mode this
        page was rebuilt to fix."""
        assert C.editorial_words(self.page()) < 1800

    def test_the_method_is_credited_once_where_the_reader_starts(self):
        """The Teaching Claude Why grounding was on both of the pages this replaces, and
        again inside the report. It belongs in the intro, once."""
        html = self.page()
        assert html.count("alignment.anthropic.com") == 1
        hero = re.search(r"<header class='hero'>.*?</header>", html, re.S).group(0)
        assert "alignment.anthropic.com" in hero

    def test_the_page_does_not_argue_a_third_route(self):
        """The belief-implantation comparison is a decision record, not something a
        reader deciding whether to use the data needs."""
        text = strip_tags(self.page()).lower()
        assert "belief implantation" not in text
        assert "third route" not in text

    def test_the_shipped_prose_files_satisfy_the_id_contract(self):
        """A section renamed in a module and not in its prose file, or a prose block
        left behind after a rename, is a build error rather than a silent hole."""
        assert set(shipped_content()) == set(P.CONTENT_IDS + D.CONTENT_IDS + S.CONTENT_IDS)


class TestFacts:
    def test_unknown_placeholder_in_page_prose_is_a_build_error(self):
        with pytest.raises(KeyError, match="unknown fact"):
            build(content=content(intro="A {{n}}-example run."))

    def test_the_page_itself_interpolates_nothing(self):
        """Every figure on the page is rendered by a section from its run's facts, so
        the page's own prose has nothing to interpolate and must not try."""
        assert P.PAGE_FACTS == {}

    def test_a_date_survives_a_manifest_that_has_none(self):
        assert P._date({}) == "—"
        assert P._date({"created_at": "not-a-date"}) == "not-a-date"
        assert P._date({"created_at": "2026-07-01T09:00:00"}) == "1 July 2026"
