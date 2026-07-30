"""Tests for report/hub.py — the page that introduces both corpora.

The risk here is different from the pipeline pages. The hub has no run of its own, so
it interpolates nothing and cannot go stale; what it CAN do is show a link to a page
that does not exist, or a card whose numbers disagree with the report they point at.
Both are tested.

Fully offline.
"""

import re

from report import hub as H

CONTENT = {k: f"Prose for {k}." for k in H.CONTENT_IDS}
CONTENT["title"] = "Two corpora"
CONTENT["lede"] = "What they are."
CONTENT["dad_card"] = "### Dilemmas\n\nChat data."
CONTENT["sdf_card"] = "### Documents\n\nPretraining prose."

DAD_AUDIT = {
    "n_prompts": 40,
    "valuable_welfare_considerations": {"available": True,
                                       "parent": {"pipeline": 17.07, "plain": 12.54}},
    "moral_patient_reasons": {"n": 39, "pipeline": {"n": 39}, "plain": {"n": 39}},
    "delivery": {"pipeline_mean": 7.03, "plain_mean": 7.85},
    "response_lengths": {"n": 39},
}
DAD_COSTS = [{"stage": "prompt_draft", "cost_usd": 14.04, "model": "m"}]

SDF_AUDIT = {"n_docs": 100, "composition": {"languages": {"en": 29, "zh": 12},
                                            "types": {"a": 1, "b": 2, "c": 3}},
             "near_dups": {"0.9": 0.0}}


def content(**overrides):
    return {**CONTENT, **overrides}


def strip_tags(html):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))


class TestHub:
    def test_builds_every_section(self):
        html = H.build(content=content())
        for sid in ("why", "corpora", "limits"):
            assert f"id='{sid}'" in html

    def test_is_a_landing_page_not_an_article(self):
        """No contents rail: the page's job is the hero and the two links out of it, and
        a rail for three short sections is furniture."""
        html = H.build(content=content())
        assert "<nav class='rail'" not in html
        assert "class='shell solo'" in html
        assert H.TOC == []

    def test_the_hero_is_a_title_a_subtitle_and_two_buttons(self):
        """The cards used to sit inside the masthead. The hero's job is the two
        destinations; the cards describe them further down."""
        html = H.build(content=content(), dad_audit=DAD_AUDIT, dad_href="dad.html")
        header = re.search(r"<header class='top'>.*?</header>", html, re.S).group(0)
        assert "<h1>" in header and "class='lede'" in header
        assert "class='btns'" in header
        assert "class='cards'" not in header
        assert header.index("class='lede'") < header.index("class='btns'")

    def test_the_cards_live_in_the_corpora_section(self):
        html = H.build(content=content(), dad_audit=DAD_AUDIT, dad_href="dad.html")
        corpora = re.search(r"<section id='corpora'>.*?</section>", html, re.S).group(0)
        assert "class='cards'" in corpora

    def test_the_third_route_reads_after_the_cards(self):
        """The paragraph on the route we turned down only lands once the reader has seen
        what the two we took are, so the prose splits around the cards."""
        html = H.build(content=content(corpora="> dek line\n\nfirst para\n\nthird route"),
                       dad_audit=DAD_AUDIT, dad_href="dad.html")
        corpora = re.search(r"<section id='corpora'>.*?</section>", html, re.S).group(0)
        assert corpora.index("first para") < corpora.index("class='cards'")
        assert corpora.index("class='cards'") < corpora.index("third route")

    def test_no_eyebrow(self):
        assert "eyebrow" not in H.build(content=content())

    def test_the_landing_page_ranges_left(self):
        """One left edge shared by the title, the buttons, the headings and the body
        copy. This replaces test_the_hero_and_headings_are_centred_but_body_copy_is_not:
        the centred hero was tried and rejected, so the spec changed, not the code's
        luck. The column is still centred in the viewport — it is the type inside it
        that is not."""
        html = H.build(content=content())
        solo = re.findall(r"\.shell\.solo[^{]*\{[^}]*\}", html)
        assert solo, "no landing-layout rules found"
        assert not any("text-align:center" in rule for rule in solo)
        assert not any("margin:0 auto" in rule for rule in solo)
        assert not re.search(r"\.btns\{[^}]*justify-content:center", html)

    def test_the_content_column_matches_the_report_pages(self):
        """The rail column is reserved on every page and the landing page just leaves it
        empty, so the text lands on the same left edge at the same width on both. That is
        what lets the contents appear BESIDE the prose when you click into a report,
        instead of shoving it sideways. Any .shell.solo rule that re-declares the grid, or
        moves main out of column 2, breaks it."""
        html = H.build(content=content())
        assert ".shell{display:grid;grid-template-columns:12.5rem minmax(0,50rem)" in html
        assert "main{grid-column:2" in html
        for rule in re.findall(r"\.shell\.solo[^{]*\{[^}]*\}", html):
            assert "grid-template-columns" not in rule, rule
            assert "grid-column" not in rule, rule
            assert "justify-content" not in rule, rule

    def test_landing_prose_uses_the_same_measure_as_a_report(self):
        """Body copy on both pages sits on the 38rem text track; the cards bleed to the
        full column exactly as a report's figures do."""
        html = H.build(content=content(), dad_audit=DAD_AUDIT, dad_href="dad.html")
        assert "section{display:grid;grid-template-columns:[text-start] minmax(0,38rem)" in html
        assert re.search(r"section>figure,[^{]*section>\.cards\{\s*grid-column:text-start/full-end",
                         html)

    def test_is_self_contained(self):
        html = H.build(content=content())
        assert not re.search(r"<(img|link|iframe)\b", html)
        assert not re.search(r"<script[^>]*\ssrc=", html)
        assert "@import" not in html and "url(" not in html

    def test_is_light_mode_only(self):
        html = H.build(content=content())
        assert "color-scheme:only light" in html
        assert "prefers-color-scheme" not in html

    def test_renders_without_any_run(self):
        """The landing page is the entry point; it must build before either report does."""
        html = H.build(content=content())
        assert "No run built yet" in html
        assert "Report in preparation" in html

    def test_an_unbuilt_report_gets_a_button_that_cannot_be_clicked(self):
        """A dead link is worse than a visibly unavailable destination, so the button for
        a report nobody has built is a span rather than an anchor."""
        html = H.build(content=content(), dad_audit=DAD_AUDIT, dad_href="dad.html")
        btns = re.search(r"<div class='btns'>.*?</div>", html, re.S).group(0)
        assert "<a class='btn' href='dad.html'>" in btns
        assert "<span class='btn off'>" in btns
        assert btns.count("href=") == 1
        assert "report in preparation" in btns

    def test_the_dad_card_numbers_come_from_the_dad_pages_own_facts(self):
        """One facts() for both, so the hub and the report cannot disagree about a
        headline."""
        html = H.build(content=content(), dad_audit=DAD_AUDIT, dad_costs=DAD_COSTS,
                       dad_href="dad.html")
        text = strip_tags(html)
        assert "39 examples measured" in text
        assert "+36%" in text
        assert "delivery 7.0 against the control's 7.8" in text
        assert "href='dad.html'" in html

    def test_no_sdf_run_means_no_sdf_link(self):
        """A card that links to a page nobody has built is worse than an honest gap."""
        html = H.build(content=content(), dad_audit=DAD_AUDIT, dad_href="dad.html")
        assert "sdf" not in html.lower().replace("sdf_card", "")
        assert "Report in preparation" in html

    def test_sdf_numbers_render_when_a_run_is_given(self):
        html = H.build(content=content(), sdf_audit=SDF_AUDIT, sdf_href="sdf.html")
        text = strip_tags(html)
        assert "100 documents" in text
        assert "2 languages" in text
        assert "href='sdf.html'" in html

    def test_a_placeholder_in_hub_prose_is_a_build_error(self):
        """The hub has no run of its own, so a number in its prose could not be checked
        against anything. Reaching for one fails the build rather than shipping it."""
        try:
            H.build(content=content(why="A {{n}}-example run."))
        except KeyError as e:
            assert "unknown fact" in str(e)
        else:
            raise AssertionError("a placeholder in hub prose should fail the build")
