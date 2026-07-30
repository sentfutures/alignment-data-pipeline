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
        for sid, _ in H.TOC:
            assert f"id='{sid}'" in html

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
        """The hub is the entry point; it must build before either report does."""
        html = H.build(content=content())
        assert "No run built yet" in html
        assert "Report in preparation" in html

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
