"""Tests for report/common.py — the plumbing both report pages share.

The two rules the reports rest on are enforced here rather than in either page, so
they are tested here: an unknown or duplicated prose id is a build error, and an
unresolved ``{{placeholder}}`` is a build error. Both exist so a page cannot silently
go stale against the run it claims to describe.

Fully offline — nothing here touches the network or the API.
"""

import pytest

from report import common as C

IDS = ("title", "lede", "body")


def _file(tmp_path, name, blocks):
    path = tmp_path / name
    path.write_text("".join(f"<!-- id: {k} -->\n{v}\n\n" for k, v in blocks.items()),
                    encoding="utf-8")
    return path


class TestParseContent:
    def test_round_trips_sections(self):
        text = "".join(f"<!-- id: {k} -->\n{k.upper()} body\n\n" for k in IDS)
        parsed = C.parse_content(text, IDS)
        assert parsed["title"] == "TITLE body"
        assert set(parsed) == set(IDS)

    def test_unknown_id_raises(self):
        text = "".join(f"<!-- id: {k} -->\nx\n\n" for k in IDS)
        with pytest.raises(ValueError, match="unknown section id"):
            C.parse_content(text + "<!-- id: nonsense -->\nx", IDS)

    def test_missing_id_raises(self):
        text = "".join(f"<!-- id: {k} -->\nx\n\n" for k in IDS if k != "body")
        with pytest.raises(ValueError, match="missing section id"):
            C.parse_content(text, IDS)

    def test_no_markers_raises(self):
        with pytest.raises(ValueError, match="no '<!-- id"):
            C.parse_content("just prose", IDS)


class TestLoadContent:
    def test_merges_two_files_into_one_namespace(self, tmp_path):
        a = _file(tmp_path, "a.md", {"title": "T", "lede": "L"})
        b = _file(tmp_path, "b.md", {"body": "B"})
        merged = C.load_content([a, b], IDS)
        assert merged == {"title": "T", "lede": "L", "body": "B"}

    def test_rejects_a_duplicate_id_across_files(self, tmp_path):
        """Two files may not both own a section: moving a block from one prose file to
        another has to be a rename, never a silent shadowing."""
        a = _file(tmp_path, "a.md", {"title": "T", "lede": "L", "body": "B"})
        b = _file(tmp_path, "b.md", {"body": "other"})
        with pytest.raises(ValueError, match="defined in both"):
            C.load_content([a, b], IDS)

    def test_missing_id_across_all_files_raises(self, tmp_path):
        a = _file(tmp_path, "a.md", {"title": "T", "lede": "L"})
        with pytest.raises(ValueError, match="missing section id"):
            C.load_content([a], IDS)


class TestFill:
    def test_resolves_known_facts(self):
        assert C.fill("n = {{n}}", {"n": 39}) == "n = 39"

    def test_unknown_placeholder_is_a_build_error(self):
        with pytest.raises(KeyError, match="unknown fact"):
            C.fill("{{not_a_fact}}", {"n": 2})

    def test_degraded_clause_keeps_the_sentence(self):
        """A run without the paid pass renders the caveat where the finding would be."""
        f = {"delivery_clause": "Judged delivery was not measured on this run"}
        assert C.fill("{{delivery_clause}}.", f).endswith("on this run.")


class TestProvenanceWarnings:
    def test_non_api_backend_is_flagged(self):
        w = C.provenance_warnings({"config": {"backend": "bedrock"}})
        assert any("bedrock" in t for _, t in w)

    def test_api_backend_is_not_flagged(self):
        assert not any("faithful mode" in t
                       for _, t in C.provenance_warnings({"config": {"backend": "api"}}))

    def test_claude_code_backend_is_the_severe_case(self):
        w = C.provenance_warnings({"config": {"backend": "claude_code"}})
        assert ("BAD", ) == tuple({sev for sev, _ in w})

    def test_dirty_tree_and_small_n(self):
        w = C.provenance_warnings({"git_dirty": True, "config": {}}, n=39)
        assert any("uncommitted" in t for _, t in w)
        assert any("n = 39" in t for _, t in w)

    def test_large_n_is_not_flagged(self):
        assert not any("n = " in t for _, t in C.provenance_warnings({"config": {}}, n=500))


class TestAuditVerdictWarnings:
    def test_reads_bad_and_ok_rows(self):
        audit = {"sections": [{"title": "Stance", "rows": [
            {"label": "moralizes", "value": "40%", "verdict": "BAD", "note": "(fault)"},
            {"label": "defers", "value": "100%", "verdict": "GOOD"}]}]}
        w = C.audit_verdict_warnings(audit)
        assert len(w) == 1 and w[0][0] == "BAD" and "moralizes" in w[0][1]

    def test_an_audit_without_verdict_rows_yields_nothing(self):
        """This is what an SDF audit looks like today: it prints its verdicts rather
        than recording them, so that page has to derive its own thresholds."""
        assert C.audit_verdict_warnings({"n_docs": 100, "near_dups": {"0.9": 0.0}}) == []


class TestWarningsTable:
    def test_bads_lead_and_the_rest_are_counted_not_dropped(self):
        ws = [("OK", f"minor {i}") for i in range(6)] + [("BAD", "the big one")]
        html = C.warnings_table(ws, inline=2)
        assert html.index("the big one") < html.index("minor 0")
        assert "4 more findings at this level" in html
        for i in range(6):
            assert f"minor {i}" in html  # collapsed, never removed

    def test_short_lists_need_no_drawer(self):
        assert "<details" not in C.warnings_table([("BAD", "x"), ("OK", "y")])

    def test_no_warnings_is_empty(self):
        assert C.warnings_table([]) == ""


class TestCost:
    COSTS = [{"stage": "prompt_draft", "cost_usd": 0.5, "model": "m1"},
             {"stage": "prompt_draft", "cost_usd": 0.25, "model": "m1"},
             {"stage": "surprise_stage", "cost_usd": 1.0, "model": "m2"}]

    def test_aggregates_by_stage(self):
        agg = C.costs_by_stage(self.COSTS)
        assert agg["prompt_draft"]["calls"] == 2
        assert agg["prompt_draft"]["cost"] == 0.75

    def test_untagged_records_are_not_lost(self):
        assert C.costs_by_stage([{"cost_usd": 1.0}])["(untagged)"]["calls"] == 1

    def test_unlabelled_stages_are_appended_not_dropped(self):
        html = C.stage_cost_table(self.COSTS, (("prompt_draft", "1b · draft"),))
        assert "1b · draft" in html and "surprise_stage" in html

    def test_no_costs_is_empty(self):
        assert C.stage_cost_table([], (("a", "A"),)) == ""


class TestMetaLine:
    def test_names_the_run_the_commit_and_the_backend(self):
        line = C.meta_line(run_id="r-1", manifest={"git_commit": "abcdef1234", "git_dirty": True,
                                                   "config": {"backend": "bedrock"}})
        assert "r-1" in line and "abcdef12" in line and "bedrock" in line
        assert "uncommitted changes" in line
