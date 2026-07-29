"""Tests for evals/publish_hf.py — staging, card generation, and the
Hub-upload chokepoints (stubbed via the stub_hf fixture; never touches
huggingface_hub or the network)."""

import json
import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from evals import publish_hf

COMPLIANCE = {"judged": 100, "clean_documents": 98, "clean_frac": 0.98}
CARD_FIDELITY = {
    "judged": 99,
    "honoured": {"document_type": 96, "resolution": 65, "centrality": 84},
    "by_card_frac": {"document_type": 0.97, "resolution": 0.657, "centrality": 0.848},
}
DIVERSITY = {"n_records": 477, "vendi": {"score": 34.45, "ratio": 0.0722},
             "mean_pairwise_cosine": 0.3709}
REALISM = {"n": 78, "layer5_mean": 8.487, "blind_same_rubric_mean": 5.782, "mean_drop": 2.705}
VENDI_CURVE = {"proj": {"1000": {"power": 50.1, "log": 48.2}, "5000": {"power": 90.4, "log": 85.1}}}
AUDIT_REPORT = {
    "n_docs": 477,
    "composition": {"language": {
        "English": 139, "Spanish": 48, "Mandarin Chinese": 57, "Urdu": 10,
    }},
}
REPORT_CONTENT = {"title": "SDF corpus audit — 477 documents", "subtitle": "A test subtitle."}
MANIFEST = {
    "run_id": "2026-07-25_15-57_fullscale-500-opus5",
    "label": "fullscale-500-opus5",
    "git_commit": "4abd78b",
    # Shaped like the real published run: the top-level model is the default,
    # while the stage that matters was overridden to Opus.
    "model": "claude-sonnet-5",
    "config": {"backend": "claude_code",
               "sdf": {"rewrite_model": "claude-opus-5", "n_prompts": 500}},
}

KNOWN_AUDIT_FILES = {
    "audit_report.json": AUDIT_REPORT,
    "compliance_report.json": COMPLIANCE,
    "card_fidelity_report.json": CARD_FIDELITY,
    "diversity_report.json": DIVERSITY,
    "realism_ablation.json": REALISM,
    "vendi_curve.json": VENDI_CURVE,
    "report_content.json": REPORT_CONTENT,
}


def make_run_dir(tmp_path, pipeline="sdf", docs=3, audit_files=None, manifest=MANIFEST,
                  include_html=True, extra_audit_files=None):
    """Build a fake run directory with the given audit files present.

    audit_files=None means "all six known + report_content.json + html";
    pass a subset of KNOWN_AUDIT_FILES' keys to omit others.
    """
    run_dir = tmp_path / "runs" / "2026-07-25_15-57_fullscale-500-opus5"
    final = run_dir / "final"
    final.mkdir(parents=True)
    corpus_name = "sdf_corpus.jsonl" if pipeline == "sdf" else "dad_corpus.jsonl"
    lines = [json.dumps({"doc_id": f"d{i}", "content": f"document {i}"}) for i in range(docs)]
    (final / corpus_name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    if manifest is not None:
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    names = list(KNOWN_AUDIT_FILES) if audit_files is None else audit_files
    if names or include_html or extra_audit_files:
        audit = run_dir / "audit"
        audit.mkdir()
        for name in names:
            (audit / name).write_text(json.dumps(KNOWN_AUDIT_FILES[name]), encoding="utf-8")
        if include_html:
            (audit / "corpus_report.html").write_text("<html>report</html>", encoding="utf-8")
        for name, content in (extra_audit_files or {}).items():
            (audit / name).write_text(json.dumps(content), encoding="utf-8")

    return run_dir, corpus_name


def _tag_for(corpus_name):
    return "sdf" if corpus_name.startswith("sdf") else "dad"


def _stage(run_dir, corpus_name, staging_dir):
    """stage_run for the pipeline implied by corpus_name, plus the per-pipeline
    dataset dir it staged into (what build_metrics_rows/build_card now read)."""
    tag = _tag_for(corpus_name)
    staged = publish_hf.stage_run(run_dir, corpus_name, staging_dir, tag)
    return staged, staging_dir / tag


def _one_card(dataset_dir, staged, content=None, license_id="cc-by-4.0",
              pretty_name="test-datasets"):
    """build_card for a single dataset — the common case in these tests."""
    return publish_hf.build_card(
        [{"pipeline": staged["pipeline"], "dir": dataset_dir,
          "staged": staged, "content": content}],
        license_id, pretty_name,
    )


class TestResolveRunDir:
    def test_sdf_run(self, tmp_path):
        run_dir, _ = make_run_dir(tmp_path, pipeline="sdf")
        resolved, corpus_name = publish_hf.resolve_corpus_file(str(run_dir))
        assert resolved == run_dir
        assert corpus_name == "sdf_corpus.jsonl"

    def test_dad_run(self, tmp_path):
        run_dir, _ = make_run_dir(tmp_path, pipeline="dad")
        _, corpus_name = publish_hf.resolve_corpus_file(str(run_dir))
        assert corpus_name == "dad_corpus.jsonl"

    def test_missing_corpus_raises(self, tmp_path):
        empty = tmp_path / "runs" / "empty-run"
        empty.mkdir(parents=True)
        with pytest.raises(SystemExit):
            publish_hf.resolve_corpus_file(str(empty))

    def test_not_a_directory_raises(self, tmp_path):
        with pytest.raises(SystemExit):
            publish_hf.resolve_corpus_file(str(tmp_path / "nope"))


class TestStageRun:
    def test_stages_corpus_manifest_and_audit_files(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = tmp_path / "staged"
        staged, dataset_dir = _stage(run_dir, corpus_name, staging_dir)

        assert staged["corpus_file"] == corpus_name
        assert staged["pipeline"] == "sdf"
        assert staged["n_docs"] == 3
        assert staged["manifest_file"] == "run_manifest.json"
        # everything lands under <pipeline>/, never at the staging root — that
        # subdir is what lets one repo hold both corpora as separate configs
        assert dataset_dir == staging_dir / "sdf"
        assert (dataset_dir / corpus_name).exists()
        assert (dataset_dir / "run_manifest.json").exists()
        assert not (staging_dir / corpus_name).exists()
        assert not (staging_dir / "run_manifest.json").exists()
        # report_content.json is editorial and must never be staged/uploaded
        assert "report_content.json" not in staged["audit_files"]
        assert not (dataset_dir / "audit" / "report_content.json").exists()
        assert set(staged["audit_files"]) == {
            "audit_report.json", "compliance_report.json", "card_fidelity_report.json",
            "diversity_report.json", "realism_ablation.json", "vendi_curve.json",
            "corpus_report.html",
        }

    def test_no_audit_dir_is_fine(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, audit_files=[], include_html=False)
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        assert staged["audit_files"] == []

    def test_no_manifest_is_fine(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, manifest=None, audit_files=[],
                                            include_html=False)
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        assert staged["manifest_file"] is None

    def test_unknown_audit_file_is_staged_anyway(self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, audit_files=[], include_html=False,
            extra_audit_files={"custom_eval.json": {"foo": "bar"}},
        )
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        assert staged["audit_files"] == ["custom_eval.json"]

    def test_jsonl_audit_files_are_staged(self, tmp_path):
        """Regression: the glob only matched *.json/*.html, silently dropping
        evals/audit_dad.py's audit/tic_candidates.jsonl and
        audit/reason_failures.jsonl for DAD runs."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", audit_files=[], include_html=False,
            extra_audit_files={
                "tic_candidates.jsonl": {"phrase": "so I've been thinking"},
                "reason_failures.jsonl": {"reason": "example"},
            },
        )
        staging_dir = tmp_path / "staged"
        staged, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        assert set(staged["audit_files"]) == {"tic_candidates.jsonl", "reason_failures.jsonl"}
        assert (dataset_dir / "audit" / "tic_candidates.jsonl").exists()

    def test_reused_staging_dir_is_cleared_of_stale_files(self, tmp_path):
        """Regression: a --staging-dir reused across two invocations (e.g. after
        fixing a typo'd --input) must reflect only the LATEST run — leftover
        files from an earlier call must not ride along into the upload.

        Under the per-pipeline layout the staging ROOT is wiped, so an earlier
        run's whole `<pipeline>/` dir goes with it. That's intended: the
        sibling dataset is regenerated from the Hub (fetch_sibling), never
        from whatever happens to be left in a local staging dir."""
        staging_dir = tmp_path / "staged"

        run_a, corpus_a = make_run_dir(tmp_path / "a", audit_files=["compliance_report.json"])
        _, dir_a = _stage(run_a, corpus_a, staging_dir)
        assert (dir_a / "audit" / "compliance_report.json").exists()

        run_b, corpus_b = make_run_dir(
            tmp_path / "b", pipeline="dad", audit_files=["audit_report.json"], include_html=False,
        )
        staged_b, dir_b = _stage(run_b, corpus_b, staging_dir)

        assert staged_b["audit_files"] == ["audit_report.json"]
        # run A's entire pipeline dir is gone, corpus and audit alike
        assert not dir_a.exists()
        assert not (staging_dir / "sdf").exists()
        assert (dir_b / corpus_b).exists()

    def test_staging_dir_equal_to_run_dir_is_rejected(self, tmp_path):
        """Regression: rmtree(staging_dir) must never fire before verifying
        staging_dir doesn't equal or contain run_dir — otherwise a mistyped
        --staging-dir pointing back at --input deletes the run being
        published before it can even be copied."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        with pytest.raises(SystemExit):
            publish_hf.stage_run(run_dir, corpus_name, run_dir, "sdf")
        # the run must survive the rejected attempt intact
        assert (run_dir / "final" / corpus_name).exists()

    def test_staging_dir_that_contains_run_dir_is_rejected(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        with pytest.raises(SystemExit):
            publish_hf.stage_run(run_dir, corpus_name, run_dir.parent, "sdf")
        assert (run_dir / "final" / corpus_name).exists()

    def test_staging_dir_equal_to_run_final_is_rejected(self, tmp_path):
        """Regression: the run_dir-only check missed this — a --staging-dir
        pointing directly at run_dir/final (an easy typo, since 'final' is a
        real, well-known subdirectory name on every run) slipped past it,
        and rmtree then deleted the corpus before it could be copied."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        with pytest.raises(SystemExit):
            publish_hf.stage_run(run_dir, corpus_name, run_dir / "final", "sdf")
        assert (run_dir / "final" / corpus_name).exists()

    def test_staging_dir_equal_to_run_audit_is_rejected(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        with pytest.raises(SystemExit):
            publish_hf.stage_run(run_dir, corpus_name, run_dir / "audit", "sdf")
        assert (run_dir / "audit" / "audit_report.json").exists()

    def test_staging_dir_nested_inside_run_dir_is_allowed(self, tmp_path):
        """The reverse nesting is safe as long as it doesn't overlap final/
        or audit/ specifically — deleting an unrelated subdir of run_dir
        doesn't touch run_dir's own data, and colocating the staged output
        with the run is a plausible deliberate choice."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = run_dir / "hf_staging"
        staged, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        assert staged["corpus_file"] == corpus_name
        assert (run_dir / "final" / corpus_name).exists()


class TestBuildMetricsRows:
    def test_files_with_a_generator_produce_a_row_each(self, tmp_path):
        """Only audit_report.json/compliance_report.json/diversity_report.json
        get a bespoke row — the only three with an actual committed generator
        (evals/audit_sdf.py, evals/compliance_sdf.py, evals/diversity.py).
        card_fidelity_report.json/realism_ablation.json/vendi_curve.json are
        one-off artifacts of a specific historical run with no generator
        anywhere in this repo — see build_metrics_rows' docstring."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = tmp_path / "staged"
        _, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        rows = publish_hf.build_metrics_rows(dataset_dir)
        assert len(rows) == 3
        joined = " ".join(f"{l}:{v}" for l, v, _source in rows)
        assert "98 of 100 judged clean (98.0%)" in joined
        assert "Vendi 34.5 effective records of 477 (ratio 0.072)" in joined
        assert any(label == "Documents (offline audit)" and value == "477"
                   for label, value, _source in rows)
        assert {source for _, _, source in rows} == {
            "audit_report.json", "compliance_report.json", "diversity_report.json",
        }

    def test_missing_files_omit_their_rows_without_error(self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, audit_files=["compliance_report.json"], include_html=False,
        )
        staging_dir = tmp_path / "staged"
        _, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        rows = publish_hf.build_metrics_rows(dataset_dir)
        assert len(rows) == 1
        assert rows[0][0] == "Constitutional compliance"

    def test_compliance_missing_clean_documents_omits_row(self, tmp_path):
        """Regression: only judged was guarded, not clean_documents — a
        compliance report with judged but no clean_documents would render a
        literal 'None of 100 judged clean' on the public card instead of
        omitting the row, contradicting the module's own stated contract."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, audit_files=[], include_html=False,
            extra_audit_files={"compliance_report.json": {"judged": 100}},
        )
        staging_dir = tmp_path / "staged"
        _, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        assert publish_hf.build_metrics_rows(dataset_dir) == []

    def test_no_audit_dir_gives_no_rows(self, tmp_path):
        dataset_dir = tmp_path / "staged" / "sdf"
        dataset_dir.mkdir(parents=True)
        assert publish_hf.build_metrics_rows(dataset_dir) == []

    def test_partial_diversity_report_omits_row_instead_of_crashing(self, tmp_path):
        """Regression: vendi_ratio/n_records were interpolated with :.3f/{} in
        the same f-string but only vendi_score was guarded — a diversity
        report with score but no ratio (not reachable via the current
        evals/diversity.py generator, but the module's own contract is
        'missing a field just omits the row') would raise instead."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, audit_files=[], include_html=False,
            extra_audit_files={"diversity_report.json": {
                "n_records": 477, "vendi": {"score": 34.45},  # ratio missing
            }},
        )
        staging_dir = tmp_path / "staged"
        _, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        assert publish_hf.build_metrics_rows(dataset_dir) == []

    def test_files_without_a_generator_never_produce_a_row(self, tmp_path):
        """card_fidelity_report.json/realism_ablation.json/vendi_curve.json are
        never parsed at all, regardless of their content — so a malformed one
        can't crash the publish (nothing to test beyond: no row, ever)."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, audit_files=[], include_html=False,
            extra_audit_files={
                "card_fidelity_report.json": {"unexpected": "shape"},
                "realism_ablation.json": {"n": 78},
                "vendi_curve.json": {"proj": "not even a dict"},
            },
        )
        staging_dir = tmp_path / "staged"
        _, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        assert publish_hf.build_metrics_rows(dataset_dir) == []


class TestDetectedLanguages:
    def test_sdf_run_derives_codes_from_audit_report_composition(self, tmp_path):
        """Regression: the card used to hardcode language: [en], but the
        culture matrix deliberately samples mostly non-English documents —
        audit_report.json's own composition.language breakdown (already read
        by build_metrics_rows) is the measured source of truth for this."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = tmp_path / "staged"
        _, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        # English, Spanish, Mandarin Chinese, Urdu -> en, es, zh, ur, sorted
        assert publish_hf.detected_languages(dataset_dir, "sdf") == ["en", "es", "ur", "zh"]

    def test_dad_run_always_falls_back_to_en(self, tmp_path):
        """DAD's audit_report.json has no composition.language breakdown —
        dilemmas are English-only by the dad.language_distribution default —
        so DAD runs shouldn't attempt the SDF-specific lookup at all."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad")
        staging_dir = tmp_path / "staged"
        _, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        assert publish_hf.detected_languages(dataset_dir, "dad") == ["en"]

    def test_missing_audit_report_falls_back_to_en(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, audit_files=[], include_html=False)
        staging_dir = tmp_path / "staged"
        _, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        assert publish_hf.detected_languages(dataset_dir, "sdf") == ["en"]

    def test_unmapped_language_name_is_skipped_not_crashed(self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, audit_files=[], include_html=False,
            extra_audit_files={"audit_report.json": {
                "n_docs": 10,
                "composition": {"language": {"English": 8, "Klingon": 2}},
            }},
        )
        staging_dir = tmp_path / "staged"
        _, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        assert publish_hf.detected_languages(dataset_dir, "sdf") == ["en"]


class TestBuildCard:
    def test_uses_report_content_title_and_subtitle_as_section_heading(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        card = _one_card(dataset_dir, staged, content=REPORT_CONTENT)
        assert card.startswith("---\n")
        # report_content's title is now the DATASET SECTION heading; the repo's
        # own pretty_name is the repo-level name, not one dataset's title
        assert "pretty_name: test-datasets" in card
        assert "# test-datasets" in card
        assert "## SDF corpus audit — 477 documents (`sdf` config)" in card
        assert "A test subtitle." in card
        assert "license: cc-by-4.0" in card
        # config paths are subdir-qualified now
        assert f"path: sdf/{corpus_name}" in card
        # multi-language corpus (see AUDIT_REPORT fixture) — not hardcoded "en"
        frontmatter = yaml.safe_load(card.split("---\n")[1])
        assert frontmatter["language"] == ["en", "es", "ur", "zh"]

    def test_pretty_name_with_yaml_breaking_characters_stays_valid(self, tmp_path):
        """Regression: pretty_name and section headings can come from
        report_content.json (editorial content this script doesn't control) —
        a raw quote or embedded newline used to corrupt the hand-built
        'pretty_name: "{title}"' line into invalid YAML. Must round-trip
        through a real YAML parser instead."""
        run_dir, corpus_name = make_run_dir(tmp_path, audit_files=[], include_html=False)
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        for tricky in ['A "quoted" name', "A name\nwith a newline", "Name: with a colon"]:
            card = _one_card(dataset_dir, staged, pretty_name=tricky)
            parsed = yaml.safe_load(card.split("---\n")[1])
            assert parsed["pretty_name"] == tricky

    def test_falls_back_to_generic_section_heading_without_content(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, audit_files=[], include_html=False,
                                            manifest=None)
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        card = _one_card(dataset_dir, staged, content=None)
        assert "## SDF corpus (`sdf` config)" in card

    def test_includes_provenance_from_manifest(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        card = _one_card(dataset_dir, staged)
        assert "`2026-07-25_15-57_fullscale-500-opus5`" in card
        assert "`4abd78b`" in card
        assert "`claude_code`" in card

    def test_card_surfaces_per_stage_model_overrides(self, tmp_path):
        """The manifest's top-level `model` alone misdescribes both published
        corpora: it reads claude-sonnet-5 while the stages that matter ran on
        Opus. A card showing only that would tell readers it's a Sonnet
        dataset."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        card = _one_card(dataset_dir, staged)
        assert "- **default model**: `claude-sonnet-5`" in card
        assert "- **per-stage models**: `claude-opus-5`" in card


class TestModelsUsed:
    def test_collects_distinct_overrides_excluding_the_default(self):
        manifest = {"model": "claude-sonnet-5", "config": {"sdf": {
            "rewrite_model": "claude-opus-5", "draft_model": "claude-sonnet-5",
            "score_model": "claude-opus-5", "n_prompts": 500,
        }}}
        assert publish_hf.models_used(manifest, "sdf") == (
            "claude-sonnet-5", ["claude-opus-5"])

    def test_no_overrides_gives_empty_list(self):
        manifest = {"model": "claude-sonnet-5", "config": {"sdf": {"n_prompts": 500}}}
        assert publish_hf.models_used(manifest, "sdf") == ("claude-sonnet-5", [])

    def test_dad_baseline_model_is_excluded(self):
        """The baseline arm is a plain-model control that is never trained on
        and never reaches the published corpus, so it must not be listed as a
        model that generated the dataset."""
        manifest = {"model": "claude-sonnet-5", "config": {"dad": {
            "constitution_rewrite_model": "claude-opus-4-8",
            "baseline": {"enabled": True, "model": "claude-haiku-4-5"},
        }}}
        default, overrides = publish_hf.models_used(manifest, "dad")
        assert default == "claude-sonnet-5"
        assert overrides == ["claude-opus-4-8"]
        assert "claude-haiku-4-5" not in overrides

    def test_missing_manifest_fields_do_not_crash(self):
        assert publish_hf.models_used({}, "sdf") == (None, [])

    def test_points_to_html_report_and_lists_extra_files(self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, audit_files=["compliance_report.json"],
            extra_audit_files={"custom_eval.json": {"foo": "bar"}},
        )
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        card = _one_card(dataset_dir, staged)
        # pointers are subdir-qualified so they resolve on the Hub
        assert "`sdf/audit/corpus_report.html`" in card
        assert "`custom_eval.json`" in card
        # compliance_report.json got its own metrics-table row — must not also
        # be duplicated into the catch-all "additional files" line
        extra_line = next(l for l in card.splitlines() if l.startswith("Additional"))
        assert "compliance_report.json" not in extra_line

    def test_no_generator_files_are_listed_as_extra_not_dropped(self, tmp_path):
        """card_fidelity_report.json/realism_ablation.json/vendi_curve.json get
        no metrics row (no committed generator reproduces them), but must
        still be visible in the card — otherwise they'd be uploaded yet
        invisible to anyone reading only the README."""
        run_dir, corpus_name = make_run_dir(tmp_path)  # default: all seven fixture files + html
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        card = _one_card(dataset_dir, staged)
        extra_line = next(l for l in card.splitlines() if l.startswith("Additional"))
        assert "`card_fidelity_report.json`" in extra_line
        assert "`realism_ablation.json`" in extra_line
        assert "`vendi_curve.json`" in extra_line
        # compliance_report.json DID get a row — must not be duplicated here
        assert "compliance_report.json" not in extra_line

    def test_known_file_with_unexpected_schema_still_listed_as_extra(self, tmp_path):
        """A known filename whose fields don't match what build_metrics_rows
        expects must not go silently invisible — it still surfaces in the
        catch-all line rather than disappearing from both sections."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, audit_files=[], include_html=False,
            extra_audit_files={"compliance_report.json": {"unexpected": "shape"}},
        )
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        assert publish_hf.build_metrics_rows(dataset_dir) == []
        card = _one_card(dataset_dir, staged)
        assert "`compliance_report.json`" in card

    def test_dad_run_counts_examples_not_documents(self, tmp_path):
        """DAD ships chat examples, not documents, and its audit generator
        reports n_prompts where SDF's reports n_docs."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", docs=40, audit_files=[], include_html=False,
            extra_audit_files={"audit_report.json": {"n_prompts": 40}},
        )
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        rows = publish_hf.build_metrics_rows(dataset_dir)
        assert [(l, v) for l, v, _ in rows] == [("Examples (offline audit)", "40")]
        card = _one_card(dataset_dir, staged)
        assert "40 chat examples." in card
        assert "## DAD corpus (`dad` config)" in card


class TestMultiDatasetCard:
    """The card covers the whole repo, so both datasets must survive one
    publish — this is what keeps a DAD publish from blanking SDF's section."""

    def _two(self, tmp_path):
        sdf_run, sdf_corpus = make_run_dir(tmp_path / "s", pipeline="sdf")
        dad_run, dad_corpus = make_run_dir(
            tmp_path / "d", pipeline="dad", docs=40, audit_files=[], include_html=False,
            extra_audit_files={"audit_report.json": {"n_prompts": 40}},
        )
        sdf_staged, sdf_dir = _stage(sdf_run, sdf_corpus, tmp_path / "stage_s")
        dad_staged, dad_dir = _stage(dad_run, dad_corpus, tmp_path / "stage_d")
        return [
            {"pipeline": "sdf", "dir": sdf_dir, "staged": sdf_staged, "content": None},
            {"pipeline": "dad", "dir": dad_dir, "staged": dad_staged, "content": None},
        ]

    def test_declares_both_configs_with_sdf_default(self, tmp_path):
        card = publish_hf.build_card(self._two(tmp_path), "cc-by-4.0", "repo-name")
        fm = yaml.safe_load(card.split("---\n")[1])
        assert [c["config_name"] for c in fm["configs"]] == ["sdf", "dad"]
        assert fm["configs"][0]["data_files"][0]["path"] == "sdf/sdf_corpus.jsonl"
        assert fm["configs"][1]["data_files"][0]["path"] == "dad/dad_corpus.jsonl"
        # only the first entry is the viewer's default
        assert fm["configs"][0].get("default") is True
        assert "default" not in fm["configs"][1]
        assert "sdf" in fm["tags"] and "dad" in fm["tags"]

    def test_language_is_the_union_across_datasets(self, tmp_path):
        """language: is repo-wide. SDF spans 16 languages and DAD is English
        only — declaring either alone would misdescribe the repo."""
        card = publish_hf.build_card(self._two(tmp_path), "cc-by-4.0", "repo-name")
        fm = yaml.safe_load(card.split("---\n")[1])
        # sdf fixture contributes en/es/ur/zh; dad contributes en
        assert fm["language"] == ["en", "es", "ur", "zh"]

    def test_both_sections_present_with_own_provenance(self, tmp_path):
        card = publish_hf.build_card(self._two(tmp_path), "cc-by-4.0", "repo-name")
        assert "## SDF corpus (`sdf` config)" in card
        assert "## DAD corpus (`dad` config)" in card
        assert "3 documents." in card       # sdf fixture default docs=3
        assert "40 chat examples." in card
        assert card.index("`sdf` config") < card.index("`dad` config")


class TestHubApiWrappers:
    def test_create_tag_passes_exist_ok(self, monkeypatch):
        """Regression: a retried publish with the same --tag (e.g. after
        fixing a typo'd --input, the exact retry the staging-dir wipe logic
        is designed to support) must not die on a "tag already exists" error
        after the corpus has already been re-uploaded."""
        calls = []

        class FakeHfApi:
            def create_tag(self, **kwargs):
                calls.append(kwargs)

        monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
        publish_hf._create_tag("sentientfutures/sdf-corpus", "v1")
        assert calls == [{
            "repo_id": "sentientfutures/sdf-corpus", "tag": "v1",
            "repo_type": "dataset", "exist_ok": True,
        }]

    def test_upload_folder_forwards_caller_scoped_delete_patterns(self, monkeypatch):
        """Republishing a run must clear that run's own audit/ on the Hub, so
        a file only an EARLIER run produced can't linger. The pattern comes
        from the caller because it must be scoped per pipeline — see
        test_delete_patterns_are_scoped_to_the_published_pipeline."""
        calls = []

        class FakeHfApi:
            def upload_folder(self, **kwargs):
                calls.append(kwargs)
                return "fake-commit"

        monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
        result = publish_hf._upload_folder(
            "/tmp/staged", "sentientfutures/x", "msg", ["sdf/audit/*"])
        assert result == "fake-commit"
        assert calls[0]["delete_patterns"] == ["sdf/audit/*"]
        assert calls[0]["repo_id"] == "sentientfutures/x"
        assert calls[0]["folder_path"] == "/tmp/staged"


def _run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["publish_hf.py", *args])
    publish_hf.main()


class TestMainEndToEnd:
    def test_dry_run_makes_no_hub_calls(self, tmp_path, monkeypatch, stub_hf, capsys):
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        _run_main(monkeypatch, "--input", str(run_dir),
                  "--repo-id", "sentientfutures/sdf-corpus", "--dry-run")
        out = capsys.readouterr().out
        assert "no Hub API calls made" in out
        assert "README.md" in out

    def test_dry_run_without_staging_dir_leaves_files_on_disk(
        self, tmp_path, monkeypatch, stub_hf, capsys
    ):
        """Regression: --dry-run's default staging dir used to live inside a
        tempfile.TemporaryDirectory() that self-deleted the instant main()
        returned, so the printed "Staged at <path>" was already gone by the
        time a human went to look — defeating the entire point of --dry-run.
        """
        run_dir, corpus_name = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        _run_main(monkeypatch, "--input", str(run_dir),
                  "--repo-id", "sentientfutures/sdf-corpus", "--dry-run")
        out = capsys.readouterr().out

        match = re.search(r"Staged at (\S+) \(left on disk", out)
        assert match, f"expected a 'Staged at <path>' message, got:\n{out}"
        staged_path = Path(match.group(1))
        try:
            assert staged_path.is_dir()
            assert (staged_path / "sdf" / corpus_name).exists()
            assert (staged_path / "README.md").exists()
        finally:
            shutil.rmtree(staged_path.parent, ignore_errors=True)

    def test_dry_run_warns_the_sibling_is_not_fetched(self, tmp_path, monkeypatch,
                                                     stub_hf, capsys):
        """--dry-run makes no network calls, so it cannot know about a sibling
        already on the Hub. It must say so rather than let the preview imply
        the sibling would be dropped."""
        run_dir, _ = make_run_dir(tmp_path, pipeline="dad", audit_files=[], include_html=False)
        stub_hf(raise_on_call=True)
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo", "--dry-run")
        out = capsys.readouterr().out
        assert "'sdf' dataset already on the Hub is not fetched" in out

    def test_publish_calls_hf_api_with_expected_args(self, tmp_path, monkeypatch, stub_hf):
        run_dir, corpus_name = make_run_dir(tmp_path)
        calls = stub_hf()
        staging_dir = tmp_path / "staged"
        _run_main(monkeypatch, "--input", str(run_dir),
                  "--repo-id", "sentientfutures/awmtd",
                  "--tag", "sdf-v1-fullscale-500-opus5",
                  "--staging-dir", str(staging_dir))

        by_fn = {c["fn"]: c for c in calls}
        assert by_fn["create_repo"]["repo_id"] == "sentientfutures/awmtd"
        assert by_fn["upload_folder"]["repo_id"] == "sentientfutures/awmtd"
        assert by_fn["upload_folder"]["folder_path"] == str(staging_dir)
        assert by_fn["create_tag"]["tag"] == "sdf-v1-fullscale-500-opus5"

        uploaded = {str(p.relative_to(staging_dir))
                    for p in staging_dir.rglob("*") if p.is_file()}
        assert f"sdf/{corpus_name}" in uploaded
        assert "sdf/run_manifest.json" in uploaded
        assert "README.md" in uploaded
        assert not any("report_content.json" in u for u in uploaded)

    def test_delete_patterns_are_scoped_to_the_published_pipeline(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """THE most dangerous line in the multi-dataset change: a bare
        "audit/*" would delete the SIBLING pipeline's audit files on every
        publish, silently gutting the other dataset in the same repo."""
        run_dir, _ = make_run_dir(tmp_path, pipeline="dad", audit_files=[], include_html=False)
        calls = stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo")
        upload = next(c for c in calls if c["fn"] == "upload_folder")
        assert upload["delete_patterns"] == ["dad/audit/*"]

    def test_publish_without_tag_skips_create_tag(self, tmp_path, monkeypatch, stub_hf):
        run_dir, _ = make_run_dir(tmp_path, audit_files=[], include_html=False)
        calls = stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo")
        assert "create_tag" not in [c["fn"] for c in calls]

    def test_dad_run_end_to_end(self, tmp_path, monkeypatch, stub_hf):
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", audit_files=[],
                                            include_html=False)
        staging_dir = tmp_path / "staged"
        stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir))
        assert (staging_dir / "dad" / "dad_corpus.jsonl").exists()
        assert "## DAD corpus (`dad` config)" in (staging_dir / "README.md").read_text()

    def test_pretty_name_defaults_to_repo_id_last_segment(self, tmp_path, monkeypatch, stub_hf):
        run_dir, _ = make_run_dir(tmp_path, audit_files=[], include_html=False)
        staging_dir = tmp_path / "staged"
        stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir),
                  "--repo-id", "sentientfutures/animal-welfare-mid-training-datasets",
                  "--staging-dir", str(staging_dir))
        card = (staging_dir / "README.md").read_text()
        assert "pretty_name: animal-welfare-mid-training-datasets" in card

    def test_explicit_pretty_name_wins(self, tmp_path, monkeypatch, stub_hf):
        run_dir, _ = make_run_dir(tmp_path, audit_files=[], include_html=False)
        staging_dir = tmp_path / "staged"
        stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--pretty-name", "Animal-welfare midtraining datasets",
                  "--staging-dir", str(staging_dir))
        card = (staging_dir / "README.md").read_text()
        assert "pretty_name: Animal-welfare midtraining datasets" in card


# What a already-published SDF dataset looks like from list_repo_files' side.
SIBLING_SDF_FILES = {
    "sdf/sdf_corpus.jsonl": None,       # listed but never downloaded
    "sdf/run_manifest.json": MANIFEST,
    "sdf/audit/audit_report.json": AUDIT_REPORT,
    "sdf/audit/diversity_report.json": DIVERSITY,
    "sdf/audit/corpus_report.html": None,   # listed but never downloaded
}


class TestSiblingPreservation:
    """Publishing one dataset regenerates the whole card, so the other
    dataset's section and config entry must be rebuilt from the Hub."""

    def _publish_dad(self, tmp_path, monkeypatch, stub_hf, repo_files):
        run_dir, _ = make_run_dir(
            tmp_path, pipeline="dad", docs=40, audit_files=[], include_html=False,
            extra_audit_files={"audit_report.json": {"n_prompts": 40}},
        )
        staging_dir = tmp_path / "staged"
        calls = stub_hf(repo_files=repo_files)
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir))
        return calls, staging_dir, (staging_dir / "README.md").read_text()

    def test_dad_publish_keeps_sdf_config_and_section(self, tmp_path, monkeypatch, stub_hf):
        _, _, card = self._publish_dad(tmp_path, monkeypatch, stub_hf, SIBLING_SDF_FILES)
        fm = yaml.safe_load(card.split("---\n")[1])
        assert [c["config_name"] for c in fm["configs"]] == ["sdf", "dad"]
        assert fm["configs"][0]["data_files"][0]["path"] == "sdf/sdf_corpus.jsonl"
        # sdf keeps default even though dad is the one being published
        assert fm["configs"][0].get("default") is True
        assert "## SDF corpus (`sdf` config)" in card
        assert "## DAD corpus (`dad` config)" in card
        # sdf's own measured numbers, read from its Hub-side audit files
        assert "477 documents." in card
        assert "40 chat examples." in card
        # union of both, not just dad's ["en"]
        assert fm["language"] == ["en", "es", "ur", "zh"]

    def test_sibling_corpus_and_html_are_never_downloaded(self, tmp_path, monkeypatch, stub_hf):
        """Only small metadata is fetched — the sibling's multi-MB corpus and
        its HTML report stay on the Hub untouched."""
        calls, _, _ = self._publish_dad(tmp_path, monkeypatch, stub_hf, SIBLING_SDF_FILES)
        downloaded = [c["filename"] for c in calls if c["fn"] == "download_file"]
        assert set(downloaded) == {
            "sdf/run_manifest.json",
            "sdf/audit/audit_report.json",
            "sdf/audit/diversity_report.json",
        }
        assert not any(f.endswith((".jsonl", ".html")) for f in downloaded)

    def test_sibling_metadata_never_enters_the_upload(self, tmp_path, monkeypatch, stub_hf):
        """The sibling is fetched OUTSIDE the staged tree, so neither its files
        nor hf_hub_download's .cache bookkeeping dir get uploaded as content."""
        _, staging_dir, _ = self._publish_dad(
            tmp_path, monkeypatch, stub_hf, SIBLING_SDF_FILES)
        staged_paths = {str(p.relative_to(staging_dir))
                        for p in staging_dir.rglob("*") if p.is_file()}
        assert not any(p.startswith("sdf/") for p in staged_paths)
        assert not any(".cache" in p for p in staged_paths)
        assert staged_paths == {"README.md", "dad/dad_corpus.jsonl",
                                "dad/run_manifest.json", "dad/audit/audit_report.json"}

    def test_no_sibling_yet_gives_a_single_config_card(self, tmp_path, monkeypatch, stub_hf):
        """First publish into a fresh repo: nothing to preserve, and the card
        must still be valid rather than declaring a config for missing data."""
        _, _, card = self._publish_dad(tmp_path, monkeypatch, stub_hf, {})
        fm = yaml.safe_load(card.split("---\n")[1])
        assert [c["config_name"] for c in fm["configs"]] == ["dad"]
        assert fm["configs"][0].get("default") is True
        assert "## SDF corpus" not in card

    def test_sibling_dir_without_a_corpus_is_skipped(self, tmp_path, monkeypatch, stub_hf):
        """A partial sibling dir with metadata but no corpus can't be declared
        as a config — better to omit it than point a config at nothing."""
        _, _, card = self._publish_dad(
            tmp_path, monkeypatch, stub_hf,
            {"sdf/run_manifest.json": MANIFEST, "sdf/audit/audit_report.json": AUDIT_REPORT},
        )
        fm = yaml.safe_load(card.split("---\n")[1])
        assert [c["config_name"] for c in fm["configs"]] == ["dad"]
