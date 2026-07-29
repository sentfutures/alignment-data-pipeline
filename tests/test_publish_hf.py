"""Tests for evals/publish_hf.py — staging, card generation, and the
Hub-upload chokepoints (stubbed via the stub_hf fixture; never touches
huggingface_hub or the network)."""

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

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
    "model": "claude-sonnet-5",
    "config": {"backend": "claude_code"},
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
        staged = publish_hf.stage_run(run_dir, corpus_name, staging_dir)

        assert staged["corpus_file"] == corpus_name
        assert staged["n_docs"] == 3
        assert staged["manifest_file"] == "run_manifest.json"
        assert (staging_dir / corpus_name).exists()
        assert (staging_dir / "run_manifest.json").exists()
        # report_content.json is editorial and must never be staged/uploaded
        assert "report_content.json" not in staged["audit_files"]
        assert not (staging_dir / "audit" / "report_content.json").exists()
        assert set(staged["audit_files"]) == {
            "audit_report.json", "compliance_report.json", "card_fidelity_report.json",
            "diversity_report.json", "realism_ablation.json", "vendi_curve.json",
            "corpus_report.html",
        }

    def test_no_audit_dir_is_fine(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, audit_files=[], include_html=False)
        staged = publish_hf.stage_run(run_dir, corpus_name, tmp_path / "staged")
        assert staged["audit_files"] == []

    def test_no_manifest_is_fine(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, manifest=None, audit_files=[],
                                            include_html=False)
        staged = publish_hf.stage_run(run_dir, corpus_name, tmp_path / "staged")
        assert staged["manifest_file"] is None

    def test_unknown_audit_file_is_staged_anyway(self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, audit_files=[], include_html=False,
            extra_audit_files={"custom_eval.json": {"foo": "bar"}},
        )
        staged = publish_hf.stage_run(run_dir, corpus_name, tmp_path / "staged")
        assert staged["audit_files"] == ["custom_eval.json"]

    def test_reused_staging_dir_is_cleared_of_stale_files(self, tmp_path):
        """Regression: a --staging-dir reused across two invocations (e.g. after
        fixing a typo'd --input) must reflect only the LATEST run — leftover
        files from an earlier call must not ride along into the upload."""
        staging_dir = tmp_path / "staged"

        run_a, corpus_a = make_run_dir(tmp_path / "a", audit_files=["compliance_report.json"])
        publish_hf.stage_run(run_a, corpus_a, staging_dir)
        assert (staging_dir / "audit" / "compliance_report.json").exists()

        run_b, corpus_b = make_run_dir(
            tmp_path / "b", pipeline="dad", audit_files=["audit_report.json"], include_html=False,
        )
        staged_b = publish_hf.stage_run(run_b, corpus_b, staging_dir)

        assert staged_b["audit_files"] == ["audit_report.json"]
        assert not (staging_dir / "audit" / "compliance_report.json").exists()
        assert not (staging_dir / "audit" / "corpus_report.html").exists()
        # run A's corpus file (a different pipeline's filename) must not survive either
        assert not (staging_dir / corpus_a).exists()
        assert (staging_dir / corpus_b).exists()

    def test_staging_dir_equal_to_run_dir_is_rejected(self, tmp_path):
        """Regression: rmtree(staging_dir) must never fire before verifying
        staging_dir doesn't equal or contain run_dir — otherwise a mistyped
        --staging-dir pointing back at --input deletes the run being
        published before it can even be copied."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        with pytest.raises(SystemExit):
            publish_hf.stage_run(run_dir, corpus_name, run_dir)
        # the run must survive the rejected attempt intact
        assert (run_dir / "final" / corpus_name).exists()

    def test_staging_dir_that_contains_run_dir_is_rejected(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        with pytest.raises(SystemExit):
            publish_hf.stage_run(run_dir, corpus_name, run_dir.parent)
        assert (run_dir / "final" / corpus_name).exists()

    def test_staging_dir_nested_inside_run_dir_is_allowed(self, tmp_path):
        """The reverse nesting is safe — deleting a subdir of run_dir doesn't
        touch run_dir's own final/audit/manifest files — and is a plausible
        deliberate choice (colocating the staged output with the run)."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = run_dir / "hf_staging"
        staged = publish_hf.stage_run(run_dir, corpus_name, staging_dir)
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
        publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        rows = publish_hf.build_metrics_rows(staging_dir)
        assert len(rows) == 3
        joined = " ".join(f"{l}:{v}" for l, v, _source in rows)
        assert "98 of 100 judged clean (98.0%)" in joined
        assert "Vendi 34.5 effective docs of 477 (ratio 0.072)" in joined
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
        publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        rows = publish_hf.build_metrics_rows(staging_dir)
        assert len(rows) == 1
        assert rows[0][0] == "Constitutional compliance"

    def test_no_audit_dir_gives_no_rows(self, tmp_path):
        staging_dir = tmp_path / "staged"
        staging_dir.mkdir()
        assert publish_hf.build_metrics_rows(staging_dir) == []

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
        publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        assert publish_hf.build_metrics_rows(staging_dir) == []

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
        publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        assert publish_hf.build_metrics_rows(staging_dir) == []


class TestDetectedLanguages:
    def test_sdf_run_derives_codes_from_audit_report_composition(self, tmp_path):
        """Regression: the card used to hardcode language: [en], but the
        culture matrix deliberately samples mostly non-English documents —
        audit_report.json's own composition.language breakdown (already read
        by build_metrics_rows) is the measured source of truth for this."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = tmp_path / "staged"
        publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        # English, Spanish, Mandarin Chinese, Urdu -> en, es, zh, ur, sorted
        assert publish_hf.detected_languages(staging_dir, "sdf") == ["en", "es", "ur", "zh"]

    def test_dad_run_always_falls_back_to_en(self, tmp_path):
        """DAD's audit_report.json has no composition.language breakdown —
        dilemmas are English-only by the dad.language_distribution default —
        so DAD runs shouldn't attempt the SDF-specific lookup at all."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad")
        staging_dir = tmp_path / "staged"
        publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        assert publish_hf.detected_languages(staging_dir, "dad") == ["en"]

    def test_missing_audit_report_falls_back_to_en(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, audit_files=[], include_html=False)
        staging_dir = tmp_path / "staged"
        publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        assert publish_hf.detected_languages(staging_dir, "sdf") == ["en"]

    def test_unmapped_language_name_is_skipped_not_crashed(self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, audit_files=[], include_html=False,
            extra_audit_files={"audit_report.json": {
                "n_docs": 10,
                "composition": {"language": {"English": 8, "Klingon": 2}},
            }},
        )
        staging_dir = tmp_path / "staged"
        publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        assert publish_hf.detected_languages(staging_dir, "sdf") == ["en"]


class TestBuildCard:
    def test_uses_report_content_title_and_subtitle(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = tmp_path / "staged"
        staged = publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        card = publish_hf.build_card(staging_dir, staged, "cc-by-4.0", "sdf", content=REPORT_CONTENT)
        assert card.startswith("---\n")
        assert "pretty_name: SDF corpus audit — 477 documents" in card
        assert "# SDF corpus audit — 477 documents" in card
        assert "A test subtitle." in card
        assert "license: cc-by-4.0" in card
        assert f"path: {corpus_name}" in card
        # multi-language corpus (see AUDIT_REPORT fixture) — not hardcoded "en"
        import yaml
        frontmatter = yaml.safe_load(card.split("---\n")[1])
        assert frontmatter["language"] == ["en", "es", "ur", "zh"]

    def test_title_with_yaml_breaking_characters_stays_valid_frontmatter(self, tmp_path):
        """Regression: title comes from report_content.json (editorial content
        this script doesn't control) — a raw quote or embedded newline used
        to corrupt the hand-built 'pretty_name: "{title}"' line into invalid
        YAML. Must round-trip through a real YAML parser instead."""
        run_dir, corpus_name = make_run_dir(tmp_path, audit_files=[], include_html=False)
        staging_dir = tmp_path / "staged"
        staged = publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        import yaml
        for tricky_title in ['A "quoted" title', "A title\nwith a newline", "Title: with a colon"]:
            card = publish_hf.build_card(
                staging_dir, staged, "cc-by-4.0", "sdf",
                content={"title": tricky_title},
            )
            frontmatter_text = card.split("---\n")[1]
            parsed = yaml.safe_load(frontmatter_text)
            assert parsed["pretty_name"] == tricky_title

    def test_falls_back_to_generic_title_without_content(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, audit_files=[], include_html=False,
                                            manifest=None)
        staging_dir = tmp_path / "staged"
        staged = publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        card = publish_hf.build_card(staging_dir, staged, "cc-by-4.0", "sdf", content=None)
        assert "# SDF corpus" in card
        assert "## Provenance" not in card

    def test_includes_provenance_from_manifest(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = tmp_path / "staged"
        staged = publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        card = publish_hf.build_card(staging_dir, staged, "cc-by-4.0", "sdf")
        assert "`2026-07-25_15-57_fullscale-500-opus5`" in card
        assert "`4abd78b`" in card
        assert "`claude_code`" in card

    def test_points_to_html_report_and_lists_extra_files(self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, audit_files=["compliance_report.json"],
            extra_audit_files={"custom_eval.json": {"foo": "bar"}},
        )
        staging_dir = tmp_path / "staged"
        staged = publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        card = publish_hf.build_card(staging_dir, staged, "cc-by-4.0", "sdf")
        assert "audit/corpus_report.html" in card
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
        staging_dir = tmp_path / "staged"
        staged = publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        card = publish_hf.build_card(staging_dir, staged, "cc-by-4.0", "sdf")
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
        staging_dir = tmp_path / "staged"
        staged = publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        assert publish_hf.build_metrics_rows(staging_dir) == []
        card = publish_hf.build_card(staging_dir, staged, "cc-by-4.0", "sdf")
        assert "`compliance_report.json`" in card


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
            assert (staged_path / corpus_name).exists()
            assert (staged_path / "README.md").exists()
        finally:
            shutil.rmtree(staged_path.parent, ignore_errors=True)

    def test_publish_calls_hf_api_with_expected_args(self, tmp_path, monkeypatch, stub_hf):
        run_dir, corpus_name = make_run_dir(tmp_path)
        calls = stub_hf()
        staging_dir = tmp_path / "staged"
        _run_main(monkeypatch, "--input", str(run_dir),
                  "--repo-id", "sentientfutures/sdf-corpus",
                  "--tag", "v1-fullscale-500-opus5",
                  "--staging-dir", str(staging_dir))

        by_fn = {c["fn"]: c for c in calls}
        assert by_fn["create_repo"]["repo_id"] == "sentientfutures/sdf-corpus"
        assert by_fn["upload_folder"]["repo_id"] == "sentientfutures/sdf-corpus"
        assert by_fn["upload_folder"]["folder_path"] == str(staging_dir)
        assert by_fn["create_tag"]["tag"] == "v1-fullscale-500-opus5"

        uploaded = {p.name for p in staging_dir.rglob("*") if p.is_file()}
        assert corpus_name in uploaded
        assert "run_manifest.json" in uploaded
        assert "README.md" in uploaded
        assert "report_content.json" not in uploaded

    def test_publish_without_tag_skips_create_tag(self, tmp_path, monkeypatch, stub_hf):
        run_dir, _ = make_run_dir(tmp_path, audit_files=[], include_html=False)
        calls = stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "sentientfutures/sdf-corpus")
        assert [c["fn"] for c in calls] == ["create_repo", "upload_folder"]

    def test_dad_run_end_to_end(self, tmp_path, monkeypatch, stub_hf):
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", audit_files=[],
                                            include_html=False)
        staging_dir = tmp_path / "staged"
        stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "sentientfutures/dad-corpus",
                  "--staging-dir", str(staging_dir))
        assert (staging_dir / "dad_corpus.jsonl").exists()
        assert "# DAD corpus" in (staging_dir / "README.md").read_text()
