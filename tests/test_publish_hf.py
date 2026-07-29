"""Tests for evals/publish_hf.py — staging, card generation, and the
Hub-upload chokepoints (stubbed via the stub_hf fixture; never touches
huggingface_hub or the network)."""

import json
import sys

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
AUDIT_REPORT = {"n_docs": 477}
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
        resolved, corpus_name = publish_hf.resolve_run_dir(str(run_dir))
        assert resolved == run_dir
        assert corpus_name == "sdf_corpus.jsonl"

    def test_dad_run(self, tmp_path):
        run_dir, _ = make_run_dir(tmp_path, pipeline="dad")
        _, corpus_name = publish_hf.resolve_run_dir(str(run_dir))
        assert corpus_name == "dad_corpus.jsonl"

    def test_missing_corpus_raises(self, tmp_path):
        empty = tmp_path / "runs" / "empty-run"
        empty.mkdir(parents=True)
        with pytest.raises(SystemExit):
            publish_hf.resolve_run_dir(str(empty))

    def test_not_a_directory_raises(self, tmp_path):
        with pytest.raises(SystemExit):
            publish_hf.resolve_run_dir(str(tmp_path / "nope"))


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


class TestBuildMetricsRows:
    def test_all_known_files_produce_a_row_each(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = tmp_path / "staged"
        publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        rows = publish_hf.build_metrics_rows(staging_dir)
        assert len(rows) == 6
        joined = " ".join(f"{l}:{v}" for l, v, _source in rows)
        assert "98 of 100 judged clean (98.0%)" in joined
        assert "resolution 65.7%" in joined
        assert "Vendi 34.5 effective docs of 477 (ratio 0.072)" in joined
        assert "8.49 (in-spec) vs 5.78 (spec hidden), drop 2.71" in joined
        assert "n=1000" in joined and "n=5000" in joined
        assert any(label == "Documents (offline audit)" and value == "477"
                   for label, value, _source in rows)
        assert {source for _, _, source in rows} == {
            "audit_report.json", "compliance_report.json", "card_fidelity_report.json",
            "diversity_report.json", "realism_ablation.json", "vendi_curve.json",
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


class TestBuildCard:
    def test_uses_report_content_title_and_subtitle(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = tmp_path / "staged"
        staged = publish_hf.stage_run(run_dir, corpus_name, staging_dir)
        card = publish_hf.build_card(staging_dir, staged, "cc-by-4.0", "sdf", content=REPORT_CONTENT)
        assert card.startswith("---\n")
        assert 'pretty_name: "SDF corpus audit — 477 documents"' in card
        assert "# SDF corpus audit — 477 documents" in card
        assert "A test subtitle." in card
        assert "license: cc-by-4.0" in card
        assert f"path: {corpus_name}" in card

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
