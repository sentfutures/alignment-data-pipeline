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

# A fully-merged verdict — what merge_state returns on a clean main checkout.
MERGED_STATE = {
    "branch": "main", "head": "4abd78b", "head_merged": True, "ahead": 0,
    "run_commit": "4abd78b", "run_commit_merged": True,
    "fetched": True, "notes": [],
}


def unmerged_state(branch="declan/wip", commit="deadbee", ahead=3, **over):
    """merge_state's verdict for a run whose code never reached origin/main."""
    return {**MERGED_STATE, "branch": branch, "head": commit, "ahead": ahead,
            "head_merged": False, "run_commit": commit,
            "run_commit_merged": False, **over}


@pytest.fixture(autouse=True)
def _default_merged(monkeypatch):
    """Pin merge_state to "merged" for every test in this module.

    Without this the real helper runs, and the suite's result would depend on
    the branch the developer happens to be on — green on main, and blocked on a
    typed confirmation everywhere else. Tests that exercise the guard override
    this with their own monkeypatch.
    """
    monkeypatch.setattr(publish_hf, "merge_state",
                        lambda commit, fetch=True: dict(MERGED_STATE))


def make_run_dir(tmp_path, pipeline="sdf", docs=3, audit_files=None, manifest=MANIFEST,
                  include_html=True, extra_audit_files=None, run_name=None):
    """Build a fake run directory with the given audit files present.

    audit_files=None means "all six known + report_content.json + html";
    pass a subset of KNOWN_AUDIT_FILES' keys to omit others. run_name lets a
    combined-publish test build several distinct run dirs under one tmp_path.
    """
    run_dir = tmp_path / "runs" / (run_name or "2026-07-25_15-57_fullscale-500-opus5")
    final = run_dir / "final"
    final.mkdir(parents=True)
    corpus_name = "sdf_corpus.jsonl" if pipeline == "sdf" else "dad_corpus.jsonl"
    if pipeline == "sdf":
        lines = [json.dumps({"doc_id": f"d{i}", "content": f"document {i}"})
                 for i in range(docs)]
    else:
        lines = [json.dumps({
            "record_id": f"r{i}", "example_gid": f"E-{i:04d}",
            "response_gid": f"R-{i:04d}",
            "messages": [
                {"role": "user", "content": f"user prompt {i}"},
                {"role": "assistant", "content": f"assistant response {i}"},
            ],
        }) for i in range(docs)]
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
    run_dirs = run_dir if isinstance(run_dir, list) else [run_dir]
    staged = publish_hf.stage_run(run_dirs, corpus_name, staging_dir, tag)
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
            publish_hf.stage_run([run_dir], corpus_name, run_dir, "sdf")
        # the run must survive the rejected attempt intact
        assert (run_dir / "final" / corpus_name).exists()

    def test_staging_dir_that_contains_run_dir_is_rejected(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        with pytest.raises(SystemExit):
            publish_hf.stage_run([run_dir], corpus_name, run_dir.parent, "sdf")
        assert (run_dir / "final" / corpus_name).exists()

    def test_staging_dir_equal_to_run_final_is_rejected(self, tmp_path):
        """Regression: the run_dir-only check missed this — a --staging-dir
        pointing directly at run_dir/final (an easy typo, since 'final' is a
        real, well-known subdirectory name on every run) slipped past it,
        and rmtree then deleted the corpus before it could be copied."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        with pytest.raises(SystemExit):
            publish_hf.stage_run([run_dir], corpus_name, run_dir / "final", "sdf")
        assert (run_dir / "final" / corpus_name).exists()

    def test_staging_dir_equal_to_run_audit_is_rejected(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        with pytest.raises(SystemExit):
            publish_hf.stage_run([run_dir], corpus_name, run_dir / "audit", "sdf")
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


class TestFlattenDadCorpus:
    def test_staged_dad_records_are_flat_columns(self, tmp_path):
        """The published copy shows one column per field (example_gid,
        user_prompt, assistant_response) — no messages array, no role keys."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=2,
                                            audit_files=[], include_html=False)
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        assert staged["n_docs"] == 2
        lines = (dataset_dir / corpus_name).read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        assert len(records) == 2
        for i, rec in enumerate(records):
            assert set(rec) == {"example_gid", "user_prompt", "assistant_response"}
            assert rec["example_gid"] == f"E-{i:04d}"
            assert rec["user_prompt"] == f"user prompt {i}"
            assert rec["assistant_response"] == f"assistant response {i}"

    def test_published_rows_carry_no_run_column(self, tmp_path):
        """Row-to-run attribution is the repo's job (git grep on the globally
        unique example_gid), not a repeated run_id on every row."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=2,
                                            audit_files=[], include_html=False)
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        raw = (dataset_dir / corpus_name).read_text(encoding="utf-8")
        assert "source_run" not in raw
        assert MANIFEST["run_id"] not in raw

    def test_local_training_corpus_is_left_untouched(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=1,
                                            audit_files=[], include_html=False)
        src = run_dir / "final" / corpus_name
        before = src.read_text(encoding="utf-8")
        _stage(run_dir, corpus_name, tmp_path / "staged")
        assert src.read_text(encoding="utf-8") == before
        assert "messages" in before  # the SFT chat shape stays on disk

    def test_sdf_corpus_is_copied_verbatim(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="sdf", docs=2,
                                            audit_files=[], include_html=False)
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        src = (run_dir / "final" / corpus_name).read_text(encoding="utf-8")
        assert (dataset_dir / corpus_name).read_text(encoding="utf-8") == src

    def test_record_without_assistant_message_aborts(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=1,
                                            audit_files=[], include_html=False)
        bad = {"example_gid": "E-9999",
               "messages": [{"role": "user", "content": "only a user turn"}]}
        (run_dir / "final" / corpus_name).write_text(
            json.dumps(bad) + "\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="E-9999"):
            _stage(run_dir, corpus_name, tmp_path / "staged")

    def test_non_ascii_content_is_not_escaped(self, tmp_path):
        """The corpus is multilingual; published rows must keep native script
        readable, not \\uXXXX escapes."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=1,
                                            audit_files=[], include_html=False)
        rec = {"example_gid": "E-0001", "messages": [
            {"role": "user", "content": "鶏の福祉について"},
            {"role": "assistant", "content": "丁寧に考えます"},
        ]}
        (run_dir / "final" / corpus_name).write_text(
            json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        raw = (dataset_dir / corpus_name).read_text(encoding="utf-8")
        assert "鶏の福祉について" in raw


def _dad_manifest(run_id, backend="api", dirty=None, dirty_files=None):
    """A DAD run manifest. dirty=None omits the git_dirty fields entirely (a
    manifest predating them); pass dirty=True/False to set them, shaped as
    shared.utils records them."""
    m = {"run_id": run_id, "label": run_id.split("_", 2)[-1],
         "git_commit": "abc1234", "model": "claude-sonnet-5",
         "config": {"backend": backend,
                    "dad": {"constitution_rewrite_model": "claude-opus-5"}}}
    if dirty is not None:
        m["git_dirty"] = dirty
        m["git_dirty_files"] = list(dirty_files or [])
    return m


class TestCombinedPublish:
    def _two_runs(self, tmp_path, second_audit=False):
        run_a, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", docs=2, audit_files=[], include_html=False,
            run_name="2026-07-28_17-32_pareto200",
            manifest=_dad_manifest("2026-07-28_17-32_pareto200"))
        run_b, _ = make_run_dir(
            tmp_path, pipeline="dad", docs=3,
            audit_files=["diversity_report.json"] if second_audit else [],
            include_html=False,
            run_name="2026-07-29_23-58_archetype1000",
            manifest=_dad_manifest("2026-07-29_23-58_archetype1000"))
        return [run_a, run_b], corpus_name

    def test_corpora_concatenate_in_input_order(self, tmp_path):
        """Every run's rows land in the combined corpus, in --input order. The
        rows carry no run column — example_gid is what identifies them, and
        _two_runs gives both runs the same gids on purpose (each run dir is
        built independently, exactly as separate real runs would be) so the
        assertion can't accidentally lean on distinct ids."""
        run_dirs, corpus_name = self._two_runs(tmp_path)
        staged, dataset_dir = _stage(run_dirs, corpus_name, tmp_path / "staged")

        assert staged["n_docs"] == 5
        assert [r["n_docs"] for r in staged["runs"]] == [2, 3]
        records = [json.loads(line) for line in
                   (dataset_dir / corpus_name).read_text(encoding="utf-8").splitlines()]
        assert len(records) == 5
        assert "source_run" not in records[0]
        # run_a contributed 2 rows, then run_b's 3 — docs=2 and docs=3 make the
        # user_prompt sequence restart, which is what pins the concatenation order.
        assert [r["user_prompt"] for r in records] == [
            "user prompt 0", "user prompt 1",
            "user prompt 0", "user prompt 1", "user prompt 2"]

    def test_per_run_table_reports_each_run_s_code_state(self, tmp_path):
        """Mixed states must be named per run, not collapsed — a combined
        corpus is only as reproducible as its least-reproducible run."""
        run_a, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", docs=2, audit_files=[], include_html=False,
            run_name="2026-07-28_17-32_pareto200",
            manifest=_dad_manifest("2026-07-28_17-32_pareto200",
                                   dirty=True, dirty_files=["config.yaml"]))
        run_b, _ = make_run_dir(
            tmp_path, pipeline="dad", docs=3, audit_files=[], include_html=False,
            run_name="2026-07-29_23-58_archetype1000",
            manifest=_dad_manifest("2026-07-29_23-58_archetype1000", dirty=False))
        staged, dataset_dir = _stage([run_a, run_b], corpus_name, tmp_path / "staged")
        card = _one_card(dataset_dir, staged)

        assert "| code state |" in card
        assert "| `abc1234` | dirty (1 uncommitted file) |" in card
        assert "| `abc1234` | clean |" in card

    def test_manifests_and_audits_are_run_scoped(self, tmp_path):
        """Several runs in one dataset dir must not collide on filenames:
        manifests land under manifests/<run_id>.json and audit files under
        audit/<run_id>/."""
        run_dirs, corpus_name = self._two_runs(tmp_path, second_audit=True)
        staged, dataset_dir = _stage(run_dirs, corpus_name, tmp_path / "staged")

        assert not (dataset_dir / "run_manifest.json").exists()
        assert (dataset_dir / "manifests" / "2026-07-28_17-32_pareto200.json").exists()
        assert (dataset_dir / "manifests" / "2026-07-29_23-58_archetype1000.json").exists()
        assert staged["audit_files"] == [
            "2026-07-29_23-58_archetype1000/diversity_report.json"]
        assert (dataset_dir / "audit" / "2026-07-29_23-58_archetype1000"
                / "diversity_report.json").exists()

    def test_card_section_renders_per_run_table(self, tmp_path):
        run_dirs, corpus_name = self._two_runs(tmp_path, second_audit=True)
        staged, dataset_dir = _stage(run_dirs, corpus_name, tmp_path / "staged")
        card = _one_card(dataset_dir, staged)

        assert "5 chat examples." in card
        assert "this table is the provenance record" in card
        assert "`source_run`" not in card
        assert "| `2026-07-28_17-32_pareto200` | 2 |" in card
        assert "| `2026-07-29_23-58_archetype1000` | 3 |" in card
        # per-run metrics line only for the run that has an audit
        assert "**`2026-07-29_23-58_archetype1000`** — Semantic diversity" in card
        assert "dad/audit/2026-07-29_23-58_archetype1000/" in card
        # the frontmatter still declares exactly one dad config
        fm = yaml.safe_load(card.split("---")[1])
        assert [c["config_name"] for c in fm["configs"]] == ["dad"]

    def test_single_run_layout_is_unchanged(self, tmp_path):
        """One --input keeps the original shape: top-level run_manifest.json,
        flat audit/, no manifests/ dir."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=2,
                                            audit_files=[], include_html=False)
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        assert staged["manifest_file"] == "run_manifest.json"
        assert (dataset_dir / "run_manifest.json").exists()
        assert not (dataset_dir / "manifests").exists()


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

    def test_card_reports_dirty_tree_alongside_the_commit(self, tmp_path):
        """A bare SHA implies the run is reproducible from that commit. It
        isn't when the tree was dirty, so the card says so next to it."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", audit_files=[], include_html=False,
            manifest=_dad_manifest(
                "2026-07-29_23-58_archetype1000", dirty=True,
                dirty_files=["dad_pipeline/run.py", "prompts/dad/step1d_refine.txt"]))
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        card = _one_card(dataset_dir, staged)
        assert "- **git commit**: `abc1234`" in card
        assert "- **code state**: dirty (2 uncommitted files)" in card

    def test_card_reports_unknown_code_state_for_older_manifests(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)  # MANIFEST has no git_dirty
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        assert "- **code state**: unknown" in _one_card(dataset_dir, staged)


class TestCodeState:
    """The commit alone overstates reproducibility: every DAD run published so
    far ran with uncommitted changes, several touching pipeline code and prompt
    templates."""

    def test_dirty_reports_the_file_count(self):
        assert publish_hf.code_state(
            {"git_dirty": True,
             "git_dirty_files": ["dad_pipeline/run.py", "config.yaml"]}
        ) == "dirty (2 uncommitted files)"

    def test_one_dirty_file_is_singular(self):
        assert publish_hf.code_state(
            {"git_dirty": True, "git_dirty_files": ["config.yaml"]}
        ) == "dirty (1 uncommitted file)"

    def test_dirty_without_a_file_list_still_reports_dirty(self):
        assert publish_hf.code_state({"git_dirty": True}) == "dirty"

    def test_clean_tree(self):
        assert publish_hf.code_state(
            {"git_dirty": False, "git_dirty_files": []}) == "clean"

    def test_missing_fields_are_unknown_not_clean(self):
        """A manifest predating the fields must not be advertised as clean."""
        assert publish_hf.code_state({"git_commit": "abc1234"}) == "unknown"


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

    def test_intro_leads_with_what_it_is_and_where_it_came_from(self, tmp_path):
        # The card's hand-written half lives in build_card because the card is
        # regenerated whole on every publish; the Hub's editor would be overwritten.
        card = publish_hf.build_card(self._two(tmp_path), "cc-by-4.0", "repo-name")
        assert "Synthetic training data that teaches a model to reason carefully" in card
        assert "pretraining-style documents" in card
        assert "single-turn chat exchanges" in card
        assert "Teaching Claude Why" in card
        # source names the repo, and leads rather than trailing the audit sections
        assert f"[{publish_hf.REPO_NAME}]({publish_hf.REPO_URL})" in card
        assert card.index("## Source") < card.index("## DAD corpus")

    def test_intro_names_only_the_corpora_being_published(self, tmp_path):
        # a publish whose sibling is absent (or a --dry-run, which cannot see it)
        run, corpus = make_run_dir(tmp_path, pipeline="dad")
        staged, ddir = _stage(run, corpus, tmp_path / "stage_solo")
        card = _one_card(ddir, staged)
        assert "single-turn chat exchanges" in card
        assert "pretraining-style documents" not in card

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

    def test_norwegian_code_survives_yaml_round_trip(self, tmp_path):
        """Regression (found on the live published card): hand-built
        frontmatter lines emitted a bare `- no` for Norwegian, which YAML
        parses as the boolean False — so the published language list was
        malformed. The whole block goes through a real YAML emitter now."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, audit_files=[], include_html=False,
            extra_audit_files={"audit_report.json": {
                "n_docs": 5,
                "composition": {"language": {"Norwegian": 3, "English": 2}},
            }},
        )
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        card = _one_card(dataset_dir, staged)
        fm = yaml.safe_load(card.split("---\n")[1])
        assert fm["language"] == ["en", "no"]
        assert all(isinstance(c, str) for c in fm["language"]), fm["language"]
        assert False not in fm["language"]

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
            "/tmp/staged", "sentientfutures/x", "msg",
            ["sdf/audit/*", "sdf/card_meta.json"])
        assert result == "fake-commit"
        # forwarded verbatim — the wrapper adds nothing of its own
        assert calls[0]["delete_patterns"] == ["sdf/audit/*", "sdf/card_meta.json"]
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
        # run_manifest.json + manifests/* are both cleared so a publish that
        # switches layout (single-run <-> combined) can't leave the other
        # layout's manifest file(s) behind; upload_folder keeps freshly staged
        # paths, so the staged layout survives its own pattern.
        assert upload["delete_patterns"] == [
            "dad/audit/*", "dad/run_manifest.json", "dad/manifests/*",
            "dad/card_meta.json"]
        # every pattern must stay under this pipeline's own prefix
        assert all(p.startswith("dad/") for p in upload["delete_patterns"])

    def test_card_meta_is_cleared_so_a_stale_heading_cannot_survive(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """Regression: card_meta.json lives outside audit/, so the audit-only
        delete pattern left it behind. Publish run A with a curated title, then
        run B without one, and run A's sidecar would linger on the Hub — the
        next sibling publish would restore a title that is no longer what's
        published. It must be in delete_patterns even on a run that writes no
        sidecar of its own.

        Deleting unconditionally is safe because upload_folder drops any
        deletion whose path is also being added (verified against the installed
        huggingface_hub), so a freshly staged sidecar still survives."""
        # run B: no report_content.json, so no sidecar is staged
        run_dir, _ = make_run_dir(tmp_path, audit_files=[], include_html=False)
        staging_dir = tmp_path / "staged"
        calls = stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir))
        assert not (staging_dir / "sdf" / "card_meta.json").exists()
        upload = next(c for c in calls if c["fn"] == "upload_folder")
        assert "sdf/card_meta.json" in upload["delete_patterns"]

    def test_card_meta_still_uploaded_when_the_run_has_one(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """The other half of the above: the delete pattern is present, but the
        sidecar is also staged, so upload_folder's add-wins-over-delete rule
        keeps it."""
        run_dir, _ = make_run_dir(tmp_path)  # includes report_content.json
        staging_dir = tmp_path / "staged"
        calls = stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir))
        assert (staging_dir / "sdf" / "card_meta.json").exists()
        upload = next(c for c in calls if c["fn"] == "upload_folder")
        assert "sdf/card_meta.json" in upload["delete_patterns"]

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
                  "--repo-id", "sentientfutures/animal-welfare-training-dataset",
                  "--staging-dir", str(staging_dir))
        card = (staging_dir / "README.md").read_text()
        assert "pretty_name: animal-welfare-training-dataset" in card

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
    "sdf/card_meta.json": {"title": REPORT_CONTENT["title"],
                           "subtitle": REPORT_CONTENT["subtitle"]},
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
        # curated heading restored from the card_meta.json sidecar
        assert f"## {REPORT_CONTENT['title']} (`sdf` config)" in card
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
            "sdf/card_meta.json",
            "sdf/audit/audit_report.json",
            "sdf/audit/diversity_report.json",
        }
        assert not any(f.endswith((".jsonl", ".html")) for f in downloaded)

    def test_sibling_keeps_its_curated_heading(self, tmp_path, monkeypatch, stub_hf):
        """Regression (seen on the live card): report_content.json is never
        uploaded, so without the card_meta.json sidecar the sibling's curated
        title and subtitle were unrecoverable and its section silently
        downgraded to the generic 'SDF corpus' every time DAD was published."""
        _, _, card = self._publish_dad(tmp_path, monkeypatch, stub_hf, SIBLING_SDF_FILES)
        assert f"## {REPORT_CONTENT['title']} (`sdf` config)" in card
        assert REPORT_CONTENT["subtitle"] in card

    def test_sibling_without_card_meta_falls_back_to_generic_heading(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """A sibling published before the sidecar existed has no card_meta.json;
        it must still render, just with the generic heading."""
        files = {k: v for k, v in SIBLING_SDF_FILES.items() if k != "sdf/card_meta.json"}
        _, _, card = self._publish_dad(tmp_path, monkeypatch, stub_hf, files)
        assert "## SDF corpus (`sdf` config)" in card

    def test_card_meta_sidecar_is_written_for_the_published_pipeline(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """The pipeline being published writes its own sidecar so the NEXT
        publish of the other pipeline can restore this heading."""
        run_dir, _ = make_run_dir(tmp_path)  # sdf, includes report_content.json
        staging_dir = tmp_path / "staged"
        stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir))
        meta = json.loads((staging_dir / "sdf" / "card_meta.json").read_text())
        assert meta == {"title": REPORT_CONTENT["title"],
                        "subtitle": REPORT_CONTENT["subtitle"]}
        # the large editorial source itself still never ships
        assert not (staging_dir / "sdf" / "audit" / "report_content.json").exists()

    def test_no_card_meta_written_when_run_has_no_report_content(
        self, tmp_path, monkeypatch, stub_hf
    ):
        run_dir, _ = make_run_dir(tmp_path, audit_files=[], include_html=False)
        staging_dir = tmp_path / "staged"
        stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir))
        assert not (staging_dir / "sdf" / "card_meta.json").exists()

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

    def test_sibling_download_failure_keeps_config_and_does_not_abort(
        self, tmp_path, monkeypatch, stub_hf, capsys
    ):
        """A transient failure fetching the sibling's metadata must not abort
        the publish (create_repo has already run and this pipeline's corpus is
        staged and valid) and must not drop the sibling's config entry either —
        its files stay on the Hub, so removing the config would leave them
        present but unloadable. Only the prose detail may degrade."""
        calls = stub_hf(repo_files=SIBLING_SDF_FILES)

        from evals import publish_hf as ph
        real_download = ph._download_file

        def flaky(repo_id, filename, local_dir):
            if filename.endswith("audit_report.json"):
                raise OSError("transient network blip")
            return real_download(repo_id, filename, local_dir)

        monkeypatch.setattr(ph, "_download_file", flaky)

        run_dir, _ = make_run_dir(
            tmp_path, pipeline="dad", docs=40, audit_files=[], include_html=False,
            extra_audit_files={"audit_report.json": {"n_prompts": 40}},
        )
        staging_dir = tmp_path / "staged"
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir))

        out = capsys.readouterr().out
        assert "could not fetch" in out and "OSError" in out
        # the publish still completed
        assert any(c["fn"] == "upload_folder" for c in calls)

        card = (staging_dir / "README.md").read_text()
        fm = yaml.safe_load(card.split("---\n")[1])
        # sdf's config entry survives — that's what keeps its data loadable
        assert [c["config_name"] for c in fm["configs"]] == ["sdf", "dad"]
        assert fm["configs"][0]["data_files"][0]["path"] == "sdf/sdf_corpus.jsonl"
        # card_meta.json downloaded fine, so the curated heading survives
        assert f"## {REPORT_CONTENT['title']} (`sdf` config)" in card
        # the metrics row sourced from the file that failed is gone...
        assert "Documents (offline audit)" not in card
        # ...the file that DID download still contributes its row...
        assert "Vendi 34.5" in card
        # ...and the record count survives anyway, because fetch_sibling falls
        # back from audit_report.json's n_docs to diversity_report's n_records
        assert "477 documents." in card

    def test_sibling_listing_failure_treats_it_as_absent(self, tmp_path, monkeypatch, stub_hf):
        """If the repo can't even be listed (e.g. it doesn't exist yet on a
        first publish) there's nothing to preserve, so proceed single-config."""
        calls = stub_hf(repo_files=SIBLING_SDF_FILES)

        from evals import publish_hf as ph

        def boom(repo_id):
            raise OSError("cannot reach hub")

        monkeypatch.setattr(ph, "_list_repo_files", boom)

        run_dir, _ = make_run_dir(tmp_path, pipeline="dad", audit_files=[],
                                  include_html=False)
        staging_dir = tmp_path / "staged"
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir))
        assert any(c["fn"] == "upload_folder" for c in calls)
        card = (staging_dir / "README.md").read_text()
        fm = yaml.safe_load(card.split("---\n")[1])
        assert [c["config_name"] for c in fm["configs"]] == ["dad"]

    def test_sibling_dir_without_a_corpus_is_skipped(self, tmp_path, monkeypatch, stub_hf):
        """A partial sibling dir with metadata but no corpus can't be declared
        as a config — better to omit it than point a config at nothing."""
        _, _, card = self._publish_dad(
            tmp_path, monkeypatch, stub_hf,
            {"sdf/run_manifest.json": MANIFEST, "sdf/audit/audit_report.json": AUDIT_REPORT},
        )
        fm = yaml.safe_load(card.split("---\n")[1])
        assert [c["config_name"] for c in fm["configs"]] == ["dad"]


class TestUnmergedGuard:
    """The pre-flight provenance gate. It warns and asks rather than refusing:
    the HF write token lives on contributors' laptops, so a hard block would
    push an unmerged publish out of this script — and out of the only place
    that records provenance at all. What makes it stick is the card stamp.
    """

    def _unmerged(self, monkeypatch, **over):
        state = unmerged_state(**over)
        monkeypatch.setattr(publish_hf, "merge_state",
                            lambda commit, fetch=True: dict(state))
        return state

    def test_merged_run_publishes_silently(self, tmp_path, monkeypatch, stub_hf, capsys):
        """The default path must stay quiet — a warning that also fires on
        merged runs is one people learn to type straight past."""
        run_dir, _ = make_run_dir(tmp_path)
        calls = stub_hf()
        staging_dir = tmp_path / "staged"
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir))

        err = capsys.readouterr().err
        assert "NOT been merged" not in err
        card = (staging_dir / "README.md").read_text()
        assert "unmerged branch" not in card.lower()
        sidecar = json.loads((staging_dir / "sdf" / "card_meta.json").read_text())
        assert "unmerged" not in sidecar
        upload = next(c for c in calls if c["fn"] == "upload_folder")
        assert upload["commit_message"] == \
            "Publish sdf: 2026-07-25_15-57_fullscale-500-opus5"

    def test_non_interactive_without_flag_refuses_before_any_hub_call(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """An agent, a pipe, or a CI job has nobody to answer a prompt. It must
        exit naming the flag, and must not have touched the Hub first."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        self._unmerged(monkeypatch)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

        with pytest.raises(SystemExit) as excinfo:
            _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo")
        assert "--allow-unmerged" in str(excinfo.value)

    def test_interactive_yes_proceeds(self, tmp_path, monkeypatch, stub_hf):
        run_dir, _ = make_run_dir(tmp_path)
        calls = stub_hf()
        self._unmerged(monkeypatch)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda _prompt: "yes")

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo")
        assert any(c["fn"] == "upload_folder" for c in calls)

    def test_interactive_anything_else_aborts_with_no_hub_call(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """Only the exact word publishes. 'y' is a reflex; 'yes' is a decision."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        self._unmerged(monkeypatch)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")

        with pytest.raises(SystemExit):
            _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo")

    def test_allow_unmerged_publishes_and_stamps_everything(
        self, tmp_path, monkeypatch, stub_hf, capsys
    ):
        """The durable half of the guard: the card, the persisted sidecar, and
        the Hub commit message all record that this was unmerged."""
        run_dir, _ = make_run_dir(tmp_path)
        calls = stub_hf()
        staging_dir = tmp_path / "staged"
        self._unmerged(monkeypatch, branch="declan/wip", commit="deadbee")

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir), "--allow-unmerged")

        err = capsys.readouterr().err
        assert "NOT been merged" in err
        assert "declan/wip" in err

        card = (staging_dir / "README.md").read_text()
        assert "Unmerged code warning" in card
        assert "`declan/wip`" in card and "`deadbee`" in card

        sidecar = json.loads((staging_dir / "sdf" / "card_meta.json").read_text())
        assert sidecar["unmerged"]["runs"] == [{
            "run_id": "2026-07-25_15-57_fullscale-500-opus5",
            "branch": "declan/wip", "commit": "deadbee"}]

        upload = next(c for c in calls if c["fn"] == "upload_folder")
        assert "unmerged run(s): 2026-07-25_15-57_fullscale-500-opus5" \
            in upload["commit_message"]

    def test_stamp_survives_a_run_with_no_curated_title(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """The sidecar used to be written only when a run had a curated
        title/subtitle. The stamp has to be written regardless, or a run
        without report_content.json publishes with no warning on its card."""
        run_dir, _ = make_run_dir(tmp_path, audit_files=[], include_html=False)
        stub_hf()
        staging_dir = tmp_path / "staged"
        self._unmerged(monkeypatch)

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir), "--allow-unmerged")

        sidecar = json.loads((staging_dir / "sdf" / "card_meta.json").read_text())
        assert sidecar["unmerged"]["runs"][0]["branch"] == "declan/wip"
        assert "title" not in sidecar
        assert "Unmerged code warning" in \
            (staging_dir / "README.md").read_text()

    def test_dry_run_shows_the_warning_and_stamp_without_prompting(
        self, tmp_path, monkeypatch, stub_hf, capsys
    ):
        """A preview that hid the warning would be the wrong preview — but
        --dry-run publishes nothing, so there is nothing to confirm."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        self._unmerged(monkeypatch)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input",
                            lambda _prompt: pytest.fail("--dry-run must not prompt"))

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--dry-run")

        captured = capsys.readouterr()
        assert "NOT been merged" in captured.err
        assert "Unmerged code warning" in captured.out

    def test_dry_run_does_not_contact_the_remote(self, tmp_path, monkeypatch, stub_hf):
        """--dry-run is documented as making zero network calls, and a git fetch
        would break that promise just as surely as a Hub call."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        seen = {}

        def record(commit, fetch=True):
            seen["fetch"] = fetch
            return dict(MERGED_STATE)

        monkeypatch.setattr(publish_hf, "merge_state", record)

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--dry-run")
        assert seen["fetch"] is False

    def test_unknown_provenance_is_treated_as_unmerged(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """An unverifiable claim is not a safe one: a manifest with no git
        commit, or a commit this clone has never seen, must warn rather than
        sail through."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        self._unmerged(monkeypatch, head_merged=True, run_commit_merged=None,
                       notes=["commit deadbee is not in this clone (never pushed?)"])
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

        with pytest.raises(SystemExit):
            _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo")

    def test_unverifiable_is_not_reported_as_definitely_unmerged(
        self, tmp_path, monkeypatch, stub_hf, capsys
    ):
        """Both verdicts block, but they must not read the same. Claiming a run
        is unmerged when the truth is "couldn't tell" teaches people the warning
        is inaccurate, which is how a guardrail loses its authority."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf()
        self._unmerged(monkeypatch, head_merged=True, run_commit_merged=None,
                       notes=["commit deadbee is not in this clone"])

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--allow-unmerged")
        err = capsys.readouterr().err
        assert "could NOT be verified against main" in err
        assert "has NOT been merged" not in err

    def test_notes_reach_the_operator(self, tmp_path, monkeypatch, stub_hf, capsys):
        """merge_state's plain-English reasons are the only explanation of an
        unknown verdict, so they must be printed, not swallowed."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf()
        self._unmerged(monkeypatch, head_merged=True, run_commit_merged=None,
                       notes=["this clone has no origin/main reference to compare against"])

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--allow-unmerged")
        assert "no origin/main reference" in capsys.readouterr().err

    def test_stamp_names_the_branch_the_data_was_generated_on(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """A run generated on one branch can be published from another, and the
        card must keep the two straight: the RUN entry names the branch the data
        came from (v3 manifests' git_branch), while an unmerged checkout is
        reported separately as the publish branch. Collapsing them would let a
        reader think the corpus was generated by whatever happens to be checked
        out now."""
        run_dir, _ = make_run_dir(
            tmp_path,
            manifest={**MANIFEST, "git_branch": "aidan/local-only",
                      "git_commit": "cafe123"},
        )
        stub_hf()
        staging_dir = tmp_path / "staged"
        self._unmerged(monkeypatch, branch="declan/publishing-from-here",
                       commit="cafe123")

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir), "--allow-unmerged")

        card = (staging_dir / "README.md").read_text()
        # The run's own line credits where the data was generated...
        assert "Run `2026-07-25_15-57_fullscale-500-opus5` (branch " \
               "`aidan/local-only`, commit `cafe123`)" in card
        # ...and the publish branch is a separate statement, not conflated with it.
        assert "Published from branch `declan/publishing-from-here`" in card

        sidecar = json.loads((staging_dir / "sdf" / "card_meta.json").read_text())
        assert sidecar["unmerged"]["runs"][0]["branch"] == "aidan/local-only"
        assert sidecar["unmerged"]["publish_branch"] == "declan/publishing-from-here"

    def test_pre_v3_manifest_falls_back_to_the_live_branch(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """No existing manifest records git_branch, so the fallback is the
        common case, not an edge case."""
        run_dir, _ = make_run_dir(tmp_path)   # MANIFEST has no git_branch
        stub_hf()
        staging_dir = tmp_path / "staged"
        self._unmerged(monkeypatch, branch="declan/wip")

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir), "--allow-unmerged")
        assert "`declan/wip`" in (staging_dir / "README.md").read_text()

    def _dad_runs(self, tmp_path, *specs):
        """Several DAD run dirs, one per (run_name, commit, branch) spec."""
        dirs = []
        for run_name, commit, branch in specs:
            rd, _ = make_run_dir(
                tmp_path, pipeline="dad", audit_files=[], include_html=False,
                run_name=run_name,
                manifest={**MANIFEST, "run_id": run_name,
                          "git_commit": commit, "git_branch": branch},
            )
            dirs.append(rd)
        return dirs

    def test_combined_publish_names_only_the_unmerged_runs(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """The point of per-run stamping: a combined corpus is only as merged as
        its least-merged run, and a reader can trace a row to its run through
        the repo (example_gid) — so the warning must say WHICH runs, or that
        trace can't tell them whether a given row's code was reviewed. The
        merged run must not be smeared with the others' warning."""
        runs = self._dad_runs(
            tmp_path,
            ("2026-07-01_10-00_merged", "aaaaaaa", "main"),
            ("2026-07-02_10-00_wip", "bbbbbbb", "aidan/wip"),
            ("2026-07-03_10-00_other", "ccccccc", "constance/other"),
        )
        staging_dir = tmp_path / "staged"
        stub_hf()

        # HEAD is clean; only the 2nd and 3rd runs' commits are unmerged.
        def per_run(commit, fetch=True):
            merged = commit == "aaaaaaa"
            return {**MERGED_STATE, "run_commit": commit,
                    "run_commit_merged": merged}

        monkeypatch.setattr(publish_hf, "merge_state", per_run)

        _run_main(monkeypatch, "--input", *[str(r) for r in runs],
                  "--repo-id", "org/repo", "--staging-dir", str(staging_dir),
                  "--allow-unmerged")

        sidecar = json.loads((staging_dir / "dad" / "card_meta.json").read_text())
        assert [r["run_id"] for r in sidecar["unmerged"]["runs"]] == [
            "2026-07-02_10-00_wip", "2026-07-03_10-00_other"]
        # HEAD was merged, so there is no publish-branch line to add.
        assert "publish_branch" not in sidecar["unmerged"]

        card = (staging_dir / "README.md").read_text()
        warning = [ln for ln in card.splitlines() if ln.startswith(">")]
        assert any("`aidan/wip`" in ln for ln in warning)
        assert any("`constance/other`" in ln for ln in warning)
        # The merged run is not smeared with the warning...
        assert not any("2026-07-01_10-00_merged" in ln for ln in warning)
        # ...but still appears in the per-run provenance table, which covers all
        # three regardless of merge status.
        assert "| `2026-07-01_10-00_merged` |" in card

    def test_combined_publish_stays_silent_when_every_run_is_merged(
        self, tmp_path, monkeypatch, stub_hf, capsys
    ):
        runs = self._dad_runs(
            tmp_path,
            ("2026-07-01_10-00_a", "aaaaaaa", "main"),
            ("2026-07-02_10-00_b", "bbbbbbb", "main"),
        )
        staging_dir = tmp_path / "staged"
        stub_hf()

        _run_main(monkeypatch, "--input", *[str(r) for r in runs],
                  "--repo-id", "org/repo", "--staging-dir", str(staging_dir))

        assert "NOT been merged" not in capsys.readouterr().err
        # These runs have no curated title either, so with nothing to stamp the
        # sidecar is correctly never written at all.
        assert not (staging_dir / "dad" / "card_meta.json").exists()
        assert "Unmerged code warning" not in (staging_dir / "README.md").read_text()

    def test_combined_publish_fetches_the_remote_only_once(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """merge_state fetches origin/main; doing that once per run dir would
        make a 10-run publish hit the network 10 times for the same answer."""
        runs = self._dad_runs(
            tmp_path,
            ("2026-07-01_10-00_a", "aaaaaaa", "main"),
            ("2026-07-02_10-00_b", "bbbbbbb", "main"),
            ("2026-07-03_10-00_c", "ccccccc", "main"),
        )
        stub_hf()
        fetches = []

        def record(commit, fetch=True):
            fetches.append(fetch)
            return dict(MERGED_STATE)

        monkeypatch.setattr(publish_hf, "merge_state", record)

        _run_main(monkeypatch, "--input", *[str(r) for r in runs],
                  "--repo-id", "org/repo", "--staging-dir", str(tmp_path / "s"))
        assert fetches == [True, False, False]

    def test_siblings_own_stamp_survives_the_other_pipeline_publishing(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """The card is regenerated whole from the sibling's Hub metadata. A
        stamp derived from live git would both mislabel the sibling and erase
        its own warning — which is why it rides in card_meta.json."""
        run_dir, _ = make_run_dir(tmp_path, pipeline="dad", audit_files=[],
                                  include_html=False)
        staging_dir = tmp_path / "staged"
        sibling = dict(SIBLING_SDF_FILES)
        sibling["sdf/card_meta.json"] = {
            **SIBLING_SDF_FILES["sdf/card_meta.json"],
            "unmerged": {"runs": [{"run_id": "sdf-run",
                                   "branch": "aidan/experiment",
                                   "commit": "cafe123"}]},
        }
        stub_hf(repo_files=sibling)

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir))

        card = (staging_dir / "README.md").read_text()
        assert "`aidan/experiment`" in card
        # ...and the dad section being published, which IS merged, stays clean.
        assert card.count("Unmerged code warning") == 1
