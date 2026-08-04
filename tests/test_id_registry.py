"""Stable content-keyed id registry: assignment, reuse, persistence."""

import json

import pytest

from dad_pipeline import id_registry
from dad_pipeline.id_registry import (IdRegistry, example_fingerprint, prompt_fingerprint,
                                      registry_path, response_fingerprint,
                                      scenario_fingerprint)


def test_assigns_incrementing_ids_and_reuses_same_content(tmp_path):
    reg = IdRegistry(tmp_path / "id_registry.json")
    assert reg.assign("scenario", "fp-A") == 1
    assert reg.assign("scenario", "fp-B") == 2
    assert reg.assign("scenario", "fp-A") == 1   # seen content keeps its id
    assert reg.assign("scenario", "fp-C") == 3   # new content counts up


def test_kinds_are_independent(tmp_path):
    reg = IdRegistry(tmp_path / "id_registry.json")
    assert reg.assign("scenario", "x") == 1
    assert reg.assign("prompt", "x") == 1        # separate id space per kind


def test_persists_and_keeps_counting_across_instances(tmp_path):
    path = tmp_path / "id_registry.json"
    reg = IdRegistry(path)
    reg.assign("scenario", "fp-A")
    reg.assign("scenario", "fp-B")
    reg.save()
    # a fresh instance (a later "run") loads the file and does not reset
    reg2 = IdRegistry(path)
    assert reg2.assign("scenario", "fp-A") == 1  # stable across runs
    assert reg2.assign("scenario", "fp-C") == 3  # keeps counting up


def test_fingerprints_ignore_ids_and_normalize_whitespace():
    s1 = {"scenario_id": "S-001", "scenario_gid": "S-0007", "domain": ["x"], "conflict": "c"}
    s2 = {"scenario_id": "S-999", "domain": ["x"], "conflict": "c"}
    assert scenario_fingerprint(s1) == scenario_fingerprint(s2)  # own ids don't affect identity
    assert prompt_fingerprint("hello   world\n") == prompt_fingerprint("hello world")


def test_corrupt_registry_fails_loudly(tmp_path):
    # DELIBERATE SPEC CHANGE [CQ 2.3], not a test fixed to match the code: this
    # used to be test_corrupt_registry_starts_fresh, pinning a silent reset to
    # {}. A reset re-issues P-0001/R-0001/... for new content, colliding with the
    # numbers those gids already name in committed runs, audits, and the viewer.
    # Refusing to run is the recoverable failure; the reset is not.
    path = tmp_path / "id_registry.json"
    path.write_text("not json{{")
    with pytest.raises(SystemExit, match=r"id_registry\.json"):
        IdRegistry(path)
    try:
        IdRegistry(path)
    except SystemExit as exc:
        assert "git checkout" in str(exc)   # tells the user how to restore it


def test_merge_conflicted_registry_names_the_conflict(tmp_path):
    path = tmp_path / "id_registry.json"
    path.write_text('<<<<<<< HEAD\n{"prompt": {"a": 1}}\n=======\n'
                    '{"prompt": {"b": 1}}\n>>>>>>> other\n')
    with pytest.raises(SystemExit, match="merge conflict"):
        IdRegistry(path)


def test_missing_registry_still_starts_empty(tmp_path):
    # the first-ever run has no file yet — that must stay a silent fresh start
    reg = IdRegistry(tmp_path / "nested" / "id_registry.json")
    assert reg.assign("prompt", "fp") == 1


def test_non_object_registry_fails_loudly(tmp_path):
    path = tmp_path / "id_registry.json"
    path.write_text('[{"prompt": {}}]')            # valid JSON, wrong shape
    with pytest.raises(SystemExit, match="not an object"):
        IdRegistry(path)


def test_non_numeric_table_value_fails_loudly(tmp_path):
    path = tmp_path / "id_registry.json"
    path.write_text('{"prompt": {"fp": "one"}}')
    with pytest.raises(SystemExit, match="not a number"):
        IdRegistry(path)


def _boom(*args, **kwargs):
    raise RuntimeError("disk went away")


@pytest.mark.parametrize("victim", ["replace", "fsync"])
def test_failed_save_leaves_the_previous_registry_intact(tmp_path, monkeypatch, victim):
    path = tmp_path / "id_registry.json"
    reg = IdRegistry(path)
    reg.assign("prompt", "fp-A")
    reg.save()
    before = path.read_text(encoding="utf-8")

    monkeypatch.setattr(id_registry.os, victim, _boom)
    reg.assign("prompt", "fp-B")
    with pytest.raises(RuntimeError):
        reg.save()

    assert path.read_text(encoding="utf-8") == before      # previous contents survive
    assert json.loads(before)["prompt"] == {"fp-A": 1}     # ...and still parse
    assert not list(tmp_path.glob("*.tmp"))                # no debris left behind


def test_unknown_kinds_survive_a_load_save_round_trip(tmp_path):
    # a registry written by a newer branch must not be pruned by an older checkout
    path = tmp_path / "id_registry.json"
    path.write_text(json.dumps({"prompt": {"fp-A": 1}, "future_kind": {"fp-Z": 7}}))
    reg = IdRegistry(path)
    reg.assign("prompt", "fp-B")
    reg.save()
    assert json.loads(path.read_text(encoding="utf-8"))["future_kind"] == {"fp-Z": 7}


def test_duplicate_numbers_warn_but_still_load(tmp_path, capsys):
    # a merge of two branches' registries can hand one number to two fingerprints;
    # renumbering would move committed gids, so this warns rather than dying
    path = tmp_path / "id_registry.json"
    path.write_text(json.dumps({"prompt": {"fp-A": 4, "fp-B": 4}}))
    reg = IdRegistry(path)
    assert "WARNING" in capsys.readouterr().err
    assert reg.assign("prompt", "fp-A") == 4      # committed numbers stay put
    assert reg.assign("prompt", "fp-C") == 5      # new ids allocate above the max


def test_gid_formats_with_kind_prefix_and_reuses(tmp_path):
    reg = IdRegistry(tmp_path / "id_registry.json")
    assert reg.gid("response", "fp-A") == "R-0001"
    assert reg.gid("plain", "fp-A") == "C-0001"    # separate id space per kind
    assert reg.gid("example", "fp-A") == "E-0001"
    assert reg.gid("response", "fp-A") == "R-0001"  # seen content keeps its id
    assert reg.gid("response", "fp-B") == "R-0002"
    assert reg.gid("scenario", "fp-A") == "S-0001"
    assert reg.gid("prompt", "fp-A") == "P-0001"


def test_response_and_example_fingerprints_normalize_whitespace():
    assert response_fingerprint("a  b\n") == response_fingerprint("a b")
    assert example_fingerprint("u ", "a\n") == example_fingerprint("u", "a")
    # the pair is ordered — a swapped user/assistant is a different example
    assert example_fingerprint("u", "a") != example_fingerprint("a", "u")


def test_registry_path_walks_up_to_the_runs_root(tmp_path):
    stage_dir = tmp_path / "outputs" / "dad" / "runs" / "2026-07-01_00-00_x" / "step2"
    assert registry_path(stage_dir) == tmp_path / "outputs" / "dad" / "id_registry.json"
    # non-standard layouts (bare tmp dirs in tests) keep the registry local
    assert registry_path(tmp_path / "bare") == tmp_path / "bare" / "id_registry.json"
