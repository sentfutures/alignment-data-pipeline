"""Tests for shared.utils.merge_state — the "is this run's code in main?"
provenance check behind publish_hf's unmerged-publish guard.

These drive real git against throwaway repos built under tmp_path, with a local
directory standing in for "origin", so nothing here touches the network
(pytest-socket would block it anyway) and nothing depends on the branch the
developer happens to be on.
"""

import subprocess

import pytest

from shared import utils


def _git(cwd, *args, check=True):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                            text=True, encoding="utf-8")
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
    return result


def _identify(repo):
    """CI runners have no global git identity, and a signing config inherited
    from the developer's machine would prompt — pin both per repo."""
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")


def _commit(repo, text):
    (repo / "f.txt").write_text(text, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", text)
    return _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()


@pytest.fixture
def clone(tmp_path):
    """A clone whose origin/main holds one commit. Returns (work_dir, sha)."""
    origin = tmp_path / "origin"
    origin.mkdir()
    # -b main explicitly: the default branch name is a git config, not a given.
    _git(origin, "init", "-b", "main")
    _identify(origin)
    _commit(origin, "one")

    work = tmp_path / "work"
    _git(tmp_path, "clone", str(origin), str(work))
    _identify(work)
    return work, _git(work, "rev-parse", "--short", "HEAD").stdout.strip()


class TestMerged:
    def test_clean_clone_at_origin_main_is_merged(self, clone):
        work, sha = clone
        state = utils.merge_state(sha, fetch=False, repo=work)
        assert state["head_merged"] is True
        assert state["run_commit_merged"] is True
        assert state["ahead"] == 0
        assert state["branch"] == "main"

    def test_fetch_refreshes_against_a_local_origin(self, clone):
        """The fetch path must work, not just be skippable — a fetch that always
        failed would silently mark every publish's origin/main as stale."""
        work, sha = clone
        state = utils.merge_state(sha, fetch=True, repo=work)
        assert state["fetched"] is True
        assert not any("out of date" in n for n in state["notes"])

    def test_no_fetch_records_that_origin_main_may_be_stale(self, clone):
        work, sha = clone
        state = utils.merge_state(sha, fetch=False, repo=work)
        assert any("may be out of date" in n for n in state["notes"])


class TestUnmerged:
    def test_local_commit_ahead_of_origin_main(self, clone):
        work, _ = clone
        new_sha = _commit(work, "two")
        state = utils.merge_state(new_sha, fetch=False, repo=work)
        assert state["head_merged"] is False
        assert state["run_commit_merged"] is False
        assert state["ahead"] == 1

    def test_merged_run_commit_published_from_an_unmerged_checkout(self, clone):
        """The two checks are independent: a run generated from merged code can
        still be published from a branch that has moved on."""
        work, merged_sha = clone
        _commit(work, "two")
        state = utils.merge_state(merged_sha, fetch=False, repo=work)
        assert state["head_merged"] is False
        assert state["run_commit_merged"] is True


class TestUnknownIsNotMerged:
    def test_commit_absent_from_this_clone(self, clone):
        """A commit that only ever existed on someone's laptop can't be checked
        at all — which must read differently from "on a branch"."""
        work, _ = clone
        state = utils.merge_state("deadbee", fetch=False, repo=work)
        assert state["run_commit_merged"] is None
        assert any("not in this clone" in n for n in state["notes"])

    def test_manifest_with_no_git_commit(self, clone):
        work, _ = clone
        state = utils.merge_state(None, fetch=False, repo=work)
        assert state["run_commit_merged"] is None
        assert any("records no git commit" in n for n in state["notes"])

    def test_repo_without_origin_main(self, tmp_path):
        solo = tmp_path / "solo"
        solo.mkdir()
        _git(solo, "init", "-b", "main")
        _identify(solo)
        sha = _commit(solo, "one")
        state = utils.merge_state(sha, fetch=False, repo=solo)
        assert state["head_merged"] is None
        assert state["run_commit_merged"] is None
        assert any("no origin/main" in n for n in state["notes"])

    def test_not_a_git_checkout_at_all(self, tmp_path):
        bare = tmp_path / "not_git"
        bare.mkdir()
        state = utils.merge_state("4abd78b", fetch=False, repo=bare)
        assert state["head"] is None
        assert state["head_merged"] is None
        assert state["run_commit_merged"] is None
        assert any("not a git checkout" in n for n in state["notes"])
