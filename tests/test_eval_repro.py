"""Tests for governed_bi.eval.repro (corpus-versioning reproducibility metadata)."""

from __future__ import annotations

from pathlib import Path

from governed_bi.eval.repro import corpus_git_state


def test_corpus_git_state_on_real_repo():
    repo_root = Path(__file__).resolve().parents[1]
    state = corpus_git_state(repo_root)
    assert state["commit"] is not None
    assert len(state["commit"]) == 40  # a real git SHA
    assert state["dirty"] in (True, False)


def test_corpus_git_state_not_a_repo(tmp_path):
    state = corpus_git_state(tmp_path)
    assert state["commit"] is None
    assert state["dirty"] is None
    assert "error" in state


def test_corpus_git_state_dirty_when_uncommitted_change(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.yaml").write_text("x: 1\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    state = corpus_git_state(tmp_path)
    assert state["commit"] is not None
    assert state["dirty"] is False

    (corpus_dir / "a.yaml").write_text("x: 2\n")
    dirty_state = corpus_git_state(tmp_path)
    assert dirty_state["commit"] == state["commit"]  # same last commit
    assert dirty_state["dirty"] is True
