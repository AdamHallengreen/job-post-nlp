from pathlib import Path

import pytest

from job_post_nlp.utils.find_project_root import find_project_root


def test_find_project_root_with_git_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    assert find_project_root(tmp_path) == tmp_path


def test_find_project_root_with_no_markers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        _ = find_project_root(tmp_path)


def test_find_project_root_with_nested_structure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    marker_file = tmp_path / "pyproject.toml"
    marker_file.touch()
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    monkeypatch.chdir(nested_dir)
    assert find_project_root(nested_dir) == tmp_path


def test_find_project_root_with_custom_markers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom_marker = tmp_path / ".custom_marker"
    custom_marker.touch()
    monkeypatch.chdir(tmp_path)
    assert find_project_root(tmp_path, markers=[".custom_marker"]) == tmp_path
