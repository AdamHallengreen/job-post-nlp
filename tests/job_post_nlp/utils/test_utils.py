import pytest

from job_post_nlp.utils.find_project_root import find_project_root


def test_find_project_root_with_git_marker(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    assert find_project_root(tmp_path) == tmp_path


def test_find_project_root_with_no_markers(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        _ = find_project_root(tmp_path)


def test_find_project_root_with_nested_structure(monkeypatch, tmp_path):
    marker_file = tmp_path / "pyproject.toml"
    marker_file.touch()
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    monkeypatch.chdir(nested_dir)
    assert find_project_root(nested_dir) == tmp_path


def test_find_project_root_with_custom_markers(monkeypatch, tmp_path):
    custom_marker = tmp_path / ".custom_marker"
    custom_marker.touch()
    monkeypatch.chdir(tmp_path)
    assert find_project_root(tmp_path, markers=[".custom_marker"]) == tmp_path
