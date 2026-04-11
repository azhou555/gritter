from pathlib import Path
import pytest
from gritter.utils.languages import detect_language
from gritter.indexing.discovery import discover_files


def test_detect_python():
    assert detect_language("src/foo.py") == "python"


def test_detect_typescript():
    assert detect_language("src/foo.ts") == "typescript"
    assert detect_language("src/foo.tsx") == "typescript"


def test_detect_rust():
    assert detect_language("src/foo.rs") == "rust"


def test_detect_unknown_returns_none():
    assert detect_language("src/foo.go") is None
    assert detect_language("src/foo.rb") is None


def test_discover_files_returns_supported_files(tmp_path):
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "lib.ts").write_text("export const x = 1;")
    (tmp_path / "notes.txt").write_text("some notes")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    results = discover_files(tmp_path)
    paths = {p.name for p, _ in results}
    assert "main.py" in paths
    assert "lib.ts" in paths
    assert "notes.txt" not in paths
    assert "image.png" not in paths


def test_discover_files_skips_excluded_dirs(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lodash.ts").write_text("export default {}")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.ts").write_text("export const app = 1;")
    results = discover_files(tmp_path)
    paths = {p.name for p, _ in results}
    assert "lodash.ts" not in paths
    assert "app.ts" in paths


def test_discover_files_respects_exclude_globs(tmp_path):
    (tmp_path / "main.py").write_text("x = 1")
    (tmp_path / "generated.py").write_text("x = 2")
    results = discover_files(tmp_path, exclude_globs=["*generated*"])
    paths = {p.name for p, _ in results}
    assert "main.py" in paths
    assert "generated.py" not in paths


def test_discover_files_skips_lockfiles(tmp_path):
    (tmp_path / "app.ts").write_text("export {}")
    (tmp_path / "package-lock.json").write_text("{}")
    results = discover_files(tmp_path)
    paths = {p.name for p, _ in results}
    assert "app.ts" in paths
    assert "package-lock.json" not in paths


def test_discover_files_skips_hidden_files(tmp_path):
    (tmp_path / ".env").write_text("SECRET=abc")
    (tmp_path / ".hidden.py").write_text("x = 1")
    (tmp_path / "visible.py").write_text("y = 2")
    results = discover_files(tmp_path)
    paths = {p.name for p, _ in results}
    assert "visible.py" in paths
    assert ".hidden.py" not in paths
    assert ".env" not in paths
