from __future__ import annotations

from pathlib import Path

from gritter.generation.citations import extract_citations, format_sources, verify_citations


def test_extract_citations_basic():
    text = "See src/auth/jwt.py:L15-45 for the implementation."
    assert extract_citations(text) == ["src/auth/jwt.py:L15-45"]


def test_extract_citations_deduplication():
    text = "src/auth/jwt.py:L15-45 is mentioned and again src/auth/jwt.py:L15-45 here."
    assert extract_citations(text) == ["src/auth/jwt.py:L15-45"]


def test_extract_citations_multiple():
    text = "First see src/auth/jwt.py:L15-45, then check src/models/user.py:L10."
    assert extract_citations(text) == ["src/auth/jwt.py:L15-45", "src/models/user.py:L10"]


def test_verify_citations_real_file_in_range(tmp_path: Path):
    (tmp_path / "foo.py").write_text("\n".join(f"line{i}" for i in range(1, 21)) + "\n")
    result = verify_citations(["foo.py:L1-20"], tmp_path)
    assert result == [("foo.py:L1-20", True)]


def test_verify_citations_out_of_range(tmp_path: Path):
    (tmp_path / "foo.py").write_text("line1\nline2\n")
    result = verify_citations(["foo.py:L1-20"], tmp_path)
    assert result == [("foo.py:L1-20", False)]


def test_verify_citations_missing_file(tmp_path: Path):
    result = verify_citations(["nope.py:L1-5"], tmp_path)
    assert result == [("nope.py:L1-5", False)]


def test_verify_citations_path_traversal_rejected(tmp_path: Path):
    outside = tmp_path.parent / "secret.py"
    outside.write_text("x\n" * 10)
    result = verify_citations(["../secret.py:L1-5"], tmp_path)
    assert result == [("../secret.py:L1-5", False)]


def test_verify_citations_single_line_form(tmp_path: Path):
    (tmp_path / "foo.py").write_text("only one line\n")
    result = verify_citations(["foo.py:L1"], tmp_path)
    assert result == [("foo.py:L1", True)]


def test_format_sources_marks_unverified():
    verified = [("src/auth/jwt.py:L15-45", True), ("src/fake.py:L1-5", False)]
    output = format_sources(verified)
    assert output.startswith("Sources:")
    assert "• src/auth/jwt.py:L15-45\n" in output + "\n"
    assert "src/fake.py:L1-5 [unverified]" in output


def test_format_sources_empty():
    assert format_sources([]) == ""
