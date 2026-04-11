# tests/conftest.py
from pathlib import Path
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def python_fixture_path() -> Path:
    return FIXTURES_DIR / "python_project" / "sample.py"


@pytest.fixture
def typescript_fixture_path() -> Path:
    return FIXTURES_DIR / "typescript_project" / "sample.ts"


@pytest.fixture
def rust_fixture_path() -> Path:
    return FIXTURES_DIR / "rust_project" / "sample.rs"
