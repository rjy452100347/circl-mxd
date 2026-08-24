from pathlib import Path

from src.app_paths import writable_path


def test_source_build_keeps_relative_paths():
    assert writable_path("log") == Path("log")
