from __future__ import annotations

from pathlib import Path


def writable_path(*parts) -> Path:
    """Return a project-relative writable path for source builds."""
    return Path(*parts)
