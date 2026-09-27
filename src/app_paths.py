from __future__ import annotations

import os
from pathlib import Path


def user_data_root() -> Path:
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(local) / "MapleAssistant"


def writable_path(*parts) -> Path:
    """Return a project-relative writable path for source builds."""
    return Path(*parts)
