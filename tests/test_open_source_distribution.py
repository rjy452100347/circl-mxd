import inspect
from pathlib import Path
import subprocess

import yaml

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.engine.OpenVinoMonsterDetector import OpenVinoMonsterDetector
from src.ui.AutoBotController import AutoBotController


ROOT = Path(__file__).resolve().parents[1]


def _tracked_paths():
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8"
    )
    return [Path(line) for line in output.splitlines() if line]


def test_protected_runtime_surface_is_absent():
    assert not (ROOT / "license_server").exists()
    assert not (ROOT / "packaging").exists()
    assert not (ROOT / "src" / "licensing").exists()
    assert not (ROOT / "src" / "engine" / "ProtectedOpenVinoRuntime.py").exists()
    assert "license_guard" not in inspect.signature(AutoBotController).parameters
    assert "protected_model_key" not in inspect.signature(
        OpenVinoMonsterDetector
    ).parameters


def test_distribution_contains_no_runtime_image_or_model_assets():
    forbidden_directories = {
        "deployment", "maps", "media", "minimaps", "misc", "monster",
        "nametag", "numbers", "rune",
    }
    tracked = _tracked_paths()
    assert not any(
        path.parts and path.parts[0] in forbidden_directories
        for path in tracked
    )

    forbidden_suffixes = {
        ".bin", ".bmp", ".gif", ".ico", ".jpeg", ".jpg", ".onnx",
        ".pem", ".pfx", ".png", ".pt", ".pth",
    }
    found = [path for path in tracked if path.suffix.lower() in forbidden_suffixes]
    assert found == []


def test_default_deployment_path_is_portable():
    config = (ROOT / "config" / "config_default.yaml").read_text(
        encoding="utf-8"
    )
    assert "deployment/openvino_cpu_2class" in config
    assert ":/MapleStoryAssets" not in config


def test_missing_map_asset_fails_closed(tmp_path, monkeypatch):
    config = yaml.safe_load(
        (ROOT / "config" / "config_default.yaml").read_text(encoding="utf-8")
    )
    config["bot"]["mode"] = "normal"
    config["bot"]["map"] = "not_installed"
    monkeypatch.chdir(tmp_path)

    bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
    assert bot.load_config(config) == -1
