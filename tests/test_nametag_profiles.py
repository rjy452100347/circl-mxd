import copy
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.engine.NameTagLocalizer import NameTagLocalizer
from src.engine.NameTagProfileRepository import (
    NameTagProfileError,
    NameTagProfileRepository,
    validate_profile_name,
)
from src.ui.nametag_calibration import NameTagCalibrationDialog


def _name_image(width=48, height=14):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        image, "Hero", (1, height - 3), cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (245, 245, 245), 1, cv2.LINE_AA,
    )
    return image


def _write_profile(repository, name, *, enabled=True, count=1):
    stage = repository.create_stage(name, existing=False)
    for index in range(count):
        repository.stage_sample(
            stage, _name_image(48 + index, 14), (24, -12), enabled=enabled
        )
    repository.commit_profile(stage, require_enabled=enabled)
    return stage


@pytest.mark.parametrize("name", ["", "  ", ".", "..", "a/b", "a\\b", "CON", "abc."])
def test_profile_name_rejects_unsafe_windows_paths(name):
    with pytest.raises(NameTagProfileError):
        validate_profile_name(name)


def test_repository_supports_unicode_and_uses_max_sample_number(tmp_path):
    root = tmp_path / "人物名字"
    repository = NameTagProfileRepository(bundled_root=root, writable_root=root)
    stage = _write_profile(repository, "新角色", count=2)

    loaded = repository.load_profile("新角色", require_enabled=True)
    assert [entry["file"] for entry in loaded["samples"]] == [
        "sample_001.png", "sample_002.png"
    ]
    stage.data["samples"].pop(0)
    filename = repository.stage_sample(stage, _name_image(), (22, -10))
    assert filename == "sample_003.png"
    repository.commit_profile(stage, require_enabled=True)
    assert (root / "新角色" / "sample_003.png").is_file()


def test_user_profile_shadows_bundled_and_edit_clones_bundled_assets(tmp_path):
    bundled = tmp_path / "bundled"
    writable = tmp_path / "user"
    bundled_repo = NameTagProfileRepository(
        bundled_root=bundled, writable_root=bundled
    )
    _write_profile(bundled_repo, "classic", enabled=True)
    repository = NameTagProfileRepository(
        bundled_root=bundled, writable_root=writable
    )

    stage = repository.create_stage("classic", existing=True)
    repository.stage_player_offset(stage, "sample_001.png", (31, -9))
    target = repository.commit_profile(stage, require_enabled=True)

    assert target == writable / "classic"
    assert (target / "sample_001.png").is_file()
    assert repository.resolve_profile("classic") == target
    assert repository.load_profile("classic", require_enabled=True)["samples"][0][
        "player_offset"
    ] == [31, -9]


def test_repository_blocks_zero_enabled_samples_for_runtime(tmp_path):
    repository = NameTagProfileRepository(
        bundled_root=tmp_path / "assets", writable_root=tmp_path / "user"
    )
    stage = _write_profile(repository, "disabled", enabled=False)
    with pytest.raises(NameTagProfileError, match="至少需要一个已启用样本"):
        repository.load_profile("disabled", require_enabled=True)
    repository.stage_sample_enabled(stage, "sample_001.png", True)
    repository.commit_profile(stage, require_enabled=True)


def test_localizer_loads_unicode_profile_image_path(tmp_path):
    repository = NameTagProfileRepository(
        bundled_root=tmp_path / "资源", writable_root=tmp_path / "资源"
    )
    _write_profile(repository, "中文角色")
    localizer = NameTagLocalizer.from_profile(
        repository.resolve_profile("中文角色")
    )
    assert len(localizer.samples) == 1


class _FakeLock:
    def __init__(self, owner):
        self.owner = owner
        self.acquired = False

    def acquire(self):
        self.acquired = True

    def release(self):
        self.acquired = False


class _FakeCapture:
    def __init__(self, cfg):
        self.frame = np.zeros((100, 160, 3), dtype=np.uint8)
        self.stopped = False

    def get_frame(self):
        return self.frame.copy()

    def stop(self):
        self.stopped = True


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _dialog_cfg():
    return {
        "game_window": {
            "size": [100, 160],
            "title_bar_height": 0,
            "resize_on_start": False,
        },
        "nametag": {"name": "", "search_y_limit": 100},
        "ui_coords": {"ui_y_start": 100},
    }


def test_dialog_stages_validated_sample_and_cancel_has_no_disk_write(
    qt_app, tmp_path
):
    repository = NameTagProfileRepository(
        bundled_root=tmp_path / "assets", writable_root=tmp_path / "user"
    )
    dialog = NameTagCalibrationDialog(
        cfg=_dialog_cfg(), repository=repository,
        capture_factory=_FakeCapture, lock_factory=_FakeLock,
    )
    dialog.timer.stop()
    dialog.stage = repository.create_stage("新人物", existing=False)
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    sample = _name_image()
    frame[55:69, 30:78] = sample
    dialog.latest_frame = frame
    dialog.freeze_for_new_sample()
    dialog._name_rect_selected((30, 55, 48, 14))
    dialog._foot_selected((54, 43))
    for _ in range(dialog.VALIDATION_FRAMES):
        dialog._validation_tick(frame.copy())

    assert len(dialog.stage.data["samples"]) == 1
    assert dialog.stage.data["samples"][0]["enabled"] is True
    assert not (tmp_path / "user" / "新人物").exists()
    dialog.stage_dirty = False
    dialog.reject()


def test_dialog_failed_validation_stages_disabled_sample(qt_app, tmp_path):
    repository = NameTagProfileRepository(
        bundled_root=tmp_path / "assets", writable_root=tmp_path / "user"
    )
    dialog = NameTagCalibrationDialog(
        cfg=_dialog_cfg(), repository=repository,
        capture_factory=_FakeCapture, lock_factory=_FakeLock,
    )
    dialog.timer.stop()
    dialog.stage = repository.create_stage("低质量", existing=False)
    dialog._pending_kind = "new"
    dialog._pending_image = _name_image()
    dialog._pending_offset = (24, -12)
    dialog._validation_mode = "new"
    dialog._validation_attempts = 20
    dialog._validation_captured = 20
    dialog._validation_valid = 10
    dialog._validation_max_consecutive = 5
    dialog._validation_all_scores_ok = True
    dialog._validation_points_inside = True
    dialog._finish_validation(environment_error=False)

    entry = dialog.stage.data["samples"][0]
    assert entry["enabled"] is False
    assert entry["file"] in dialog._low_quality_samples
    dialog.stage_dirty = False
    dialog.reject()
