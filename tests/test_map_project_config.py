from pathlib import Path

import pytest

from src.engine.MapProjectConfig import (
    MAP_CONFIG_FILENAME,
    MapProjectConfigError,
    apply_map_minimap_roi_override,
    load_map_minimap_roi,
    normalized_roi_to_pixels,
    pixels_to_normalized_roi,
    save_map_minimap_roi_atomic,
    validate_normalized_roi,
)


def test_roi_conversion_matches_runtime_rounding():
    roi = (0.0034163006, 0.0902777778, 0.1078574915, 0.1623263889)

    assert normalized_roi_to_pixels(roi, 2049, 1152) == (7, 104, 221, 187)
    assert normalized_roi_to_pixels(
        pixels_to_normalized_roi((7, 104, 221, 187), 2049, 1152),
        2049,
        1152,
    ) == (7, 104, 221, 187)


@pytest.mark.parametrize(
    "roi",
    (
        None,
        [0, 0, 1],
        [True, 0, 0.5, 0.5],
        [float("nan"), 0, 0.5, 0.5],
        [-0.1, 0, 0.5, 0.5],
        [0, 0, 0, 0.5],
        [0.8, 0, 0.3, 0.5],
    ),
)
def test_invalid_normalized_roi_is_rejected(roi):
    with pytest.raises(MapProjectConfigError):
        validate_normalized_roi(roi)


def test_map_roi_is_saved_loaded_and_can_restore_global(tmp_path):
    directory = tmp_path / "training"
    roi = (0.1, 0.2, 0.3, 0.4)

    path = save_map_minimap_roi_atomic(directory, roi)

    assert path == directory / MAP_CONFIG_FILENAME
    assert load_map_minimap_roi(directory) == roi
    save_map_minimap_roi_atomic(directory, None)
    assert path.exists()
    assert load_map_minimap_roi(directory) is None


def test_map_override_precedes_global_and_missing_map_inherits(tmp_path):
    cfg = {"minimap": {"roi": [0.0, 0.0, 0.5, 0.5]}}
    save_map_minimap_roi_atomic(
        tmp_path / "map_a", (0.1, 0.2, 0.3, 0.4)
    )

    assert apply_map_minimap_roi_override(cfg, "map_a", tmp_path) == "map:map_a"
    assert cfg["minimap"]["roi"] == [0.1, 0.2, 0.3, 0.4]

    inherited = {"minimap": {"roi": [0.2, 0.2, 0.2, 0.2]}}
    assert apply_map_minimap_roi_override(inherited, "map_b", tmp_path) == "global"
    assert inherited["minimap"]["roi"] == [0.2, 0.2, 0.2, 0.2]

    automatic = {"minimap": {}}
    assert apply_map_minimap_roi_override(automatic, "map_b", tmp_path) == "automatic"


@pytest.mark.parametrize(
    "content",
    (
        "- not\n- an\n- object\n",
        "schema_version: 99\nminimap: {}\n",
        "schema_version: 1\nminimap: []\n",
        "schema_version: 1\nminimap:\n  roi: [0.9, 0, 0.2, 0.5]\n",
    ),
)
def test_malformed_existing_map_config_fails_closed(tmp_path, content):
    directory = tmp_path / "training"
    directory.mkdir()
    (directory / MAP_CONFIG_FILENAME).write_text(content, encoding="utf-8")

    with pytest.raises(MapProjectConfigError):
        load_map_minimap_roi(directory)
