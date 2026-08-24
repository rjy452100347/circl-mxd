import cv2
import numpy as np

from src.utils.common import (
    get_minimap_loc_size,
    get_player_location_on_minimap,
    prepare_game_frame,
)


def _cfg():
    return {
        "game_window": {
            "size": (1152, 2049),
            "title_bar_height": 0,
            "client_crop": (1, 48, 1, 0),
            "resize_on_start": False,
        },
        "minimap": {
            "roi": (0.0034163006, 0.0902777778, 0.1078574915, 0.1623263889),
        },
    }


def test_current_client_frame_is_not_cropped_or_resized():
    frame = np.zeros((1200, 2051, 3), dtype=np.uint8)

    prepared = prepare_game_frame(frame, _cfg())

    assert prepared.shape == (1152, 2049, 3)
    assert prepared is not frame


def test_current_client_wrong_size_is_rejected():
    assert prepare_game_frame(np.zeros((700, 1296, 3), dtype=np.uint8), _cfg()) is None


def test_normalized_minimap_roi_matches_verified_client_geometry():
    frame = np.zeros((1152, 2049, 3), dtype=np.uint8)

    assert get_minimap_loc_size(frame, _cfg()) == (7, 104, 221, 187)


def test_player_hsv_detection_accepts_current_yellow_marker():
    hsv = np.zeros((40, 60, 3), dtype=np.uint8)
    cv2.rectangle(hsv, (27, 17), (32, 22), (30, 255, 255), -1)
    minimap = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    player = get_player_location_on_minimap(
        minimap,
        player_hsv={"lower": (20, 100, 120), "upper": (40, 255, 255)},
    )

    assert player == (30, 20)


def test_player_hsv_detection_rejects_ambiguous_markers():
    hsv = np.zeros((40, 60, 3), dtype=np.uint8)
    cv2.rectangle(hsv, (7, 17), (12, 22), (30, 255, 255), -1)
    cv2.rectangle(hsv, (47, 17), (52, 22), (30, 255, 255), -1)
    minimap = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    assert get_player_location_on_minimap(
        minimap,
        player_hsv={"lower": (20, 100, 120), "upper": (40, 255, 255)},
    ) is None
