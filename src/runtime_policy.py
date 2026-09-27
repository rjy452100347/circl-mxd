"""Internal runtime policy.

These values are deliberately not configurable by user profiles.  Keeping
them in one immutable object prevents UI/config drift from changing capture,
control, or safety timing.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePolicy:
    main_fps: int = 10
    capture_fps: int = 15
    keyboard_fps: int = 30
    command_timeout_seconds: float = 0.5
    frame_stale_timeout_seconds: float = 0.75
    hotkey_debounce_seconds: float = 1.0
    route_recorder_fps: int = 10
    route_recorder_blob_cooldown: float = 0.7
    route_recorder_map_padding: int = 30
    preview_fps: int = 5
    preview_max_width: int = 1280
    auto_dice_fps: int = 1
    client_language: str = "cn"
    yolo_input_width: int = 1280
    yolo_input_height: int = 224
    yolo_max_det: int = 50


RUNTIME_POLICY = RuntimePolicy()
