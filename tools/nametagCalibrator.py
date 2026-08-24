"""Locally capture a name-tag template and calibrate the player-foot offset."""

import argparse
import time
from pathlib import Path

import cv2
import yaml

from src.input.GameWindowCapturor import GameWindowCapturor
from src.utils.common import is_mac, load_yaml, override_cfg, prepare_game_frame


WINDOW = "NameTag Calibrator"


def _display_frame(frame, cfg):
    debug_size = cfg["game_window"].get("debug_size")
    if not debug_size:
        return frame.copy(), 1.0, 1.0
    display_h, display_w = (int(v) for v in debug_size)
    shown = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
    return shown, frame.shape[1] / display_w, frame.shape[0] / display_h


def _wait_for_foot_click(image, name_rect):
    state = {"point": None}

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["point"] = (x, y)

    cv2.setMouseCallback(WINDOW, on_mouse)
    while state["point"] is None:
        canvas = image.copy()
        x, y, w, h = name_rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            canvas, "Click the character's feet (Q: cancel)", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )
        cv2.imshow(WINDOW, canvas)
        if cv2.waitKey(20) & 0xFF in {ord("q"), 27}:
            return None
    return state["point"]


def _save_sample(profile_name, frame, rect, foot, scale_x, scale_y):
    x, y, w, h = rect
    px0, py0 = round(x * scale_x), round(y * scale_y)
    px1, py1 = round((x + w) * scale_x), round((y + h) * scale_y)
    foot_x, foot_y = round(foot[0] * scale_x), round(foot[1] * scale_y)
    crop = frame[py0:py1, px0:px1]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        raise ValueError("Selected name-tag rectangle is too small")

    profile_dir = Path("nametag") / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / "profile.yaml"
    if profile_path.exists():
        with profile_path.open("r", encoding="utf-8") as stream:
            profile = yaml.safe_load(stream) or {}
    else:
        profile = {
            "schema_version": 1,
            "settings": {
                "max_score": 0.30,
                "local_search_radius": 140,
                "global_refresh_frames": 30,
                "max_jump": 250,
                "jump_confirm_frames": 2,
                "jump_confirm_radius": 25,
                "max_missed_frames": 5,
                "edge_weight": 0.65,
            },
            "samples": [],
        }

    sample_name = f"sample_{len(profile.get('samples', [])) + 1:03d}.png"
    if not cv2.imwrite(str(profile_dir / sample_name), crop):
        raise OSError(f"Unable to save {sample_name}")
    profile.setdefault("samples", []).append({
        "file": sample_name,
        "player_offset": [foot_x - px0, foot_y - py0],
        "enabled": True,
    })
    with profile_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(profile, stream, allow_unicode=True, sort_keys=False)
    return profile_path, sample_name, (foot_x - px0, foot_y - py0)


def main(args):
    cfg = load_yaml("config/config_default.yaml")
    if is_mac():
        cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
    cfg = override_cfg(cfg, load_yaml(f"config/config_{args.cfg}.yaml"))
    capture = GameWindowCapturor(cfg)
    try:
        frozen = None
        while frozen is None:
            raw = capture.get_frame()
            frame = prepare_game_frame(raw, cfg) if raw is not None else None
            if frame is None:
                time.sleep(0.05)
                continue
            shown, scale_x, scale_y = _display_frame(frame, cfg)
            cv2.putText(
                shown, "SPACE: freeze  Q: cancel", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(20) & 0xFF
            if key == ord(" "):
                frozen = frame.copy()
                frozen_shown = shown.copy()
            elif key in {ord("q"), 27}:
                return 1

        rect = tuple(int(v) for v in cv2.selectROI(
            WINDOW, frozen_shown, showCrosshair=True, fromCenter=False
        ))
        if rect[2] <= 0 or rect[3] <= 0:
            return 1
        foot = _wait_for_foot_click(frozen_shown, rect)
        if foot is None:
            return 1
        profile_path, sample_name, offset = _save_sample(
            args.profile, frozen, rect, foot, scale_x, scale_y
        )
        print(f"Saved name-tag sample: {profile_path.parent / sample_name}")
        print(f"Saved profile: {profile_path}")
        print(f"Player foot offset: {offset}")
        return 0
    finally:
        capture.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="classic_cn_player")
    parser.add_argument("--cfg", default="classic_cn")
    raise SystemExit(main(parser.parse_args()))
