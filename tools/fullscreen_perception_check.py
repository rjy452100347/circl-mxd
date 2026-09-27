"""Text-only model check. Never constructs a keyboard controller or shows images."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.engine.FullscreenSelfDetector import detector_from_config, SelfLocalizationGate
from src.utils.common import load_yaml, override_cfg, prepare_game_frame


def check(cfg, *, live=False, frames=20, timeout=10):
    detector = detector_from_config(cfg)
    gate = SelfLocalizationGate()
    capture = None
    rows = []
    skipped = Counter()
    try:
        if live:
            from src.input.GameWindowCapturor import GameWindowCapturor
            cfg['game_window']['resize_on_start'] = False
            capture = GameWindowCapturor(cfg)
        deadline = time.monotonic()+timeout
        previous = None
        while len(rows) < frames and time.monotonic() < deadline:
            if live:
                packet = capture.get_frame_packet()
                if packet is None:
                    skipped['no_frame'] += 1
                    time.sleep(.02)
                    continue
                raw, captured_at, sequence = packet
                if sequence == previous:
                    time.sleep(.01)
                    continue
                previous = sequence
                if not 0 <= time.monotonic()-captured_at <= .3:
                    skipped['stale_capture'] += 1
                    gate.invalidate('frame_stale')
                    continue
                frame = prepare_game_frame(raw, cfg)
                if frame is None:
                    skipped['invalid_crop_or_size'] += 1
                    gate.invalidate('capture_invalid')
                    time.sleep(.02)
                    continue
            else:
                h, w = cfg['game_window']['size']
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                captured_at, sequence = time.monotonic(), len(rows)
            scene = detector.detect_scene(frame, frame_token=sequence)
            age = time.monotonic()-captured_at
            location = gate.update(scene, sequence) if age <= .3 else gate.invalidate('frame_stale')
            rows.append(dict(sequence=sequence, monster=len(scene.monsters),
                player=len(scene.players), self=len(scene.selves), foot=location.foot,
                self_confirmed=location.valid, reason=location.reason,
                infer_ms=round(scene.timing.infer_ms, 2),
                pipeline_ms=round(scene.timing.total_ms, 2), frame_age_ms=round(age*1000, 2)))
    finally:
        if capture is not None:
            capture.stop()
    timings = [r['pipeline_ms'] for r in rows]
    return dict(source='live_game' if live else 'blank_array', model=str(detector.root),
        variant=detector.variant, control_enabled=False, frames=len(rows),
        skipped=dict(skipped), mean_pipeline_ms=round(float(np.mean(timings)), 2) if rows else None,
        p95_pipeline_ms=round(float(np.percentile(timings, 95)), 2) if rows else None,
        observations=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/config_classic_cn.yaml')
    parser.add_argument('--live', action='store_true', help='Read game capture without input or image display')
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--timeout', type=float, default=10.)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.frames <= 0 or args.timeout <= 0:
        parser.error('frames and timeout must be positive')
    cfg = override_cfg(load_yaml('config/config_default.yaml'), load_yaml(args.config))
    try:
        report = check(cfg, live=args.live, frames=args.frames, timeout=args.timeout)
        exit_code = 0 if report['frames'] == args.frames else 2
    except Exception as exc:
        report = dict(source='live_game' if args.live else 'blank_array', control_enabled=False, error=str(exc))
        exit_code = 2
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text+'\n', encoding='utf-8')
    print(text)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
