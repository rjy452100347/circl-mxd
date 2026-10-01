"""Standalone launcher for Route Studio.

Example:
    python -m tools.routeStudio --cfg classic_cn --map newcomer_training_ground_1
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from src.ui.route_studio import RouteStudioWindow
from src.utils.common import is_mac, load_yaml, override_cfg


def load_config(profile):
    cfg = load_yaml("config/config_default.yaml")
    if is_mac():
        cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
    if profile and profile != "default":
        cfg = override_cfg(cfg, load_yaml(f"config/config_{profile}.yaml"))
    return cfg


def build_parser():
    parser = argparse.ArgumentParser(description="路线录制与编辑")
    parser.add_argument("--cfg", default="classic_cn", help="配置名（不含 config_ 前缀）")
    parser.add_argument("--map", default="", help="启动后预选的地图名")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    window = RouteStudioWindow(
        cfg=load_config(args.cfg), initial_map=args.map,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
