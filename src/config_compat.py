"""Strict compatibility checks for user-supplied configuration files."""

from __future__ import annotations


REMOVED_SECTIONS = frozenset({
    "input", "buff_skill", "party_red_bar", "character", "email", "system",
    "route_recoder", "rune_warning_cn", "rune_warning_eng",
    "rune_enable_msg_cn", "rune_enable_msg_eng", "rune_detect", "rune_find",
    "rune_solver",
})

REMOVED_MONSTER_FIELDS = frozenset({
    "mode", "max_candidates_per_template", "template_scale",
    "debug_interval_seconds", "diff_thres", "diff_thres_by_monster",
    "search_box_margin", "search_range_x", "search_range_y", "contour_blur",
    "contour_black_max", "contour_working_scale", "contour_scales_by_monster",
    "with_enemy_hp_bar", "hp_bar_color", "max_mob_area_trigger",
    "health_bar_box_width", "health_bar_box_height",
    "pursuit_vertical_tolerance", "target_lost_grace_seconds",
})

REMOVED_KEY_FIELDS = frozenset({"party"})


class ConfigCompatibilityError(ValueError):
    """Raised when a custom profile still contains removed settings."""

    def __init__(self, paths):
        self.paths = tuple(sorted(set(paths)))
        detail = "\n".join(f"• {path}" for path in self.paths)
        super().__init__(f"配置包含已删除的项目，请先移除：\n{detail}")


def find_removed_config_paths(config):
    """Return every removed key path in *config*; do not stop at first error."""
    if not isinstance(config, dict):
        return ["<root>"]
    paths = [name for name in REMOVED_SECTIONS if name in config]
    nametag = config.get("nametag")
    if isinstance(nametag, dict) and "enable" in nametag:
        paths.append("nametag.enable")
    monster = config.get("monster_detect")
    if isinstance(monster, dict):
        paths.extend(
            f"monster_detect.{key}"
            for key in REMOVED_MONSTER_FIELDS
            if key in monster
        )
    key_cfg = config.get("key")
    if isinstance(key_cfg, dict):
        paths.extend(f"key.{key}" for key in REMOVED_KEY_FIELDS if key in key_cfg)
    return sorted(paths)


def validate_custom_config(config):
    paths = find_removed_config_paths(config)
    if paths:
        raise ConfigCompatibilityError(paths)
    return config
