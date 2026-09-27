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


def migrate_health_monitor_config(config):
    """Normalize the legacy global health switch into per-resource switches."""
    if not isinstance(config, dict):
        return config
    health = config.get("health_monitor")
    if not isinstance(health, dict):
        return config
    has_legacy_switch = "enable" in health
    legacy_value = health.get("enable", True)
    # YAML booleans must remain booleans.  Treat malformed strings such as
    # "false" as disabled rather than Python's truthy string value.
    legacy_enabled = legacy_value is True

    def threshold_is_enabled(key):
        try:
            return float(health[key]) > 0
        except (KeyError, TypeError, ValueError):
            return False

    for kind in ("hp", "mp"):
        enabled_key = f"auto_{kind}_enabled"
        threshold_key = f"add_{kind}_percent"
        if enabled_key in health:
            if not isinstance(health[enabled_key], bool):
                raise ValueError(
                    f"health_monitor.{enabled_key} 必须是布尔值 true/false。"
                )
            continue
        if has_legacy_switch:
            # An explicit old global switch applies to both resources.  If the
            # old profile omitted one threshold, keep the enabled state so the
            # merged default threshold remains effective.
            health[enabled_key] = (
                legacy_enabled and (
                    threshold_key not in health or
                    threshold_is_enabled(threshold_key)
                )
            )
        elif threshold_key in health:
            # Partial legacy profiles commonly override HP only.  Migrate only
            # fields they actually own; omitted fields must still inherit the
            # current base profile during override_cfg().
            health[enabled_key] = threshold_is_enabled(threshold_key)
    health.pop("enable", None)
    return config
