"""Chinese display strings for the Qt UI.

Configuration keys and values deliberately remain English.  Only the text
shown to the user is translated here so UI wording cannot change persistence.
"""

SECTION_LABELS = {
    "directional_attack": "方向攻击",
    "aoe_skill": "范围技能", "health_monitor": "生命与魔法监控", "teleport": "瞬移",
    "edge_teleport": "边缘瞬移", "nametag": "名字标签定位",
    "monster_detect": "YOLO 怪物检测", "combat_tracking": "追击策略", "channel_change": "自动换线",
    "scheduled_channel_switching": "定时换线", "ui_coords": "游戏界面坐标", "route": "路线跟随",
    "watchdog": "看门狗", "minimap": "小地图", "patrol": "巡逻",
    "fixed_platform": "固定平台区域攻击",
    "continuous_attack": "持续定向攻击",
    "game_window": "游戏窗口", "profiler": "性能分析",
}

FIELD_LABELS = {
    "cooldown": "冷却时间", "range_x": "横向范围", "range_y": "纵向范围",
    "character_turn_delay": "人物转向延迟", "enable": "启用",
    "force_heal": "生命值优先补给（遵守冷却）",
    "auto_hp_enabled": "自动补充生命值", "auto_mp_enabled": "自动补充魔法值",
    "add_hp_percent": "生命值补给阈值", "add_mp_percent": "魔法值补给阈值",
    "add_hp_cooldown": "生命值补给冷却", "add_mp_cooldown": "魔法值补给冷却",
    "fps_limit": "帧率上限", "return_home_if_no_potion": "无药水时回城",
    "return_home_watch_dog_timeout": "回城监控超时", "is_use_teleport_to_walk": "移动时使用瞬移",
    "trigger_box_width": "触发框宽度", "trigger_box_height": "触发框高度", "color_code": "颜色代码",
    "offset": "位置偏移", "name": "名称", "mode": "模式",
    "search_y_limit": "名字搜索下边界 Y",
    "diff_thres": "匹配阈值", "global_diff_thres": "全局匹配阈值", "split_width": "分块宽度",
    "openvino_deployment_root": "OpenVINO 部署目录",
    "yolo_variant": "YOLO 模型版本",
    "yolo_confidence": "YOLO 置信度",
    "yolo_min_monster_box_side": "怪物框最小边长",
    "yolo_player_exclusion_width": "人物排除区宽度",
    "yolo_player_exclusion_height": "人物排除区高度",
    "yolo_max_det": "YOLO 最大检测数",
    "pursuit_vertical_tolerance": "追击纵向容差",
    "target_lost_grace_seconds": "目标丢失追击时间",
    "other_player_move_thres": "其他玩家移动阈值",
    "interval_seconds": "间隔秒数", "ui_y_start": "界面起始 Y", "menu": "菜单坐标",
    "channel": "频道按钮坐标", "random_channel": "随机频道坐标",
    "random_channel_confirm": "随机频道确认坐标", "select_character": "选择人物坐标",
    "login_button_thres": "登录按钮阈值", "login_button_top_left": "登录按钮左上角",
    "login_button_bottom_right": "登录按钮右下角", "search_range": "路线搜索范围",
    "traversal_commit_seconds": "已开始路线动作保护时间",
    "mount_horizontal_tolerance": "挂梯粗对位容差",
    "mount_precision_tolerance": "挂梯精确对位容差",
    "mount_settle_frames": "挂梯站稳确认帧数",
    "mount_forward_hold_frames": "向前挂梯方向保持帧数",
    "mount_retry_runup_distance": "向前挂梯重试助跑距离",
    "mount_attempt_timeout": "单次挂梯超时",
    "mount_confirm_frames": "挂梯成功确认帧数",
    "mount_max_attempts": "挂梯最大尝试次数",
    "mount_retry_frames": "挂梯重试帧数",
    "mount_success_distance": "挂梯成功上升距离",
    "ladder_record_combo_seconds": "录制上跳组合窗口",
    "ladder_record_timeout_seconds": "录制挂梯确认超时",
    "ladder_record_success_distance": "录制挂梯上升距离",
    "ladder_record_confirm_frames": "录制挂梯确认帧数",
    "jump_down_cooldown": "下跳冷却", "color_code_up_down": "上下动作颜色代码",
    "range": "范围", "timeout": "超时", "last_attack_timeout": "距上次攻击超时",
    "last_attack_timeout_action": "攻击超时后动作",
    "player_color": "玩家颜色", "other_player_color": "其他玩家颜色",
    "debug_window_upscale": "调试窗口放大倍数", "turn_point_thres": "转向点阈值",
    "patrol_attack_interval": "巡逻攻击间隔", "width_px": "巡逻总宽度",
    "title": "窗口标题", "size": "窗口尺寸", "ratio_tolerance": "宽高比容差",
    "title_bar_height": "标题栏高度", "print_frequency": "输出频率",
}

ENUM_LABELS = {
    "normal": "普通", "aux": "辅助", "patrol": "巡逻",
    "fixed_platform": "固定平台区域攻击",
    "continuous_attack": "持续定向攻击",
    "directional": "方向攻击", "aoe_skill": "范围技能",
    "grayscale": "灰度", "white_mask": "白色掩膜", "histogram_eq": "直方图均衡",
    "auto": "自动（默认 INT8 Mixed + 检测头 FP）",
    "int8_v2": "INT8 Mixed + 检测头 FP（默认）",
    "fp16": "FP16（精度对照）", "int8": "INT8 v1（旧版对照）",
    "true": "检测到即换线", "pixel": "检测移动后换线", "go_home": "回城",
    "change_channel": "换线",
}


def section_label(key):
    return SECTION_LABELS.get(key, f"高级设置（{key}）")


def field_label(key):
    return FIELD_LABELS.get(key, f"配置项（{key}）")


def enum_label(value):
    return ENUM_LABELS.get(value, value)


def field_tooltip(section, key):
    """Keep a concise Chinese explanation while exposing the stable key."""
    if section == "health_monitor" and key == "force_heal":
        return (
            "生命值低于阈值时暂停攻击、跳跃和补蓝，保留左右移动并优先补血；"
            "仍遵守生命值补给冷却。内部配置键：health_monitor.force_heal"
        )
    return f"{field_label(key)}。内部配置键：{section}.{key}"
