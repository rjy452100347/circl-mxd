"""Compile-time announcement content for the main application shell."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnnouncementContent:
    status: str
    free_notice: str
    qq_group: str
    group_purpose: str
    risk_notice: str


ANNOUNCEMENT = AnnouncementContent(
    status="内测期间免费",
    free_notice=(
        "本辅助软件目前处于内部测试阶段，内测期间免费。"
        "请勿相信任何以软件本体收费、代购或转售为名的信息。"
    ),
    qq_group="860498805",
    group_purpose="用于问题反馈、地图路线交流和内测通知。",
    risk_notice=(
        "本软件不是游戏官方产品。自动化操作可能违反游戏规则，并可能导致"
        "账号限制或封禁；请仅在获得明确许可的测试环境中使用。软件不提供"
        "反作弊绕过能力，使用风险由使用者承担。"
    ),
)
