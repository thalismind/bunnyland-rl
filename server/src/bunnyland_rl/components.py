"""ECS components contributed by the Bunnyland RL plugin."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic.dataclasses import dataclass
from relics import Component

PolicyNetName = Literal["mlp", "deep", "residual"]
ControllerMode = Literal["standalone", "behavior_overlay"]


@dataclass(frozen=True)
class RLControllerComponent(Component):
    """Autonomous controller backed by an RL policy artifact."""

    model_id: str = "builtin:untrained"
    policy_net: PolicyNetName = "mlp"
    lenses: tuple[str, ...] = ("room_text", "perception_text", "stats_vector")
    mode: ControllerMode = "standalone"
    behavior_name: str | None = None
    act_every_ticks: int = Field(default=1, ge=1)


__all__ = ["ControllerMode", "PolicyNetName", "RLControllerComponent"]
