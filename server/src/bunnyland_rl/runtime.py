"""Runtime wiring for the RL plugin."""

from __future__ import annotations

from bunnyland.llm_agents import register_autonomous_controller

from .agent import rl_agent_factory
from .api import training_service
from .components import RLControllerComponent


def install_rl_runtime(actor) -> None:
    register_autonomous_controller(RLControllerComponent, rl_agent_factory)
    training_service(actor)


__all__ = ["install_rl_runtime"]
