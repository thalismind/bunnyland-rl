"""Bunnyland plugin entrypoint for the out-of-tree RL extension."""

from __future__ import annotations

from bunnyland.plugins import EcsContribution, Plugin, RuntimeContribution

from .api import install_rl_routes
from .components import RLControllerComponent
from .runtime import install_rl_runtime

PLUGIN_ID = "bunnyland.rl"


def plugin() -> Plugin:
    return Plugin(
        id=PLUGIN_ID,
        name="Bunnyland RL",
        version="0.1.0",
        default_enabled=True,
        ecs=EcsContribution(components=(RLControllerComponent,)),
        runtime=RuntimeContribution(
            controller_factories=(install_rl_runtime,),
            server_routers=(install_rl_routes,),
        ),
    )


def bunnyland_plugins() -> list[Plugin]:
    return [plugin()]


__all__ = ["PLUGIN_ID", "bunnyland_plugins", "plugin"]
