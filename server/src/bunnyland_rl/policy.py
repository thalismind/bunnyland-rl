"""Policy-net registry with optional lazy RL dependencies."""

from __future__ import annotations

from collections.abc import Callable

from .components import PolicyNetName

PolicyNetFactory = Callable[[int, int], object]
POLICY_NETS: dict[str, PolicyNetFactory] = {}


def _torch():
    import torch

    return torch


class MLPPolicyNet:
    name = "mlp"

    def __init__(self, input_size: int, output_size: int, *, hidden: int = 64) -> None:
        torch = _torch()
        self.module = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, output_size),
        )

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class DeepPolicyNet:
    name = "deep"

    def __init__(self, input_size: int, output_size: int, *, hidden: int = 128) -> None:
        torch = _torch()
        self.module = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, output_size),
        )

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ResidualPolicyNet:
    name = "residual"

    def __init__(self, input_size: int, output_size: int, *, hidden: int = 128) -> None:
        torch = _torch()

        class _Residual(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input = torch.nn.Linear(input_size, hidden)
                self.block = torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden, hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden, hidden),
                )
                self.output = torch.nn.Linear(hidden, output_size)

            def forward(self, value):
                hidden_value = self.input(value)
                return self.output(hidden_value + self.block(hidden_value))

        self.module = _Residual()

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def register_policy_net(name: str, factory: PolicyNetFactory) -> None:
    if not name:
        raise ValueError("policy net name is required")
    POLICY_NETS[name] = factory


def policy_net_names() -> tuple[str, ...]:
    return tuple(POLICY_NETS)


def validate_policy_net(name: str) -> PolicyNetName:
    if name not in POLICY_NETS:
        raise ValueError(f"unknown policy net {name!r}; expected one of {', '.join(POLICY_NETS)}")
    return name  # type: ignore[return-value]


def build_policy_net(name: str, input_size: int, output_size: int):
    validate_policy_net(name)
    return POLICY_NETS[name](input_size, output_size)


register_policy_net(MLPPolicyNet.name, MLPPolicyNet)
register_policy_net(DeepPolicyNet.name, DeepPolicyNet)
register_policy_net(ResidualPolicyNet.name, ResidualPolicyNet)


__all__ = [
    "DeepPolicyNet",
    "MLPPolicyNet",
    "POLICY_NETS",
    "PolicyNetFactory",
    "ResidualPolicyNet",
    "build_policy_net",
    "policy_net_names",
    "register_policy_net",
    "validate_policy_net",
]
