"""RL controller agent that emits normal Bunnyland tool calls."""

from __future__ import annotations

from typing import Any

from bunnyland.llm_agents import BehaviorTreeAgent, ToolCall, resolve_behavior_tree
from bunnyland.prompts.builder import PromptContext

from .components import RLControllerComponent
from .encode import stable_score


class RLAgent:
    def __init__(self, component: RLControllerComponent) -> None:
        self.component = component
        self._fallback: BehaviorTreeAgent | None = None

    def decide(
        self,
        prompt: str,
        context: PromptContext,
        *,
        character_id: str,
        model: str | None = None,
        provider: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ToolCall | None:
        del provider
        call = self._policy_decision(
            prompt,
            context,
            character_id=character_id,
            model=model,
            tools=tools or [],
        )
        if call is not None:
            return call
        if self.component.mode == "behavior_overlay" and self.component.behavior_name:
            if self._fallback is None:
                tree = resolve_behavior_tree(self.component.behavior_name)
                self._fallback = BehaviorTreeAgent(tree)
            return self._fallback.decide(prompt, context, character_id=character_id)
        return None

    def _policy_decision(
        self,
        prompt: str,
        context: PromptContext,
        *,
        character_id: str,
        model: str | None,
        tools: list[dict[str, Any]],
    ) -> ToolCall | None:
        candidates = _candidate_calls(context, character_id=character_id, tools=tools or [])
        candidates.append(ToolCall("wait", {}))
        if not candidates:
            return None
        ranked = sorted(
            candidates,
            key=lambda call: stable_score(
                self.component.model_id,
                self.component.policy_net,
                self.component.lenses,
                character_id,
                model or "",
                prompt[:512],
                call.name,
                sorted(call.arguments.items()),
            ),
            reverse=True,
        )
        best = ranked[0]
        if best.name == "wait":
            return None
        return best


def rl_agent_factory(dispatch, character_id: str, component: object):
    del dispatch, character_id
    assert isinstance(component, RLControllerComponent)
    return RLAgent(component), component.model_id, "rl"


def _candidate_calls(
    context: PromptContext,
    *,
    character_id: str,
    tools: list[dict[str, Any]],
) -> list[ToolCall]:
    candidates: list[ToolCall] = []
    for schema in tools:
        function = schema.get("function", {})
        name = str(function.get("name", ""))
        parameters = function.get("parameters", {})
        if not name or not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties", {})
        required = tuple(str(key) for key in parameters.get("required", ()))
        if not isinstance(properties, dict):
            properties = {}
        if name == "wait":
            continue
        if not required:
            candidates.append(ToolCall(name, _optional_arguments(properties, character_id)))
            continue
        if required == ("direction",) and context.exits:
            for exit_label in context.exits:
                candidates.append(ToolCall(name, {"direction": exit_label.split(" ", 1)[0]}))
            continue
        target_key = _single_target_key(required)
        if target_key is not None:
            target = _visible_target_for(target_key, context)
            if target:
                candidates.append(ToolCall(name, {target_key: target}))
    return candidates


def _optional_arguments(properties: dict[str, object], character_id: str) -> dict[str, object]:
    for key in ("target_id", "character_id", "entity_id"):
        if key in properties:
            return {key: character_id}
    return {}


def _single_target_key(required: tuple[str, ...]) -> str | None:
    if len(required) != 1:
        return None
    key = required[0]
    if key.endswith("_id") or key in {"target", "item"}:
        return key
    return None


def _visible_target_for(key: str, context: PromptContext) -> str:
    if key == "item_id" and context.visible_objects:
        return context.visible_objects[0]
    if key in {"target_id", "character_id"} and context.visible_characters:
        return context.visible_characters[0]
    if context.visible_objects:
        return context.visible_objects[0]
    if context.visible_characters:
        return context.visible_characters[0]
    return ""


__all__ = ["RLAgent", "rl_agent_factory"]
