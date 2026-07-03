"""Bunnyland-specific deterministic observation lenses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bunnyland.core import (
    ActionPointsComponent,
    CharacterComponent,
    FocusPointsComponent,
    IdentityComponent,
    PortableComponent,
    RoomComponent,
    container_of,
    contents,
)
from bunnyland.core.edges import ExitTo
from bunnyland.prompts.builder import PromptContext, render_prompt
from relics import Entity, World

from .encode import hashed_text_vector


@dataclass(frozen=True)
class LensOutput:
    name: str
    values: tuple[float, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.values),)


class Lens(Protocol):
    name: str
    size: int

    def encode(self, world: World, character: Entity, context: PromptContext) -> LensOutput: ...


class RoomTextLens:
    name = "room_text"
    size = 64

    def encode(self, world: World, character: Entity, context: PromptContext) -> LensOutput:
        del world, character
        text = "\n".join((context.location_title, context.room_summary))
        return LensOutput(self.name, hashed_text_vector(text, dims=self.size))


class PerceptionTextLens:
    name = "perception_text"
    size = 64

    def encode(self, world: World, character: Entity, context: PromptContext) -> LensOutput:
        del world, character
        text = "\n".join(
            (
                "exits: " + ", ".join(context.exits),
                "characters: " + ", ".join(context.visible_characters),
                "objects: " + ", ".join(context.visible_objects),
            )
        )
        return LensOutput(self.name, hashed_text_vector(text, dims=self.size))


class StatsVectorLens:
    name = "stats_vector"
    size = 4

    def encode(self, world: World, character: Entity, context: PromptContext) -> LensOutput:
        del world, character
        action_current, action_max = context.action
        focus_current, focus_max = context.focus
        return LensOutput(
            self.name,
            (
                _ratio(action_current, action_max),
                float(action_max),
                _ratio(focus_current, focus_max),
                float(focus_max),
            ),
        )


class ComponentsVectorLens:
    name = "components_vector"
    size = 6

    def encode(self, world: World, character: Entity, context: PromptContext) -> LensOutput:
        del context
        room_id = container_of(character)
        room_contents = (
            contents(world.get_entity(room_id))
            if room_id is not None and world.has_entity(room_id)
            else []
        )
        return LensOutput(
            self.name,
            (
                1.0 if character.has_component(ActionPointsComponent) else 0.0,
                1.0 if character.has_component(FocusPointsComponent) else 0.0,
                float(len(contents(character))),
                float(
                    sum(
                        1
                        for entity_id in room_contents
                        if world.has_entity(entity_id)
                        and world.get_entity(entity_id).has_component(CharacterComponent)
                    )
                ),
                float(
                    sum(
                        1
                        for entity_id in room_contents
                        if world.has_entity(entity_id)
                        and world.get_entity(entity_id).has_component(PortableComponent)
                    )
                ),
                1.0 if character.has_component(IdentityComponent) else 0.0,
            ),
        )


class RoomGridLens:
    name = "room_grid"
    size = 9

    def encode(self, world: World, character: Entity, context: PromptContext) -> LensOutput:
        del context
        grid = [0.0] * self.size
        room_id = container_of(character)
        if room_id is None or not world.has_entity(room_id):
            return LensOutput(self.name, tuple(grid))
        room = world.get_entity(room_id)
        if not room.has_component(RoomComponent):
            return LensOutput(self.name, tuple(grid))
        grid[4] = 1.0
        exit_count = len(room.get_relationships(ExitTo))
        occupant_count = len(contents(room))
        grid[1] = min(1.0, exit_count / 4.0)
        grid[7] = min(1.0, max(0, occupant_count - 1) / 8.0)
        return LensOutput(self.name, tuple(grid))


LENSES: dict[str, Lens] = {
    lens.name: lens
    for lens in (
        RoomTextLens(),
        PerceptionTextLens(),
        StatsVectorLens(),
        ComponentsVectorLens(),
        RoomGridLens(),
    )
}


def encode_lenses(
    names: tuple[str, ...],
    world: World,
    character: Entity,
    context: PromptContext,
) -> tuple[LensOutput, ...]:
    outputs: list[LensOutput] = []
    for name in names:
        lens = LENSES.get(name)
        if lens is None:
            raise ValueError(f"unknown RL lens {name!r}")
        outputs.append(lens.encode(world, character, context))
    return tuple(outputs)


def render_context_text(context: PromptContext) -> str:
    return render_prompt(context)


def _ratio(current: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return round(max(0.0, min(1.0, current / maximum)), 6)


__all__ = ["LENSES", "LensOutput", "encode_lenses", "render_context_text"]
