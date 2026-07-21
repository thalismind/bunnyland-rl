from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest
from bunnyland.core import (
    ActionArgument,
    ActionDefinition,
    ActionPointsComponent,
    CharacterComponent,
    ContainmentMode,
    Contains,
    ExitTo,
    IdentityComponent,
    RoomComponent,
    WorldActor,
    spawn_entity,
)
from bunnyland.llm_agents import ControllerDispatch, ScriptedAgent, ToolCall, tool_names
from bunnyland.plugins import apply_plugins
from bunnyland.prompts.builder import PromptBuilder
from safetensors.torch import load_file

from bunnyland_rl.components import RLControllerComponent
from bunnyland_rl.lenses import encode_lenses
from bunnyland_rl.plugin import bunnyland_plugins as _plugins
from bunnyland_rl.policy import (
    DeepPolicyNet,
    MLPPolicyNet,
    ResidualPolicyNet,
    policy_net_names,
    register_policy_net,
    validate_policy_net,
)
from bunnyland_rl.training import MIN_ACTION_HEAD_SIZE, RLTrainingService, TrainingConfig
from bunnyland_rl.wandb_tracking import WANDB_ENABLED_ENV


def scenario() -> tuple[WorldActor, str]:
    actor = WorldActor()
    room_a = spawn_entity(actor.world, [RoomComponent(title="Mosslit Burrow")])
    room_b = spawn_entity(actor.world, [RoomComponent(title="North Tunnel")])
    room_a.add_relationship(ExitTo(direction="north"), room_b.id)
    character = spawn_entity(
        actor.world,
        [
            IdentityComponent(name="Juniper", kind="character"),
            CharacterComponent(species="bunny"),
            ActionPointsComponent(current=5.0, maximum=5.0),
        ],
    )
    room_a.add_relationship(Contains(mode=ContainmentMode.ROOM_CONTENT), character.id)
    return actor, str(character.id)


def test_plugin_imports_and_contributes_ecs_and_runtime():
    plugins = _plugins()

    assert [plugin.id for plugin in plugins] == ["bunnyland.rl"]
    assert RLControllerComponent in plugins[0].ecs.components
    assert plugins[0].runtime.controller_factories
    assert plugins[0].runtime.http
    assert plugins[0].runtime.http[0].zone.value == "admin"


def test_policy_registry_validates_and_accepts_new_policy():
    assert {"mlp", "deep", "residual"} <= set(policy_net_names())
    assert MLPPolicyNet.name == "mlp"
    assert DeepPolicyNet.name == "deep"
    assert ResidualPolicyNet.name == "residual"

    class TestPolicy:
        def __init__(self, input_size: int, output_size: int) -> None:
            self.shape = (input_size, output_size)

    register_policy_net("test", TestPolicy)
    assert "test" in policy_net_names()
    with pytest.raises(ValueError, match="unknown policy net"):
        validate_policy_net("missing")


def test_lenses_are_deterministic_and_have_stable_shapes():
    actor, character_id = scenario()
    builder = PromptBuilder(actor.world)
    character_entity = next(
        iter(actor.world.query().with_all([CharacterComponent]).execute_entities())
    )
    character = actor.world.get_entity(character_entity.id)
    context = builder.build(character.id)

    first = encode_lenses(
        ("room_text", "perception_text", "stats_vector", "components_vector", "room_grid"),
        actor.world,
        character,
        context,
    )
    second = encode_lenses(tuple(output.name for output in first), actor.world, character, context)

    assert first == second
    assert [output.shape for output in first] == [(64,), (64,), (4,), (6,), (9,)]
    assert character_id


def test_training_job_completes_saves_and_reloads_model(tmp_path):
    actor, character_id = scenario()
    service = RLTrainingService(actor, storage_dir=tmp_path)
    job = service.create_job(
        TrainingConfig(
            character_id=character_id,
            policy_net="mlp",
            lenses=("room_text", "stats_vector"),
            behavior_name="wanderer",
            episodes=1,
            updates_per_episode=2,
            action_size=8,
        )
    )

    asyncio.run(service.run_job(job.job_id))
    completed = service.get_job(job.job_id)

    assert completed is not None
    assert completed.status == "completed"
    assert completed.model_id is not None
    model = service.get_model(completed.model_id)
    assert model is not None
    assert model.config.mode == "behavior_overlay"
    assert model.config.behavior_name == "wanderer"
    assert model.config.action_size >= MIN_ACTION_HEAD_SIZE
    assert model.config.output_size == model.config.action_size + model.config.target_size
    assert model.metrics.reward_curve
    assert model.metrics.action_histogram
    assert model.artifact_path
    assert model.weights_path
    assert model.weights_format == "safetensors"
    weights = load_file(model.weights_path)
    assert weights
    assert model.weights_path.endswith(".safetensors")
    preview = service.preview_model_weights(model.model_id, max_rows=6, max_columns=7)
    assert preview["model_id"] == model.model_id
    assert preview["layers"]
    selected = preview["selected_layer"]
    assert selected["name"]
    assert selected["values"]
    assert len(selected["values"]) <= 6
    assert len(selected["values"][0]) <= 7
    assert selected["min"] <= selected["mean"] <= selected["max"]
    bias_layer = next(
        layer["name"] for layer in preview["layers"] if layer["name"].endswith("bias")
    )
    bias_preview = service.preview_model_weights(model.model_id, layer_name=bias_layer)
    assert bias_preview["selected_layer"]["rows"] == 1
    with pytest.raises(ValueError, match="layer 'missing' does not exist"):
        service.preview_model_weights(model.model_id, layer_name="missing")

    reloaded = RLTrainingService(actor, storage_dir=tmp_path)
    reloaded_model = reloaded.get_model(completed.model_id)
    assert reloaded_model is not None
    assert reloaded_model.weights_path == model.weights_path


def test_training_job_logs_to_wandb_when_enabled(monkeypatch, tmp_path):
    runs = []
    artifacts = []

    class FakeRun:
        id = "run-123"
        url = "https://wandb.test/run-123"

        def __init__(self) -> None:
            self.logs = []
            self.finished = False

        def log(self, payload, step=None) -> None:
            self.logs.append((payload, step))

        def log_artifact(self, artifact) -> None:
            artifacts.append(artifact)

        def finish(self, exit_code=0) -> None:
            self.finished = exit_code == 0

    class FakeArtifact:
        def __init__(self, name, type) -> None:
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path) -> None:
            self.files.append(path)

    def fake_init(**kwargs):
        run = FakeRun()
        run.kwargs = kwargs
        runs.append(run)
        return run

    monkeypatch.setenv(WANDB_ENABLED_ENV, "1")
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=fake_init, Artifact=FakeArtifact),
    )
    actor, character_id = scenario()
    service = RLTrainingService(actor, storage_dir=tmp_path)
    job = service.create_job(
        TrainingConfig(
            character_id=character_id,
            behavior_name="guard",
            episodes=1,
            updates_per_episode=1,
        )
    )

    asyncio.run(service.run_job(job.job_id))
    completed = service.get_job(job.job_id)
    model = service.get_model(completed.model_id)

    assert completed.wandb_run_id == "run-123"
    assert completed.wandb_url == "https://wandb.test/run-123"
    assert model.wandb_run_id == "run-123"
    assert runs[0].kwargs["project"] == "bunnyland-rl"
    assert any("train/reward" in payload for payload, _step in runs[0].logs)
    assert artifacts and artifacts[0].files
    assert any(path == model.weights_path for path in artifacts[0].files)
    assert runs[0].finished


def test_assignment_creates_rl_controller_and_bumps_generation(tmp_path):
    actor, character_id = scenario()
    service = RLTrainingService(actor, storage_dir=tmp_path)
    job = service.create_job(
        TrainingConfig(
            character_id=character_id,
            behavior_name="guard",
            episodes=1,
            updates_per_episode=1,
        )
    )
    asyncio.run(service.run_job(job.job_id))
    model_id = service.get_job(job.job_id).model_id

    first = service.assign_model(character_id=character_id, model_id=model_id)
    second = service.assign_model(character_id=character_id, model_id=model_id)

    assert first["generation"] == 0
    assert second["generation"] == 1
    controller_entity = next(
        iter(actor.world.query().with_all([RLControllerComponent]).execute_entities())
    )
    controller = actor.world.get_entity(controller_entity.id)
    assert controller.has_component(RLControllerComponent)
    assigned = controller.get_component(RLControllerComponent)
    assert assigned.mode == "behavior_overlay"
    assert assigned.behavior_name == "guard"


def test_rl_dispatch_emits_normal_tool_calls(monkeypatch):
    monkeypatch.setattr(
        "bunnyland_rl.agent.stable_score",
        lambda *parts: 1 if "move" in parts else 0,
    )
    actor, character_id = scenario()
    actor.register_action_definition(
        ActionDefinition(
            command_type="wait",
            tool_name="wait",
            arguments={},
        )
    )
    actor.register_action_definition(
        ActionDefinition(
            command_type="move",
            tool_name="move",
            arguments={"direction": ActionArgument(required=True)},
        )
    )
    apply_plugins(_plugins(), actor)
    character_entity = next(
        iter(actor.world.query().with_all([CharacterComponent]).execute_entities())
    )
    character = actor.world.get_entity(character_entity.id)
    controller = spawn_entity(actor.world, [RLControllerComponent(model_id="builtin:untrained")])
    actor.assign_controller(character.id, controller.id)
    dispatch = ControllerDispatch(
        actor,
        PromptBuilder(actor.world),
        ScriptedAgent([ToolCall("wait", {})]),
    )

    async def dispatch_once():
        decisions = await dispatch.run_once()
        decisions.extend(await dispatch.await_pending())
        return decisions

    decisions = asyncio.run(dispatch_once())

    assert decisions
    assert decisions[0].tool is None or decisions[0].tool in tool_names(actor.action_definitions())
    assert character_id


def test_admin_rl_routes_are_contributed_under_admin(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")

    from bunnyland.server.app import create_app

    monkeypatch.setenv("BUNNYLAND_RL_DIR", str(tmp_path))
    actor, character_id = scenario()
    plugins = _plugins()
    apply_plugins(plugins, actor)
    app = create_app(actor, plugins=plugins, allow_unauthenticated_embedding=True)

    paths = {getattr(route, "path", "") for route in app.routes}
    extension = "/v1/admin/extensions/bunnyland.rl/rl"
    assert f"{extension}/status" in paths
    assert f"{extension}/training/jobs" in paths
    assert f"{extension}/models/{{model_id}}/weights/preview" in paths
    assert all(path.startswith(extension) for path in paths if "/rl/" in path)
    assert character_id
