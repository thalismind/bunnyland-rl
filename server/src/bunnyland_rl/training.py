"""Offline arena training job service with persistent model artifacts."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from uuid import uuid4

from bunnyland.core import CharacterComponent, parse_entity_id, spawn_entity
from bunnyland.core.world_actor import WorldActor
from relics import EntityId

from .components import RLControllerComponent
from .encode import stable_score
from .lenses import LENSES
from .policy import validate_policy_net

RL_DIR_ENV = "BUNNYLAND_RL_DIR"
DEFAULT_RL_DIR = "data/rl"
MIN_ACTION_HEAD_SIZE = 512


@dataclass(frozen=True)
class TrainingConfig:
    character_id: str
    policy_net: str = "mlp"
    lenses: tuple[str, ...] = ("room_text", "perception_text", "stats_vector")
    episodes: int = 8
    updates_per_episode: int = 4
    seed: str = ""
    action_size: int = MIN_ACTION_HEAD_SIZE
    target_size: int = MIN_ACTION_HEAD_SIZE
    output_size: int = MIN_ACTION_HEAD_SIZE * 2


@dataclass(frozen=True)
class TrainingMetrics:
    episode: int = 0
    update: int = 0
    reward_curve: tuple[float, ...] = ()
    loss_curve: tuple[float, ...] = ()
    action_histogram: dict[str, int] = field(default_factory=dict)
    trust_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    created_at_unix: float
    config: TrainingConfig
    metrics: TrainingMetrics
    checkpoint_path: str
    artifact_path: str
    wandb_run_id: str | None = None
    wandb_url: str | None = None


@dataclass(frozen=True)
class TrainingJob:
    job_id: str
    status: str
    created_at_unix: float
    updated_at_unix: float
    config: TrainingConfig
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    latest_checkpoint: str | None = None
    model_id: str | None = None
    error: str = ""
    cancel_requested: bool = False
    wandb_run_id: str | None = None
    wandb_url: str | None = None


class RLTrainingService:
    def __init__(self, actor: WorldActor, *, storage_dir: str | Path | None = None) -> None:
        self.actor = actor
        self.storage_dir = Path(storage_dir or os.environ.get(RL_DIR_ENV, DEFAULT_RL_DIR))
        self.models_dir = self.storage_dir / "models"
        self.checkpoints_dir = self.storage_dir / "checkpoints"
        self.jobs: dict[str, TrainingJob] = {}
        self.models: dict[str, ModelArtifact] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._load_models()

    def create_job(self, config: TrainingConfig) -> TrainingJob:
        config = self._validated_config(config)
        job_id = uuid4().hex
        now = time.time()
        job = TrainingJob(
            job_id=job_id,
            status="queued",
            created_at_unix=now,
            updated_at_unix=now,
            config=config,
        )
        self.jobs[job_id] = job
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            self._tasks[job_id] = loop.create_task(self.run_job(job_id))
        return job

    def list_jobs(self) -> list[TrainingJob]:
        return sorted(self.jobs.values(), key=lambda job: job.created_at_unix, reverse=True)

    def get_job(self, job_id: str) -> TrainingJob | None:
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> TrainingJob | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.status not in {"queued", "running"}:
            return job
        updated = replace(job, cancel_requested=True, updated_at_unix=time.time())
        self.jobs[job_id] = updated
        return updated

    def list_models(self) -> list[ModelArtifact]:
        return sorted(self.models.values(), key=lambda model: model.created_at_unix, reverse=True)

    def get_model(self, model_id: str) -> ModelArtifact | None:
        if model_id == "builtin:untrained":
            return ModelArtifact(
                model_id=model_id,
                created_at_unix=0.0,
                config=TrainingConfig(character_id=""),
                metrics=TrainingMetrics(),
                checkpoint_path="",
                artifact_path="",
            )
        return self.models.get(model_id)

    def assign_model(
        self,
        *,
        character_id: str,
        model_id: str,
        policy_net: str | None = None,
        lenses: tuple[str, ...] | None = None,
        mode: str = "standalone",
        behavior_name: str | None = None,
        act_every_ticks: int = 1,
    ) -> dict[str, object]:
        character_entity_id = _character_id(self.actor, character_id)
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"model {model_id!r} does not exist")
        selected_policy = validate_policy_net(policy_net or model.config.policy_net)
        selected_lenses = tuple(lenses or model.config.lenses)
        _validate_lenses(selected_lenses)
        if mode not in {"standalone", "behavior_overlay"}:
            raise ValueError("mode must be standalone or behavior_overlay")
        if mode == "behavior_overlay" and not behavior_name:
            raise ValueError("behavior_name is required for behavior_overlay mode")

        controller = spawn_entity(
            self.actor.world,
            [
                RLControllerComponent(
                    model_id=model_id,
                    policy_net=selected_policy,
                    lenses=selected_lenses,
                    mode=mode,  # type: ignore[arg-type]
                    behavior_name=behavior_name,
                    act_every_ticks=max(1, int(act_every_ticks)),
                )
            ],
        )
        generation = self.actor.assign_controller(character_entity_id, controller.id)
        return {
            "character_id": str(character_entity_id),
            "controller_id": str(controller.id),
            "generation": generation,
            "model_id": model_id,
        }

    async def run_job(self, job_id: str) -> None:
        job = self.jobs[job_id]
        self.jobs[job_id] = replace(job, status="running", updated_at_unix=time.time())
        try:
            await self._simulate_training(job_id)
        except Exception as exc:  # noqa: BLE001 - job errors are reported in status
            current = self.jobs[job_id]
            self.jobs[job_id] = replace(
                current,
                status="failed",
                error=str(exc),
                updated_at_unix=time.time(),
            )

    async def _simulate_training(self, job_id: str) -> None:
        job = self.jobs[job_id]
        reward_curve: list[float] = []
        loss_curve: list[float] = []
        histogram: dict[str, int] = {}
        trust = _initial_trust(job.config.lenses)
        total_updates = max(1, job.config.episodes * job.config.updates_per_episode)
        for episode in range(1, job.config.episodes + 1):
            for update in range(1, job.config.updates_per_episode + 1):
                current = self.jobs[job_id]
                if current.cancel_requested:
                    self.jobs[job_id] = replace(
                        current,
                        status="cancelled",
                        updated_at_unix=time.time(),
                    )
                    return
                progress = ((episode - 1) * job.config.updates_per_episode + update) / total_updates
                reward_curve.append(round(progress * 10.0, 4))
                loss_curve.append(round(max(0.01, 1.0 - progress), 4))
                action = _sample_action(job.config.seed, episode, update)
                histogram[action] = histogram.get(action, 0) + 1
                trust = _updated_trust(job.config.lenses, progress)
                checkpoint = self._write_checkpoint(job_id, episode, update)
                self.jobs[job_id] = replace(
                    current,
                    status="running",
                    updated_at_unix=time.time(),
                    latest_checkpoint=str(checkpoint),
                    metrics=TrainingMetrics(
                        episode=episode,
                        update=update,
                        reward_curve=tuple(reward_curve),
                        loss_curve=tuple(loss_curve),
                        action_histogram=dict(histogram),
                        trust_weights=trust,
                    ),
                )
                await asyncio.sleep(0)
        completed = self.jobs[job_id]
        artifact = self._save_model(completed)
        self.models[artifact.model_id] = artifact
        self.jobs[job_id] = replace(
            completed,
            status="completed",
            updated_at_unix=time.time(),
            model_id=artifact.model_id,
            latest_checkpoint=artifact.checkpoint_path,
            wandb_run_id=artifact.wandb_run_id,
            wandb_url=artifact.wandb_url,
        )

    def _validated_config(self, config: TrainingConfig) -> TrainingConfig:
        _character_id(self.actor, config.character_id)
        validate_policy_net(config.policy_net)
        _validate_lenses(config.lenses)
        live_actions = len(self.actor.action_definitions())
        action_size = max(MIN_ACTION_HEAD_SIZE, live_actions, int(config.action_size))
        target_size = max(MIN_ACTION_HEAD_SIZE, int(config.target_size))
        return replace(
            config,
            episodes=max(1, int(config.episodes)),
            updates_per_episode=max(1, int(config.updates_per_episode)),
            action_size=action_size,
            target_size=target_size,
            output_size=action_size + target_size,
        )

    def _write_checkpoint(self, job_id: str, episode: int, update: int) -> Path:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoints_dir / f"{job_id}-e{episode:04d}-u{update:04d}.json"
        payload = {"job_id": job_id, "episode": episode, "update": update}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _save_model(self, job: TrainingJob) -> ModelArtifact:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_id = f"rl-{job.job_id}"
        checkpoint_path = job.latest_checkpoint or ""
        artifact_path = self.models_dir / f"{model_id}.json"
        artifact = ModelArtifact(
            model_id=model_id,
            created_at_unix=time.time(),
            config=job.config,
            metrics=job.metrics,
            checkpoint_path=checkpoint_path,
            artifact_path=str(artifact_path),
            wandb_run_id=None,
            wandb_url=None,
        )
        artifact_path.write_text(
            json.dumps(_artifact_dict(artifact), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return artifact

    def _load_models(self) -> None:
        if not self.models_dir.exists():
            return
        for path in sorted(self.models_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                artifact = _artifact_from_dict(data, artifact_path=str(path))
            except (OSError, ValueError, TypeError, KeyError):
                continue
            self.models[artifact.model_id] = artifact


def _character_id(actor: WorldActor, raw: str) -> EntityId:
    parsed = parse_entity_id(raw)
    if parsed is None or not actor.world.has_entity(parsed):
        raise ValueError("character does not exist")
    entity = actor.world.get_entity(parsed)
    if not entity.has_component(CharacterComponent):
        raise ValueError("entity is not a character")
    return parsed


def _validate_lenses(names: tuple[str, ...]) -> None:
    if not names:
        raise ValueError("at least one lens is required")
    unknown = [name for name in names if name not in LENSES]
    if unknown:
        raise ValueError(f"unknown lens(es): {', '.join(unknown)}")


def _initial_trust(lenses: tuple[str, ...]) -> dict[str, float]:
    weight = round(1.0 / max(1, len(lenses)), 4)
    return {lens: weight for lens in lenses}


def _updated_trust(lenses: tuple[str, ...], progress: float) -> dict[str, float]:
    raw = {lens: 1.0 + ((index + 1) * progress) for index, lens in enumerate(lenses)}
    total = sum(raw.values()) or 1.0
    return {lens: round(value / total, 4) for lens, value in raw.items()}


def _sample_action(seed: str, episode: int, update: int) -> str:
    actions = ("move", "take", "say", "wait")
    index = int(stable_score(seed, episode, update) * len(actions)) % len(actions)
    return actions[index]


def _artifact_dict(artifact: ModelArtifact) -> dict[str, object]:
    return {
        "model_id": artifact.model_id,
        "created_at_unix": artifact.created_at_unix,
        "config": asdict(artifact.config),
        "metrics": asdict(artifact.metrics),
        "checkpoint_path": artifact.checkpoint_path,
        "artifact_path": artifact.artifact_path,
        "wandb_run_id": artifact.wandb_run_id,
        "wandb_url": artifact.wandb_url,
    }


def _artifact_from_dict(data: dict[str, object], *, artifact_path: str) -> ModelArtifact:
    config_data = dict(data["config"])  # type: ignore[arg-type]
    metrics_data = dict(data["metrics"])  # type: ignore[arg-type]
    config = TrainingConfig(
        character_id=str(config_data.get("character_id", "")),
        policy_net=str(config_data.get("policy_net", "mlp")),
        lenses=tuple(config_data.get("lenses", ())),
        episodes=int(config_data.get("episodes", 1)),
        updates_per_episode=int(config_data.get("updates_per_episode", 1)),
        seed=str(config_data.get("seed", "")),
        action_size=int(config_data.get("action_size", MIN_ACTION_HEAD_SIZE)),
        target_size=int(config_data.get("target_size", MIN_ACTION_HEAD_SIZE)),
        output_size=int(config_data.get("output_size", MIN_ACTION_HEAD_SIZE * 2)),
    )
    metrics = TrainingMetrics(
        episode=int(metrics_data.get("episode", 0)),
        update=int(metrics_data.get("update", 0)),
        reward_curve=tuple(float(value) for value in metrics_data.get("reward_curve", ())),
        loss_curve=tuple(float(value) for value in metrics_data.get("loss_curve", ())),
        action_histogram=dict(metrics_data.get("action_histogram", {})),
        trust_weights=dict(metrics_data.get("trust_weights", {})),
    )
    return ModelArtifact(
        model_id=str(data["model_id"]),
        created_at_unix=float(data.get("created_at_unix", 0.0)),
        config=config,
        metrics=metrics,
        checkpoint_path=str(data.get("checkpoint_path", "")),
        artifact_path=artifact_path,
        wandb_run_id=data.get("wandb_run_id") or None,  # type: ignore[arg-type]
        wandb_url=data.get("wandb_url") or None,  # type: ignore[arg-type]
    )


__all__ = [
    "DEFAULT_RL_DIR",
    "MIN_ACTION_HEAD_SIZE",
    "ModelArtifact",
    "RLTrainingService",
    "TrainingConfig",
    "TrainingJob",
    "TrainingMetrics",
]
