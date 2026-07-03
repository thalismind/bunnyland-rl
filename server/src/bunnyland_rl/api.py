"""FastAPI router contributed by the RL plugin."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from bunnyland.core.world_actor import WorldActor
from pydantic import BaseModel, Field

from .components import RLControllerComponent
from .lenses import LENSES
from .policy import policy_net_names
from .training import RLTrainingService, TrainingConfig
from .wandb_tracking import wandb_enabled

SERVICE_ATTR = "_bunnyland_rl_training_service"


class TrainingJobRequest(BaseModel):
    character_id: str
    policy_net: str = "mlp"
    lenses: list[str] = Field(
        default_factory=lambda: ["room_text", "perception_text", "stats_vector"]
    )
    mode: str = "behavior_overlay"
    behavior_name: str = "idle"
    episodes: int = 8
    updates_per_episode: int = 4
    seed: str = ""
    action_size: int = 512
    target_size: int = 512


class ModelAssignRequest(BaseModel):
    character_id: str
    policy_net: str | None = None
    lenses: list[str] | None = None
    mode: str | None = None
    behavior_name: str | None = None
    act_every_ticks: int = 1


def install_rl_routes(app, actor: WorldActor, **_context) -> None:
    try:
        from fastapi import HTTPException
    except ImportError as exc:
        raise RuntimeError("RL admin API requires FastAPI") from exc

    service = training_service(actor)

    @app.get("/admin/rl/status")
    async def rl_status() -> dict[str, Any]:
        return {
            "ok": True,
            "schema_version": 1,
            "world_epoch": actor.epoch,
            "policy_nets": list(policy_net_names()),
            "lenses": list(LENSES),
            "models": [_model_view(model) for model in service.list_models()],
            "controller_component": RLControllerComponent.__name__,
            "wandb_enabled": wandb_enabled(),
        }

    @app.post("/admin/rl/training/jobs")
    async def create_training_job(request: TrainingJobRequest) -> dict[str, Any]:
        try:
            job = service.create_job(
                TrainingConfig(
                    character_id=request.character_id,
                    policy_net=request.policy_net,
                    lenses=tuple(request.lenses),
                    mode=request.mode,
                    behavior_name=request.behavior_name,
                    episodes=request.episodes,
                    updates_per_episode=request.updates_per_episode,
                    seed=request.seed,
                    action_size=request.action_size,
                    target_size=request.target_size,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _job_view(job)

    @app.get("/admin/rl/training/jobs")
    async def list_training_jobs() -> dict[str, Any]:
        return {"jobs": [_job_view(job) for job in service.list_jobs()]}

    @app.get("/admin/rl/training/jobs/{job_id}")
    async def get_training_job(job_id: str) -> dict[str, Any]:
        job = service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="training job does not exist")
        return _job_view(job)

    @app.post("/admin/rl/training/jobs/{job_id}/cancel")
    async def cancel_training_job(job_id: str) -> dict[str, Any]:
        job = service.cancel_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="training job does not exist")
        return _job_view(job)

    @app.get("/admin/rl/models")
    async def list_models() -> dict[str, Any]:
        return {"models": [_model_view(model) for model in service.list_models()]}

    @app.get("/admin/rl/models/{model_id}")
    async def get_model(model_id: str) -> dict[str, Any]:
        model = service.get_model(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="model does not exist")
        return _model_view(model)

    @app.post("/admin/rl/models/{model_id}/assign")
    async def assign_model(model_id: str, request: ModelAssignRequest) -> dict[str, Any]:
        try:
            result = service.assign_model(
                character_id=request.character_id,
                model_id=model_id,
                policy_net=request.policy_net,
                lenses=tuple(request.lenses) if request.lenses is not None else None,
                mode=request.mode,
                behavior_name=request.behavior_name,
                act_every_ticks=request.act_every_ticks,
            )
        except ValueError as exc:
            detail = str(exc)
            status = 404 if "does not exist" in detail else 400
            raise HTTPException(status_code=status, detail=detail) from exc
        return {"ok": True, **result}


def training_service(actor: WorldActor) -> RLTrainingService:
    service = getattr(actor, SERVICE_ATTR, None)
    if service is None:
        service = RLTrainingService(actor)
        setattr(actor, SERVICE_ATTR, service)
    return service


def _job_view(job) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "created_at_unix": job.created_at_unix,
        "updated_at_unix": job.updated_at_unix,
        "config": asdict(job.config),
        "metrics": asdict(job.metrics),
        "latest_checkpoint": job.latest_checkpoint,
        "model_id": job.model_id,
        "error": job.error,
        "cancel_requested": job.cancel_requested,
        "wandb_run_id": job.wandb_run_id,
        "wandb_url": job.wandb_url,
    }


def _model_view(model) -> dict[str, Any]:
    return {
        "model_id": model.model_id,
        "created_at_unix": model.created_at_unix,
        "config": asdict(model.config),
        "metrics": asdict(model.metrics),
        "checkpoint_path": model.checkpoint_path,
        "weights_path": model.weights_path,
        "weights_format": model.weights_format,
        "artifact_path": model.artifact_path,
        "wandb_run_id": model.wandb_run_id,
        "wandb_url": model.wandb_url,
    }


__all__ = ["SERVICE_ATTR", "install_rl_routes", "training_service"]
