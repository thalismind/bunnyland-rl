"""Optional Weights & Biases tracking for RL training jobs."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

WANDB_ENABLED_ENV = "BUNNYLAND_RL_WANDB"
WANDB_PROJECT_ENV = "BUNNYLAND_RL_WANDB_PROJECT"
WANDB_ENTITY_ENV = "BUNNYLAND_RL_WANDB_ENTITY"
WANDB_GROUP_ENV = "BUNNYLAND_RL_WANDB_GROUP"
DEFAULT_WANDB_PROJECT = "bunnyland-rl"


@dataclass(frozen=True)
class WandbRunInfo:
    run_id: str | None = None
    url: str | None = None


class WandbTracker:
    def __init__(self, *, enabled: bool | None = None) -> None:
        self.enabled = wandb_enabled() if enabled is None else enabled
        self._runs: dict[str, Any] = {}

    def start_job(self, job) -> WandbRunInfo:
        if not self.enabled:
            return WandbRunInfo()
        wandb = _wandb()
        run = wandb.init(
            project=os.environ.get(WANDB_PROJECT_ENV, DEFAULT_WANDB_PROJECT),
            entity=os.environ.get(WANDB_ENTITY_ENV) or None,
            group=os.environ.get(WANDB_GROUP_ENV) or None,
            name=f"rl-{job.job_id[:8]}",
            job_type="offline-training",
            config={"job_id": job.job_id, **asdict(job.config)},
            reinit=True,
        )
        self._runs[job.job_id] = run
        return WandbRunInfo(run_id=getattr(run, "id", None), url=getattr(run, "url", None))

    def log_metrics(self, job, *, checkpoint_path: str | None = None) -> None:
        run = self._runs.get(job.job_id)
        if run is None:
            return
        metrics = job.metrics
        payload: dict[str, object] = {
            "job/status": job.status,
            "train/episode": metrics.episode,
            "train/update": metrics.update,
        }
        if metrics.reward_curve:
            payload["train/reward"] = metrics.reward_curve[-1]
        if metrics.loss_curve:
            payload["train/loss"] = metrics.loss_curve[-1]
        if checkpoint_path:
            payload["checkpoint/path"] = checkpoint_path
        payload.update(
            {f"action_histogram/{name}": count for name, count in metrics.action_histogram.items()}
        )
        payload.update(
            {f"lens_trust/{name}": weight for name, weight in metrics.trust_weights.items()}
        )
        run.log(payload, step=max(0, len(metrics.reward_curve)))

    def finish_model(self, job, artifact) -> None:
        run = self._runs.get(job.job_id)
        if run is None:
            return
        run.log(
            {
                "job/status": "completed",
                "model/id": artifact.model_id,
                "model/artifact_path": artifact.artifact_path,
                "model/checkpoint_path": artifact.checkpoint_path,
            },
            step=max(0, len(artifact.metrics.reward_curve)),
        )
        self._log_model_artifact(run, artifact)
        run.finish()
        self._runs.pop(job.job_id, None)

    def finish_job(self, job, *, status: str) -> None:
        run = self._runs.get(job.job_id)
        if run is None:
            return
        run.log({"job/status": status, "job/error": job.error})
        run.finish(exit_code=1 if status == "failed" else 0)
        self._runs.pop(job.job_id, None)

    @staticmethod
    def _log_model_artifact(run, artifact) -> None:
        try:
            wandb = _wandb()
            model_artifact = wandb.Artifact(artifact.model_id, type="bunnyland-rl-model")
            artifact_path = Path(artifact.artifact_path)
            checkpoint_path = Path(artifact.checkpoint_path) if artifact.checkpoint_path else None
            if artifact_path.exists():
                model_artifact.add_file(str(artifact_path))
            if checkpoint_path is not None and checkpoint_path.exists():
                model_artifact.add_file(str(checkpoint_path))
            run.log_artifact(model_artifact)
        except AttributeError:
            return


def wandb_enabled() -> bool:
    explicit = os.environ.get(WANDB_ENABLED_ENV, "").strip().lower()
    if explicit in {"0", "false", "no", "off", "disabled"}:
        return False
    if explicit in {"1", "true", "yes", "on", "enabled"}:
        return True
    return bool(
        os.environ.get("WANDB_API_KEY")
        or os.environ.get("WANDB_MODE") in {"offline", "dryrun", "run"}
    )


def _wandb():
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking is enabled but wandb is not installed; install bunnyland-rl[tracking]"
        ) from exc
    return wandb


__all__ = [
    "DEFAULT_WANDB_PROJECT",
    "WANDB_ENABLED_ENV",
    "WANDB_ENTITY_ENV",
    "WANDB_GROUP_ENV",
    "WANDB_PROJECT_ENV",
    "WandbRunInfo",
    "WandbTracker",
    "wandb_enabled",
]
