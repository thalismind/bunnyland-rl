# Bunnyland RL

Out-of-tree reinforcement-learning controller plugin for Bunnyland.

- `server/` exposes `bunnyland_rl.bunnyland_plugins()`.
- `web/` builds the admin dashboard served under `/rl/`.
- `Dockerfile.server` installs the plugin into the Bunnyland server image.
- `Dockerfile.web` copies the dashboard into `/usr/share/nginx/html/rl`.

Training v1 runs offline arena jobs from the live world state. Jobs do not submit live
commands. Completed jobs save reloadable JSON model artifacts under `BUNNYLAND_RL_DIR`
(default `data/rl`).

Set `BUNNYLAND_RL_WANDB=1` (or provide normal `WANDB_*` settings) to track jobs in
Weights & Biases. The plugin logs per-update reward/loss/action/trust/checkpoint stats
and records the saved model JSON/checkpoint as a W&B artifact.

Run both 3D and RL plugins:

```bash
bunnyland serve --module bunnyland_3d --module bunnyland_rl ...
```
