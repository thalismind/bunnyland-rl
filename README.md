# Bunnyland RL

Out-of-tree reinforcement-learning controller plugin for Bunnyland.

- `server/` exposes `bunnyland_rl.bunnyland_plugins()`.
- `web/` builds the admin dashboard served under `/rl/`.
- `Dockerfile.server` installs the plugin into the Bunnyland server image.
- `Dockerfile.web` copies the dashboard into `/usr/share/nginx/html/rl`.

Training v1 runs offline arena jobs from the live world state. Jobs do not submit live
commands. Completed jobs save reloadable JSON model artifacts under `BUNNYLAND_RL_DIR`
(default `data/rl`). W&B fields are present in job/model responses for v2 tracking.

Run both 3D and RL plugins:

```bash
bunnyland serve --module bunnyland_3d --module bunnyland_rl ...
```
