# Bunnyland RL Server Plugin

The plugin entrypoint is `bunnyland_rl.bunnyland_plugins()`. It contributes:

- `RLControllerComponent`, an ECS controller component for trained or built-in RL models.
- a dispatch registration for RL controllers that emits normal Bunnyland `ToolCall`s.
- `/admin/rl/*` FastAPI routes for offline training jobs, model listing, and assignment.

Training jobs are offline arena jobs seeded from the live world state. They do not submit
live-world commands. Completed jobs save model metadata/artifacts to `BUNNYLAND_RL_DIR`
(default: `data/rl`) and models are reloadable after server restart.

W&B tracking is optional and lazy. Set `BUNNYLAND_RL_WANDB=1` or configure `WANDB_API_KEY`
/ `WANDB_MODE=offline`; jobs then log metrics, checkpoint paths, lens trust weights, action
histograms, and saved model artifacts. Install the `tracking` extra to include `wandb`.

Run with:

```bash
bunnyland serve --module bunnyland_3d --module bunnyland_rl ...
```
