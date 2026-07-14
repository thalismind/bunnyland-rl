# Bunnyland RL

Out-of-tree [Bunnyland](https://github.com/thalismind/bunnyland-server) plugin that adds
reinforcement-learning controller state, offline arena training, model assignment, and an
admin dashboard.

Training v1 runs offline jobs seeded from live world state. Jobs do not submit live commands;
they learn from an arena copy, save reloadable model metadata under `BUNNYLAND_RL_DIR`, and
write neural-network weights as safetensors files. Trained models can then be assigned to
characters through the admin API or dashboard as normal Bunnyland controllers.

This repo intentionally keeps the RL work outside the main `bunnyland-server` and
`bunnyland-web` repos so controller experiments can evolve without becoming required server
runtime.

## Layout

- `server/` - Python Bunnyland plugin package with RL controller components, runtime
  controller dispatch, offline training, policy networks, lenses, admin API routes, W&B
  tracking helpers, and tests.
- `web/` - standalone Vite admin dashboard for training jobs, model assignment, and weight
  previews.
- `scripts/test-server` - runs Python tests against a sibling `bunnyland-server` checkout.
- `scripts/test-web` - runs the web checks.
- `scripts/check` - runs both server and web checks.
- `Dockerfile.server` - extends the published Bunnyland server image with the RL plugin.
- `Dockerfile.web` - extends the published Bunnyland web image with `/rl/` static assets.

## Server Plugin

The plugin exposes `bunnyland_rl.bunnyland_plugins()` and contributes:

- `RLControllerComponent` - ECS controller state for a trained or built-in RL policy.
- RL controller runtime registration - turns assigned controller state into normal Bunnyland
  `ToolCall`s, so RL output uses the same action pipeline as other controllers.
- `/admin/rl/status` - reports schema, world epoch, available policy networks, lenses, saved
  models, controller component name, and W&B status.
- `/admin/rl/training/jobs` - creates and lists offline training jobs.
- `/admin/rl/training/jobs/{job_id}` - reads a specific job.
- `/admin/rl/training/jobs/{job_id}/cancel` - requests cancellation for a job.
- `/admin/rl/models` and `/admin/rl/models/{model_id}` - lists and reads saved models.
- `/admin/rl/models/{model_id}/weights/preview` - returns a bounded preview of safetensors
  weights for dashboard inspection.
- `/admin/rl/models/{model_id}/assign` - assigns a trained model to a character.

`default_enabled=True`, so loading the module is enough for Bunnyland to register the plugin.
The `bunnyland_rl` package must be importable by the server, either installed into the server
environment or available on `PYTHONPATH`.

## Training Artifacts

Training v1 runs offline arena jobs from the live world state. Jobs do not submit live
commands. Completed jobs save reloadable JSON model artifacts under `BUNNYLAND_RL_DIR`
(default `data/rl`). The JSON metadata points at safetensors NN weights under
`BUNNYLAND_RL_DIR/weights/`.

Set `BUNNYLAND_RL_WANDB=1` (or provide normal `WANDB_*` settings) to track jobs in
Weights & Biases. The plugin logs per-update reward/loss/action/trust/checkpoint stats
and records the saved model JSON/checkpoint/safetensors files as a W&B artifact.

Install the `tracking` extra to include the optional `wandb` dependency.

## Running

Load the server plugin with the stock Bunnyland server:

```bash
bunnyland serve --module bunnyland_rl
```

The RL dashboard is often deployed next to the 3D plugin, but the server plugin does not
require 3D:

```bash
bunnyland serve --module bunnyland_3d --module bunnyland_rl ...
```

If a deployment overrides the container command, keep `--module bunnyland_rl` in the server
arguments so the controller component, runtime hooks, and admin routes are loaded.

## Web Dashboard

The web app is a Vite dashboard served at `/rl/` in the Docker image. It talks to the
Bunnyland admin API using the same secure HttpOnly, same-origin login cookie as the hosted
web client; it does not persist bearer credentials in browser storage.

```bash
cd web
npm install
npm run dev
```

The dashboard can:

- connect to the same-origin `/api` endpoint;
- list playable characters and available base controller behaviors;
- start offline training jobs with selected policy networks, lenses, episode counts, updates,
  and seeds;
- show job status, reward/loss curves, action histograms, trust weights, checkpoints, and W&B
  links when tracking is enabled;
- cancel active jobs;
- list saved models and assign them to characters;
- preview safetensors weight layers with bounded row/column sampling.

## Docker Images

The root Dockerfiles extend the published Bunnyland images instead of replacing them:

```bash
docker build -f Dockerfile.server \
  --build-arg BUNNYLAND_SERVER_IMAGE=ghcr.io/thalismind/bunnyland-server:main \
  -t bunnyland-rl-server .

docker build -f Dockerfile.web \
  --build-context bunnyland-ui-web=../bunnyland-ui-web \
  --build-arg BUNNYLAND_WEB_IMAGE=ghcr.io/thalismind/bunnyland-web:main \
  -t bunnyland-rl-web .
```

`Dockerfile.server` installs the out-of-tree Python plugin into the base server virtualenv
and uses a default `bunnyland serve --module bunnyland_rl ...` command.

`Dockerfile.web` builds the dashboard and copies it into the extended web image at
`/usr/share/nginx/html/rl`.

## Development

Run all checks from the repo root:

```bash
scripts/check
```

For focused server work:

```bash
BUNNYLAND_SERVER_PATH=../bunnyland-server scripts/test-server
```

For focused web work:

```bash
scripts/test-web
```

See [`server/README.md`](server/README.md) for the server plugin summary.

## Contributing & Conduct

This plugin follows the Bunnyland project's [contribution guidelines](CONTRIBUTING.md) and
[code of conduct](CODE_OF_CONDUCT.md), which point back to the `bunnyland-server`
repository.

## License

Licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
