# Bunnyland RL administration

The RL add-on contributes a server plugin and a static admin dashboard. The server plugin
adds `RLControllerComponent`, offline training jobs, saved model artifacts, model assignment,
and `/admin/rl/*` API routes. The web bundle adds the `/rl/` dashboard.

Use this guide to install the add-on, train models, assign controllers, and inspect saved
artifacts.

## Server plugin

The Python package exposes `bunnyland_rl.bunnyland_plugins()` and registers plugin id
`bunnyland.rl`.

When loaded, it contributes:

- `RLControllerComponent` for assigned RL controller state;
- runtime controller registration that emits normal Bunnyland `ToolCall`s;
- `/admin/rl/status`;
- `/admin/rl/training/jobs`;
- `/admin/rl/training/jobs/{job_id}`;
- `/admin/rl/training/jobs/{job_id}/cancel`;
- `/admin/rl/models`;
- `/admin/rl/models/{model_id}`;
- `/admin/rl/models/{model_id}/weights/preview`;
- `/admin/rl/models/{model_id}/assign`.

Load it with the stock Bunnyland server:

```bash
bunnyland serve --module bunnyland_rl
```

It can be loaded alongside the 3D add-on:

```bash
bunnyland serve --module bunnyland_3d --module bunnyland_rl
```

The plugin is `default_enabled=True`, so no separate `--plugin` flag is required once the
module is importable. If a container or supervisor overrides the server command, keep
`--module bunnyland_rl` in the final arguments.

## Storage and tracking

Training jobs run offline arena copies seeded from the live world state. They do not submit
commands to the live world.

Completed jobs write reloadable model metadata under `BUNNYLAND_RL_DIR`, which defaults to
`data/rl`. Neural-network weights are stored separately as safetensors files under
`BUNNYLAND_RL_DIR/weights/`.

Set `BUNNYLAND_RL_WANDB=1` or normal `WANDB_*` settings to enable Weights & Biases tracking.
Install the `tracking` extra when you want the optional `wandb` dependency available.

## Dashboard

The web image copies the dashboard into `/usr/share/nginx/html/rl`.

Open `/rl/`, set the server field to `/api` for same-origin deployments, and provide the admin
secret. The dashboard stores the secret in browser `localStorage`.

The dashboard can:

- list playable characters;
- list available base controller behaviors;
- start offline training jobs;
- show job status, reward and loss curves, action histograms, trust weights, checkpoint
  paths, and W&B links;
- cancel active jobs;
- list saved models;
- preview safetensors weight layers;
- assign a saved model to a character.

Protect the dashboard and `/admin/rl/*` routes the same way you protect other admin tools.

## Start a training job

Pick a character and a base behavior. The base behavior is used as the behavior overlay mode
while the model learns. Choose a policy network, lens set, episode count, update count, and
seed.

The default lens set is:

| Lens | Purpose |
|------|---------|
| `room_text` | Text about the current room. |
| `perception_text` | Text from the character's perception. |
| `stats_vector` | Numeric character stats. |

Additional lenses may be available, such as `components_vector` and `room_grid`. Larger lens
sets can expose more state but also increase training cost and model complexity.

Start small. Short jobs are useful for checking plumbing, artifact writes, and dashboard
rendering before spending time on longer runs.

## Assign a model

After a job completes, select a saved model and assign it to a character. Assignment writes
`RLControllerComponent` to that character with the selected model id and runtime settings.

Watch the character in a client or inspector after assignment. Confirm it acts at the expected
pace, emits normal queued commands, and can be claimed by a human when needed.

## Build the images

Build the server image from the add-on repo:

```bash
docker build -f Dockerfile.server \
  --build-arg BUNNYLAND_SERVER_IMAGE=ghcr.io/thalismind/bunnyland-server:main \
  -t bunnyland-rl-server .
```

Build the web image with the sibling shared UI package as a build context:

```bash
docker build -f Dockerfile.web \
  --build-context bunnyland-ui-web=../bunnyland-ui-web \
  --build-arg BUNNYLAND_WEB_IMAGE=ghcr.io/thalismind/bunnyland-web:main \
  -t bunnyland-rl-web .
```

## Local checks

Run all add-on checks from the repo root:

```bash
scripts/check
```

Run only the server plugin checks:

```bash
BUNNYLAND_SERVER_PATH=../bunnyland-server scripts/test-server
```

Run only the web checks:

```bash
scripts/test-web
```

## Verify a deployment

After deployment:

1. Open `/rl/` and connect to `/api`.
2. Confirm characters, jobs, and models load.
3. Start a short training job against a test character.
4. Confirm the job reaches a terminal state or can be cancelled.
5. Confirm a completed job writes JSON metadata and safetensors weights under
   `BUNNYLAND_RL_DIR`.
6. Assign a model to a test character.
7. Claim and release that character from a player client to confirm handoff still works.

## Operational notes

RL is controller infrastructure, not a privilege layer. Model output still goes through
normal Bunnyland action validation, queueing, costs, and rejection paths.

Keep saved model directories backed up if you want models to survive server replacement. Do
not commit generated model artifacts, checkpoints, safetensors files, `node_modules`, or
dashboard build output unless the repo's ignore rules explicitly call for them.
