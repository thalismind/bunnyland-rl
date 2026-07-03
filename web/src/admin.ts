import { normalizeBase, sendJson } from '@bunnyland/ui-web/api';
import { escapeHtml } from '@bunnyland/ui-web/widgets';
import './style.css';

type Job = {
  job_id: string;
  status: string;
  config: Record<string, unknown>;
  metrics: {
    episode: number;
    update: number;
    reward_curve: number[];
    loss_curve: number[];
    action_histogram: Record<string, number>;
    trust_weights: Record<string, number>;
  };
  latest_checkpoint: string | null;
  model_id: string | null;
  error: string;
  wandb_url: string | null;
};

type Model = {
  model_id: string;
  created_at_unix: number;
  config: Record<string, unknown>;
  metrics: Job['metrics'];
  checkpoint_path: string;
  weights_path: string;
  weights_format: string;
  artifact_path: string;
  wandb_url: string | null;
};

type Character = {
  character_id: string;
  name: string;
  kind: string;
  suspended: boolean;
};

type State = {
  apiBase: string;
  adminSecret: string;
  characters: Character[];
  behaviors: string[];
  jobs: Job[];
  models: Model[];
  selectedCharacter: string;
  selectedBehavior: string;
  assignCharacter: string;
  error: string;
};

const state: State = {
  apiBase: normalizeBase(new URLSearchParams(location.search).get('server') || '/api'),
  adminSecret: localStorage.getItem('bunnyland-admin-secret') || '',
  characters: [],
  behaviors: [],
  jobs: [],
  models: [],
  selectedCharacter: '',
  selectedBehavior: 'idle',
  assignCharacter: '',
  error: '',
};

const app = document.getElementById('app');
if (!app) {
  throw new Error('missing app root');
}
const root = app;

function headers(): Record<string, string> {
  return state.adminSecret ? { 'X-Bunnyland-Admin-Secret': state.adminSecret } : {};
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${state.apiBase}${path}`, { headers: headers() });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json() as Promise<T>;
}

async function refresh(): Promise<void> {
  try {
    const [characters, definitions, jobs, models] = await Promise.all([
      getJson<{ characters: Character[] }>('/world/characters'),
      getJson<{ behaviors: string[] }>('/admin/controllers/definitions'),
      getJson<{ jobs: Job[] }>('/admin/rl/training/jobs'),
      getJson<{ models: Model[] }>('/admin/rl/models'),
    ]);
    state.characters = characters.characters;
    state.behaviors = definitions.behaviors;
    state.jobs = jobs.jobs;
    state.models = models.models;
    if (!state.selectedCharacter && state.characters[0]) {
      state.selectedCharacter = state.characters[0].character_id;
    }
    if (!state.assignCharacter && state.characters[0]) {
      state.assignCharacter = state.characters[0].character_id;
    }
    if (!state.behaviors.includes(state.selectedBehavior) && state.behaviors[0]) {
      state.selectedBehavior = state.behaviors.includes('idle') ? 'idle' : state.behaviors[0];
    }
    state.error = '';
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  }
  render();
}

async function startJob(): Promise<void> {
  const characterId = (document.getElementById('character-id') as HTMLSelectElement).value;
  state.selectedCharacter = characterId;
  const behaviorName = (document.getElementById('behavior-name') as HTMLSelectElement).value;
  state.selectedBehavior = behaviorName;
  const policyNet = (document.getElementById('policy-net') as HTMLSelectElement).value;
  const lenses = Array.from(document.querySelectorAll<HTMLInputElement>('[data-lens]:checked')).map(
    input => input.value,
  );
  await sendJson(state.apiBase, '/admin/rl/training/jobs', {
    body: JSON.stringify({
      character_id: characterId,
      policy_net: policyNet,
      lenses,
      mode: 'behavior_overlay',
      behavior_name: behaviorName,
      episodes: Number((document.getElementById('episodes') as HTMLInputElement).value || 8),
      updates_per_episode: Number((document.getElementById('updates') as HTMLInputElement).value || 4),
      seed: (document.getElementById('seed') as HTMLInputElement).value,
    }),
    headers: headers(),
    method: 'POST',
  });
  await refresh();
}

async function cancelJob(jobId: string): Promise<void> {
  await sendJson(state.apiBase, `/admin/rl/training/jobs/${encodeURIComponent(jobId)}/cancel`, {
    body: JSON.stringify({}),
    headers: headers(),
    method: 'POST',
  });
  await refresh();
}

async function assignModel(modelId: string): Promise<void> {
  const characterId = (document.getElementById('assign-character-id') as HTMLSelectElement).value;
  state.assignCharacter = characterId;
  await sendJson(state.apiBase, `/admin/rl/models/${encodeURIComponent(modelId)}/assign`, {
    body: JSON.stringify({ character_id: characterId }),
    headers: headers(),
    method: 'POST',
  });
  await refresh();
}

function render(): void {
  root.innerHTML = `
    <section class="shell">
      <header>
        <h1>RL Admin</h1>
        <div class="connection">
          <label>API URL
            <input id="api-base" value="${escapeHtml(state.apiBase)}" placeholder="/api" />
          </label>
          <label>Admin secret
            <input id="admin-secret" value="${escapeHtml(state.adminSecret)}" type="password" placeholder="demo-admin" />
          </label>
          <button id="refresh">Refresh</button>
        </div>
      </header>
      ${state.error ? `<p class="error">${escapeHtml(state.error)}</p>` : ''}
      <section class="controls">
        <label>Character
          ${renderCharacterSelect('character-id', state.selectedCharacter)}
        </label>
        <label>Base behavior
          ${renderBehaviorSelect('behavior-name', state.selectedBehavior)}
        </label>
        <label>Policy
          <select id="policy-net">
            <option value="mlp">mlp</option>
            <option value="deep">deep</option>
            <option value="residual">residual</option>
          </select>
        </label>
        <label>Episodes <input id="episodes" type="number" min="1" value="8" /></label>
        <label>Updates <input id="updates" type="number" min="1" value="4" /></label>
        <label>Seed <input id="seed" /></label>
        <fieldset>
          <legend>Lenses</legend>
          ${['room_text', 'perception_text', 'stats_vector', 'components_vector', 'room_grid']
            .map(lens => `<label><input data-lens type="checkbox" value="${lens}" checked /> ${lens}</label>`)
            .join('')}
        </fieldset>
        <button id="start-job">Start</button>
      </section>
      <section class="characters">
        <h2>Characters</h2>
        ${renderCharacters(state.characters)}
      </section>
      <section class="grid">
        <article>
          <h2>Training Jobs</h2>
          ${renderJobs(state.jobs)}
        </article>
        <article>
          <h2>Models</h2>
          <label>Assign to
            ${renderCharacterSelect('assign-character-id', state.assignCharacter)}
          </label>
          ${renderModels(state.models)}
        </article>
      </section>
    </section>
  `;
  bind();
}

function renderCharacterSelect(id: string, selected: string): string {
  if (!state.characters.length) {
    return `<select id="${id}" disabled><option>No characters loaded</option></select>`;
  }
  return `
    <select id="${id}">
      ${state.characters.map(character => `
        <option value="${escapeHtml(character.character_id)}" ${character.character_id === selected ? 'selected' : ''}>
          ${escapeHtml(character.name)} (${escapeHtml(character.character_id)})
        </option>
      `).join('')}
    </select>
  `;
}

function renderBehaviorSelect(id: string, selected: string): string {
  if (!state.behaviors.length) {
    return `<select id="${id}" disabled><option>No behaviors loaded</option></select>`;
  }
  return `
    <select id="${id}">
      ${state.behaviors.map(behavior => `
        <option value="${escapeHtml(behavior)}" ${behavior === selected ? 'selected' : ''}>
          ${escapeHtml(behavior)}
        </option>
      `).join('')}
    </select>
  `;
}

function renderCharacters(characters: Character[]): string {
  if (!characters.length) {
    return '<p class="empty">No characters loaded.</p>';
  }
  return `
    <div class="character-list">
      ${characters.map(character => `
        <div class="character-pill">
          <strong>${escapeHtml(character.name)}</strong>
          <span>${escapeHtml(character.character_id)}</span>
          ${character.suspended ? '<em>suspended</em>' : ''}
        </div>
      `).join('')}
    </div>
  `;
}

function renderJobs(jobs: Job[]): string {
  if (!jobs.length) {
    return '<p class="empty">No jobs.</p>';
  }
  return jobs.map(job => `
    <section class="row">
      <div>
        <h3>${escapeHtml(job.job_id)}</h3>
        <p>${escapeHtml(job.status)} · episode ${job.metrics.episode} update ${job.metrics.update}</p>
        <p>checkpoint: ${escapeHtml(job.latest_checkpoint || '')}</p>
        ${job.wandb_url ? `<p><a href="${escapeHtml(job.wandb_url)}">W&amp;B run</a></p>` : ''}
        ${job.error ? `<p class="error">${escapeHtml(job.error)}</p>` : ''}
      </div>
      <canvas data-chart="${escapeHtml(job.job_id)}" width="220" height="80"></canvas>
      <pre>${escapeHtml(JSON.stringify(job.metrics.action_histogram, null, 2))}</pre>
      <pre>${escapeHtml(JSON.stringify(job.metrics.trust_weights, null, 2))}</pre>
      ${job.status === 'queued' || job.status === 'running' ? `<button data-cancel="${escapeHtml(job.job_id)}">Cancel</button>` : ''}
    </section>
  `).join('');
}

function renderModels(models: Model[]): string {
  if (!models.length) {
    return '<p class="empty">No saved models.</p>';
  }
  return models.map(model => `
    <section class="row">
      <div>
        <h3>${escapeHtml(model.model_id)}</h3>
        <p>artifact: ${escapeHtml(model.artifact_path)}</p>
        <p>checkpoint: ${escapeHtml(model.checkpoint_path)}</p>
        <p>weights: ${escapeHtml(model.weights_path)} (${escapeHtml(model.weights_format)})</p>
        ${model.wandb_url ? `<p><a href="${escapeHtml(model.wandb_url)}">W&amp;B run</a></p>` : ''}
      </div>
      <button data-assign="${escapeHtml(model.model_id)}">Assign</button>
    </section>
  `).join('');
}

function bind(): void {
  document.getElementById('refresh')?.addEventListener('click', () => {
    state.apiBase = normalizeBase((document.getElementById('api-base') as HTMLInputElement).value);
    state.adminSecret = (document.getElementById('admin-secret') as HTMLInputElement).value;
    localStorage.setItem('bunnyland-admin-secret', state.adminSecret);
    void refresh();
  });
  document.getElementById('start-job')?.addEventListener('click', () => void startJob());
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-cancel]')) {
    button.addEventListener('click', () => void cancelJob(button.dataset.cancel || ''));
  }
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-assign]')) {
    button.addEventListener('click', () => void assignModel(button.dataset.assign || ''));
  }
  drawCharts();
}

function drawCharts(): void {
  for (const job of state.jobs) {
    const canvas = document.querySelector<HTMLCanvasElement>(`canvas[data-chart="${CSS.escape(job.job_id)}"]`);
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) {
      continue;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawLine(ctx, job.metrics.reward_curve, '#2f855a', canvas.width, canvas.height);
    drawLine(ctx, job.metrics.loss_curve, '#c53030', canvas.width, canvas.height);
  }
}

function drawLine(ctx: CanvasRenderingContext2D, values: number[], color: string, width: number, height: number): void {
  if (!values.length) {
    return;
  }
  const max = Math.max(...values, 1);
  ctx.strokeStyle = color;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = values.length === 1 ? 0 : (index / (values.length - 1)) * width;
    const y = height - (value / max) * (height - 8) - 4;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

render();
void refresh();
setInterval(refresh, 5000);
