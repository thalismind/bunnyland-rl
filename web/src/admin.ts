import '@bunnyland/ui-web/assets/bunnyland-ui.css';
import { assertSameOriginBase, login, sendJson, serverFromUrl } from '@bunnyland/ui-web/api';
import { bindThemeSelect } from '@bunnyland/ui-web/theme';
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

type WeightLayerSummary = {
  name: string;
  shape: number[];
  dtype: string;
  size: number;
};

type WeightLayerPreview = WeightLayerSummary & {
  rows: number;
  columns: number;
  row_indices: number[];
  column_indices: number[];
  values: number[][];
  min: number;
  max: number;
  mean: number;
  downsampled: boolean;
};

type WeightPreview = {
  model_id: string;
  layers: WeightLayerSummary[];
  selected_layer: WeightLayerPreview | null;
};

type Character = {
  character_id: string;
  name: string;
  kind: string;
  suspended: boolean;
};

type State = {
  apiBase: string;
  characters: Character[];
  behaviors: string[];
  jobs: Job[];
  models: Model[];
  selectedCharacter: string;
  selectedBehavior: string;
  assignCharacter: string;
  weightPreview: WeightPreview | null;
  error: string;
};

const state: State = {
  apiBase: assertSameOriginBase(serverFromUrl() || '/api'),
  characters: [],
  behaviors: [],
  jobs: [],
  models: [],
  selectedCharacter: '',
  selectedBehavior: 'idle',
  assignCharacter: '',
  weightPreview: null,
  error: '',
};

const app = document.getElementById('app');
if (!app) {
  throw new Error('missing app root');
}
const root = app;

function setApiStatus(text: string, cls: '' | 'ok' | 'err'): void {
  const status = document.getElementById('api-status');
  if (!status) {
    return;
  }
  status.textContent = text;
  status.className = cls;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${assertSameOriginBase(state.apiBase)}${path}`, {
    credentials: 'include',
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} from ${state.apiBase}${path}: ${await response.text()}`);
  }
  return response.json() as Promise<T>;
}

async function refresh(): Promise<void> {
  try {
    const [characters, definitions, jobs, models] = await Promise.all([
      getJson<{ characters: Character[] }>('/play/world/characters'),
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
    setApiStatus(`● ${state.characters.length} characters · ${state.jobs.length} jobs`, 'ok');
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
    setApiStatus('○ Not connected', 'err');
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
    method: 'POST',
  });
  await refresh();
}

async function cancelJob(jobId: string): Promise<void> {
  await sendJson(state.apiBase, `/admin/rl/training/jobs/${encodeURIComponent(jobId)}/cancel`, {
    body: JSON.stringify({}),
    method: 'POST',
  });
  await refresh();
}

async function assignModel(modelId: string): Promise<void> {
  const characterId = (document.getElementById('assign-character-id') as HTMLSelectElement).value;
  state.assignCharacter = characterId;
  await sendJson(state.apiBase, `/admin/rl/models/${encodeURIComponent(modelId)}/assign`, {
    body: JSON.stringify({ character_id: characterId }),
    method: 'POST',
  });
  await refresh();
}

async function previewModel(modelId: string, layer = ''): Promise<void> {
  const params = new URLSearchParams({ max_rows: '512', max_columns: '512' });
  if (layer) {
    params.set('layer', layer);
  }
  state.weightPreview = await getJson<WeightPreview>(
    `/admin/rl/models/${encodeURIComponent(modelId)}/weights/preview?${params}`,
  );
  render();
}

function render(): void {
  root.innerHTML = `
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
        ${renderWeightPreview(state.weightPreview)}
      </article>
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
        ${safeExternalUrl(job.wandb_url) ? `<p><a href="${escapeHtml(safeExternalUrl(job.wandb_url))}">W&amp;B run</a></p>` : ''}
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
        ${safeExternalUrl(model.wandb_url) ? `<p><a href="${escapeHtml(safeExternalUrl(model.wandb_url))}">W&amp;B run</a></p>` : ''}
      </div>
      <div class="model-actions">
        <button data-preview="${escapeHtml(model.model_id)}">Preview</button>
        <button data-assign="${escapeHtml(model.model_id)}">Assign</button>
      </div>
    </section>
  `).join('');
}

function renderWeightPreview(preview: WeightPreview | null): string {
  if (!preview) {
    return '';
  }
  const selected = preview.selected_layer;
  const selectedName = selected?.name || '';
  return `
    <section class="weight-preview">
      <div class="preview-header">
        <div>
          <h3>Weights Preview</h3>
          <p>${escapeHtml(preview.model_id)}</p>
        </div>
        <label>Layer
          <select data-preview-layer="${escapeHtml(preview.model_id)}">
            ${preview.layers.map(layer => `
              <option value="${escapeHtml(layer.name)}" ${layer.name === selectedName ? 'selected' : ''}>
                ${escapeHtml(layer.name)} ${escapeHtml(formatShape(layer.shape))}
              </option>
            `).join('')}
          </select>
        </label>
      </div>
      ${selected ? renderLayerPreview(selected) : '<p class="empty">No tensor layers.</p>'}
    </section>
  `;
}

function renderLayerPreview(layer: WeightLayerPreview): string {
  return `
    <div class="layer-meta">
      <span>${escapeHtml(formatShape(layer.shape))}</span>
      <span>${escapeHtml(layer.dtype)}</span>
      <span>${layer.rows} x ${layer.columns}</span>
      <span>min ${formatNumber(layer.min)}</span>
      <span>mean ${formatNumber(layer.mean)}</span>
      <span>max ${formatNumber(layer.max)}</span>
      ${layer.downsampled ? '<span>downsampled</span>' : ''}
    </div>
    <div class="heatmap-scroll">
      <canvas
        class="heatmap-canvas"
        data-weight-heatmap
        width="${layer.column_indices.length}"
        height="${layer.row_indices.length}"
        style="--heatmap-columns:${layer.column_indices.length}"
      ></canvas>
    </div>
    <p class="heatmap-readout" id="heatmap-readout">Hover a cell for row, column, and value.</p>
  `;
}

function formatShape(shape: number[]): string {
  return `[${shape.join(', ')}]`;
}

function safeExternalUrl(value: string | null): string {
  if (!value) return '';
  try {
    const url = new URL(value);
    return url.protocol === 'https:' ? url.href : '';
  } catch {
    return '';
  }
}

function formatNumber(value: number): string {
  return Number.isFinite(value) ? value.toFixed(4) : String(value);
}

function heatColor(value: number, min: number, max: number): [number, number, number] {
  const neutral = [42, 45, 53];
  const positive = [210, 82, 48];
  const negative = [48, 114, 186];
  const scale = Math.max(Math.abs(min), Math.abs(max), 1e-9);
  const target = value < 0 ? negative : positive;
  const ratio = Math.min(1, Math.abs(value) / scale);
  const mix = 0.18 + ratio * 0.82;
  return neutral.map((channel, index) => Math.round(channel + (target[index] - channel) * mix)) as [number, number, number];
}

// The connection controls and theme selector live in the static #toolbar, so they are
// wired once at startup rather than on every content re-render.
function initToolbar(): void {
  bindThemeSelect(document.getElementById('theme-select') as HTMLSelectElement | null);
  const apiInput = document.getElementById('api-base') as HTMLInputElement | null;
  const usernameInput = document.getElementById('username') as HTMLInputElement | null;
  const passwordInput = document.getElementById('password') as HTMLInputElement | null;
  if (apiInput) {
    apiInput.value = state.apiBase;
  }
  document.getElementById('refresh')?.addEventListener('click', () => {
    state.apiBase = assertSameOriginBase(apiInput?.value || '');
    void (async () => {
      if (usernameInput?.value && passwordInput?.value) {
        await login(state.apiBase, usernameInput.value, passwordInput.value);
        passwordInput.value = '';
      }
      await refresh();
    })();
  });
}

function bind(): void {
  document.getElementById('start-job')?.addEventListener('click', () => void startJob());
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-cancel]')) {
    button.addEventListener('click', () => void cancelJob(button.dataset.cancel || ''));
  }
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-assign]')) {
    button.addEventListener('click', () => void assignModel(button.dataset.assign || ''));
  }
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-preview]')) {
    button.addEventListener('click', () => void previewModel(button.dataset.preview || ''));
  }
  for (const select of document.querySelectorAll<HTMLSelectElement>('[data-preview-layer]')) {
    select.addEventListener('change', () => void previewModel(select.dataset.previewLayer || '', select.value));
  }
  drawCharts();
  drawWeightPreview();
}

function themeColor(name: string, fallback: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

function drawCharts(): void {
  const rewardColor = themeColor('--bl-ok', '#2f855a');
  const lossColor = themeColor('--bl-error', '#c53030');
  for (const job of state.jobs) {
    const canvas = document.querySelector<HTMLCanvasElement>(`canvas[data-chart="${CSS.escape(job.job_id)}"]`);
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) {
      continue;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawLine(ctx, job.metrics.reward_curve, rewardColor, canvas.width, canvas.height);
    drawLine(ctx, job.metrics.loss_curve, lossColor, canvas.width, canvas.height);
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

function drawWeightPreview(): void {
  const layer = state.weightPreview?.selected_layer;
  const canvas = document.querySelector<HTMLCanvasElement>('[data-weight-heatmap]');
  const ctx = canvas?.getContext('2d');
  if (!layer || !canvas || !ctx || !layer.values.length) {
    return;
  }
  const rows = layer.values.length;
  const columns = layer.values[0]?.length || 0;
  if (!columns) {
    return;
  }
  canvas.width = columns;
  canvas.height = rows;
  const image = ctx.createImageData(columns, rows);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const [red, green, blue] = heatColor(layer.values[row][column], layer.min, layer.max);
      const offset = (row * columns + column) * 4;
      image.data[offset] = red;
      image.data[offset + 1] = green;
      image.data[offset + 2] = blue;
      image.data[offset + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  canvas.onmousemove = event => updateHeatmapReadout(event, canvas, layer);
  canvas.onmouseleave = () => {
    const readout = document.getElementById('heatmap-readout');
    if (readout) {
      readout.textContent = 'Hover a cell for row, column, and value.';
    }
  };
}

function updateHeatmapReadout(
  event: MouseEvent,
  canvas: HTMLCanvasElement,
  layer: WeightLayerPreview,
): void {
  const rect = canvas.getBoundingClientRect();
  const column = Math.min(
    layer.column_indices.length - 1,
    Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * layer.column_indices.length)),
  );
  const row = Math.min(
    layer.row_indices.length - 1,
    Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * layer.row_indices.length)),
  );
  const readout = document.getElementById('heatmap-readout');
  if (readout) {
    readout.textContent = `row ${layer.row_indices[row]} · column ${layer.column_indices[column]} · ${formatNumber(layer.values[row][column])}`;
  }
}

initToolbar();
render();
void refresh();
setInterval(refresh, 5000);
