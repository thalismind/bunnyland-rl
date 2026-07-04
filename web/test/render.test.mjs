import assert from 'node:assert/strict';
import test from 'node:test';

test('dashboard source includes required admin controls', async () => {
  const source = await import('node:fs/promises').then(fs => fs.readFile(new URL('../src/admin.ts', import.meta.url), 'utf8'));
  assert.match(source, /Training Jobs/);
  assert.match(source, /Models/);
  assert.match(source, /data-cancel/);
  assert.match(source, /data-assign/);
  assert.match(source, /data-preview/);
  assert.match(source, /data-weight-heatmap/);
  assert.match(source, /downsampled/);
});
