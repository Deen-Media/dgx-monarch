import test from 'node:test';
import assert from 'node:assert/strict';
import { createRawSnippet } from 'svelte';
import { emptyDraft } from './model.ts';
import { renderComponent, visibleText } from './renderHarness.js';

/** The labels of every rendered button. */
const labels = (body: string) =>
  [...body.matchAll(/<button([^>]*)>((?:<!--[^>]*-->|[^<])*)<\/button>/g)].map(([, tag, label]) =>
    `${tag} ${label}`.replace(/<!--[^>]*-->/g, '').trim(),
  );

test('the telemetry panel has no close button; the footer and Settings show and hide it', async () => {
  const body = await renderComponent('./views/WorkspaceOutput.svelte', {
    client: { request: async () => ({}) },
    selected: null,
    comparison: null,
    artifacts: [],
    telemetry: { state: 'connected', observed_at: 1, sample: {} },
    history: [],
    connectionState: 'connected',
    activeCount: 0,
    showTelemetry: true,
    onselect() {},
    oncompare() {},
    navigate() {},
    reuse: async () => {},
  });
  const panel = body.slice(body.indexOf('<aside class="telemetry-panel"'));
  assert.ok(visibleText(panel).startsWith('Spark telemetry'), visibleText(panel).slice(0, 60));
  assert.deepEqual(labels(panel), [], 'the panel holds no buttons');
  assert.doesNotMatch(body, /Collapse telemetry|×/);
});

test('the event console has no close button; Clear view and both footer toggles remain', async () => {
  const body = await renderComponent('./components/AppShell.svelte', {
    connection: {
      connected: true,
      state: 'connected',
      hidden: false,
      lastSeen: '',
      logs: ['event one'],
      client: { request: async () => ({ records: [], next_offset: null }) },
      capabilities: { host: '127.0.0.1:8765' },
      snapshot: { records: { projects: [] } },
    },
    draft: emptyDraft(),
    activeCount: 0,
    view: 'workspace',
    presentation: false,
    width: 340,
    showTelemetry: true,
    showConsole: true,
    navigate() {},
    changeDraft() {},
    openPalette() {},
    children: createRawSnippet(() => ({ render: () => '<p>workspace</p>' })),
  });
  const consolePanel = body.slice(body.indexOf('<section class="console-panel"'));
  assert.ok(visibleText(consolePanel).startsWith('Event console'), visibleText(consolePanel));
  assert.deepEqual(labels(consolePanel), ['Clear view']);
  assert.doesNotMatch(body, /Close console|×/);
  const footer = labels(
    body.slice(body.indexOf('<footer class="app-footer"'), body.indexOf('</footer>')),
  );
  assert.equal(footer.length, 2);
  assert.match(footer[0], /aria-expanded="true" aria-controls="studio-telemetry".*Telemetry −$/);
  assert.match(footer[1], /aria-expanded="true" aria-controls="studio-console".*Console −$/);
});

test('Settings keeps a checkbox for each panel', async () => {
  // Settings names the browser's own host when the companion reports none.
  globalThis.location = { host: '127.0.0.1:8765' } as Location;
  const text = visibleText(
    await renderComponent('./views/Settings.svelte', {
      draft: emptyDraft(),
      width: 340,
      showTelemetry: false,
      showConsole: false,
      capabilities: {},
      onimport: async () => {},
      disconnect() {},
      fail() {},
    }),
  );
  assert.ok(text.includes('Spark telemetry panel'), text);
  assert.ok(text.includes('Event console'), text);
});
