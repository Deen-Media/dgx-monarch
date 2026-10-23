import test from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { compile, compileModule } from 'svelte/compiler';

// Server-render the real components without a DOM: .svelte files and rune
// modules compile on import, and extensionless relative imports resolve to .ts.
registerHooks({
  resolve(specifier, context, next) {
    if (specifier.startsWith('.') && context.parentURL?.startsWith('file:'))
      for (const candidate of [specifier, `${specifier}.ts`]) {
        const url = new URL(candidate, context.parentURL);
        if (existsSync(fileURLToPath(url))) return { url: url.href, shortCircuit: true };
      }
    return next(specifier, context);
  },
  load(url, context, next) {
    const path = url.startsWith('file:') ? fileURLToPath(url) : '';
    if (path.endsWith('.svelte'))
      return {
        format: 'module',
        source: compile(readFileSync(path, 'utf8'), { filename: path, generate: 'server' }).js.code,
        shortCircuit: true,
      };
    if (path.endsWith('.svelte.ts')) {
      const plain = ts.transpile(readFileSync(path, 'utf8'), {
        target: ts.ScriptTarget.ESNext,
        module: ts.ModuleKind.ESNext,
      });
      return {
        format: 'module',
        source: compileModule(plain, { filename: path, generate: 'server' }).js.code,
        shortCircuit: true,
      };
    }
    return next(url, context);
  },
});
globalThis.location = { host: '127.0.0.1:8765' } as Location;
const { render } = await import('svelte/server');
const component = async (path: string) => (await import(path)).default;
const Generation = await component('./views/Generation.svelte');
const AssetEditor = await component('./components/AssetEditor.svelte');
const Field = await component('./components/Field.svelte');
const StatusMessages = await component('./components/StatusMessages.svelte');
const Connect = await component('./views/Connect.svelte');
const DriverNotRunning = await component('./components/DriverNotRunning.svelte');
const { assetEditor } = await import('./assetEditorState.svelte.ts');
const { emptyDraft } = await import('./model.ts');

const html = (Component: unknown, props: Record<string, unknown>) =>
  render(Component as never, { props }).body;
/** The opening tag of the button whose text starts with `label`. */
function tagBefore(body: string, label: string) {
  const end = body.indexOf(label);
  assert.ok(end >= 0, `missing ${label}`);
  const start = body.lastIndexOf('<button', end);
  const tag = body.slice(start, body.indexOf('>', start) + 1);
  // Only hydration comments may sit between the tag and its label.
  assert.match(body.slice(start + tag.length, end), /^(<!--[^>]*-->)*$/);
  return tag;
}
const workflow = { nodes: [], extra: {} };
function generationProps(overrides: Record<string, unknown> = {}) {
  return {
    client: {},
    draft: { ...emptyDraft(), workflow, workflowName: 'Portrait', values: { steps: 20 } },
    bindings: [{ id: 'steps', type: 'INT', min: 1, max: 100, prominent: true }],
    rawWorkflow: '',
    controlTab: 'inputs',
    bindingErrors: [],
    catalog: {},
    connections: [],
    workflows: [
      { id: 'portrait', name: 'Portrait', workflow },
      { id: 'landscape', name: 'Landscape', workflow },
    ],
    workflowId: '',
    requestLimit: 4 * 1024 * 1024,
    dirty: true,
    busy: false,
    pending: false,
    driverState: 'connected',
    changeDraft() {},
    inspect: async () => {},
    fail() {},
    undo() {},
    prepareJob: async () => {},
    ...overrides,
  };
}

test('Generation enables Review only for a valid, unblocked draft', () => {
  const ready = html(Generation, generationProps());
  assert.doesNotMatch(tagBefore(ready, 'Review generation'), /disabled/);
  assert.match(ready, /Review effective settings and warnings before submitting/);
  const invalid = html(
    Generation,
    generationProps({ draft: { ...generationProps().draft, values: { steps: 0 } } }),
  );
  assert.match(tagBefore(invalid, 'Review generation'), /disabled/);
  assert.match(invalid, /Resolve 1 field issue\(s\)\./);
  for (const blocked of [{ pending: true }, { busy: true }, { draft: emptyDraft() }])
    assert.match(
      tagBefore(html(Generation, generationProps(blocked)), 'Review generation'),
      /disabled/,
    );
});

test('the workflow select shows the tracked workflow, not the first template it loaded', () => {
  const chosen = html(Generation, generationProps({ workflowId: 'landscape' }));
  assert.match(chosen, /<option value="landscape" selected="">Landscape<\/option>/);
  assert.doesNotMatch(chosen, /<option value="" selected="">/);
  // After Import JSON, Undo or Reuse the draft no longer matches a template.
  const imported = html(
    Generation,
    generationProps({ draft: { ...generationProps().draft, workflowName: 'imported.json' } }),
  );
  assert.match(imported, /<option value="" selected="">imported\.json<\/option>/);
  assert.doesNotMatch(imported, /<option value="landscape" selected/);
});

test('an unavailable driver shows its failure code in the footer', () => {
  const body = html(Generation, generationProps({ driverState: 'unavailable (database_error)' }));
  assert.match(body, /Driver unavailable \(database_error\)/);
});

test('the LoRA tab names an unread driver catalog and offers a refresh', () => {
  const failed = html(
    Generation,
    generationProps({ controlTab: 'loras', catalogError: 'Driver unreachable.' }),
  );
  assert.match(failed, /The driver catalog could not be read: Driver unreachable\./);
  assert.doesNotMatch(failed, /No matching LoRAs reported by the driver/);
  assert.doesNotMatch(tagBefore(failed, 'Refresh list'), /disabled/);
  const empty = html(Generation, generationProps({ controlTab: 'loras' }));
  assert.match(empty, /No matching LoRAs reported by the driver/);
  assert.match(empty, /Refresh list/);
});

test('the asset editor keeps its mode, target and paint across tab and view switches', () => {
  assetEditor.mode = 'mask';
  assetEditor.target = 'image';
  assetEditor.width = 640;
  assetEditor.height = 480;
  const props = generationProps({
    controlTab: 'assets',
    bindings: [{ id: 'image', type: 'COMBO', options: ['a.png'], node_id: '7', label: 'Image' }],
  });
  // Each render is a fresh mount, as a tab or view switch is.
  for (const body of [html(Generation, props), html(Generation, props)]) {
    assert.match(body, /<option value="mask" selected="">Paint mask<\/option>/);
    assert.match(body, /<option value="image" selected="">Image · 7<\/option>/);
    assert.match(body, /<canvas[^>]*width="640"[^>]*height="480"/);
    assert.match(body, /Paint stays while you switch tabs or views/);
  }
  assetEditor.mode = 'input';
  assetEditor.target = '';
});

test('the asset editor states the real input rule and needs an available target', () => {
  const props = {
    client: {},
    draft: { ...emptyDraft(), workflow },
    bindings: [{ id: 'image', type: 'COMBO', options: [], node_id: '7' }],
    changeDraft() {},
  };
  assetEditor.mode = 'mask';
  assetEditor.target = 'removed-binding';
  const stale = html(AssetEditor, props);
  assert.match(tagBefore(stale, 'Upload and bind asset'), /disabled/);
  assert.match(stale, /at most 2 MiB and 4096 pixels a side\. The companion\s+accepts any image within these limits/);
  assetEditor.target = 'image';
  assert.doesNotMatch(tagBefore(html(AssetEditor, props), 'Upload and bind asset'), /disabled/);
  assetEditor.mode = 'input';
  assert.match(tagBefore(html(AssetEditor, props), 'Upload and bind asset'), /disabled/);
  assetEditor.target = '';
});

test('a number field shows the stored value and never the default', () => {
  const binding = { id: 'steps', type: 'INT', default: 20, min: 1 };
  const cleared = html(Field, { binding, value: null, onchange() {} });
  assert.doesNotMatch(cleared, /value="20"/);
  assert.match(cleared, /Enter a finite number\./);
});

test('number inputs go through numberField instead of reading Number(value)', () => {
  for (const file of [
    'components/LoraStack.svelte',
    'components/Field.svelte',
    'components/OperationField.svelte',
    'components/WorkflowLayout.svelte',
  ]) {
    const source = readFileSync(new URL(`./${file}`, import.meta.url), 'utf8');
    assert.match(source, /use:numberField=/, file);
    assert.doesNotMatch(source, /Number\((e|event)\.currentTarget\.value\)/, file);
  }
});

function statusProps(overrides: Record<string, unknown> = {}) {
  return {
    messages: { notice: '', error: '', localError: '' },
    connection: {
      connected: true,
      state: 'connected',
      lastSeen: '',
      error: '',
      snapshot: { driver: {} },
    },
    submission: { pending: null, outcome: '', confirmedMissing: false, recoveryError: '' },
    autosave: { retained: false, storageError: '', conflict: false },
    draft: emptyDraft(),
    busy: false,
    ondismiss() {},
    onload() {},
    connect: async () => {},
    reconcilePending: async () => {},
    retryPending: async () => {},
    ...overrides,
  };
}

test('the pending banner shows why a submission outcome is unknown', () => {
  const body = html(
    StatusMessages,
    statusProps({
      submission: {
        pending: { key: 'k', preview_id: 'p' },
        outcome: 'Submission outcome unknown until reconciled. Do not create a new submission.',
        confirmedMissing: false,
        recoveryError: '',
      },
    }),
  );
  assert.match(
    body,
    /Submission outcome unknown until reconciled\. Do not create a new submission\./,
  );
  assert.match(body, /Its request identity is retained\./);
  assert.match(body, /Reconcile request/);
});

test('a retained draft, a full store and a tab conflict each get their own banner', () => {
  const retained = html(StatusMessages, statusProps({ autosave: { retained: true } }));
  assert.match(retained, /could not be restored\. Automatic saves are paused/);
  assert.match(retained, /Export saved record/);
  assert.match(retained, /Allow saving over it/);
  const full = html(
    StatusMessages,
    statusProps({ autosave: { storageError: 'Browser storage is full or blocked.' } }),
  );
  assert.match(full, /Browser storage is full or blocked\./);
  assert.match(full, /Export this draft/);
  assert.doesNotMatch(full, /Export saved record/);
  const conflict = html(StatusMessages, statusProps({ autosave: { conflict: true } }));
  assert.match(conflict, /Another tab saved a newer draft/);
  assert.match(conflict, /Load newer draft/);
});

test('the token form shows why the companion refused the old token', () => {
  const body = html(Connect, {
    token: 'old',
    busy: false,
    error:
      'The companion refused the owner token (HTTP 401). Enter the current token to reconnect.',
    connect: async () => {},
  });
  assert.match(body, /role="alert">The companion refused the owner token/);
  assert.doesNotMatch(body, /start its service/);
  assert.match(body, /dgxm studio open/);
});

// Run the shipped Operations apply handler, as libraryImport.test.ts does for LibraryTransfer.
const operations = readFileSync(new URL('./components/Operations.svelte', import.meta.url), 'utf8');
const script = operations.slice(operations.indexOf('>') + 1, operations.indexOf('</script>'));
const ast = ts.createSourceFile('Operations.ts', script, ts.ScriptTarget.Latest, true);
const applyNode = ast.statements.find(
  (node) => ts.isFunctionDeclaration(node) && node.name?.text === 'apply',
);
assert.ok(applyNode);
const applyCode = ts.transpile(applyNode.getText(ast), { target: ts.ScriptTarget.ESNext });
const runApply = new Function(
  'reviewed',
  'checked',
  `'use strict';
  return (async () => {
    let recoveryBlocked = false, preview = reviewed, busy = false, confirmed = checked,
      error = '', pending = false, requestKey = 'kept-key';
    const sent = [], reviewStorageKey = 'review', sessionStorage = { removeItem() {} };
    const rememberReview = () => {}, onrefresh = () => {}, requireRecordIdentity = (value) => value;
    class ApiError extends Error {}
    const client = { request: async (path, body) => { sent.push({ path, body }); return { id: 'applied' }; } };
    ${applyCode}
    await apply();
    return { sent, error };
  })();`,
);

test('Operations sends the confirmation checkbox state, not a literal true', async () => {
  const plan = { id: 'plan', digest: 'digest', requires_confirmation: true };
  assert.deepEqual((await runApply(plan, false)).sent, []);
  const confirmed = await runApply(plan, true);
  assert.deepEqual(confirmed.sent, [
    {
      path: '/operations/plan/apply',
      body: { digest: 'digest', confirmed: true, idempotency_key: 'kept-key' },
    },
  ]);
  // A plan without a confirmation step sends false, so the companion's gate decides.
  const readOnly = await runApply({ ...plan, requires_confirmation: false }, false);
  assert.equal(readOnly.sent[0].body.confirmed, false);
});

const CONFIG = '/home/user/.config/dgx-monarch/cluster.toml';
const ORIGIN = 'http://127.0.0.1:8191';
/** The panel for a companion whose driver origin refuses connections, with this start command. */
function panel(
  start: unknown,
  operations: unknown[] = [],
  connection: Record<string, unknown> = {},
) {
  const body = html(DriverNotRunning, {
    connection: {
      connected: true,
      driverNotRunning: true,
      capabilities: { driver_origin: ORIGIN, driver_start: start, operations },
      snapshot: { driver: { state: 'unavailable', code: 'driver_not_running', since: 5 } },
      ...connection,
    },
    connect: async () => {},
  });
  return body.replace(/\s+/g, ' ');
}
const LEAD = `Studio is connected, but nothing accepts a connection at ${ORIGIN}.`;
const ELSEWHERE =
  'If ComfyUI already runs at another address, set driver in the [studio] table of cluster.toml to that address and restart Studio. These records belong to http://127.0.0.1:8191; a new address needs fresh records, as the Studio guide explains under Configure Studio in cluster.toml.';
const LAUNCHER =
  'bash /home/user/ComfyUI/custom_nodes/dgx-monarch/scripts/comfy-driver.sh --comfy-dir /home/user/ComfyUI --python /home/user/monarch-env/bin/python --listen 127.0.0.1 --port 8191 --no-browser';

test('the driver panel shows the start command and names the setting it came from', () => {
  const variants: [unknown, string, string][] = [
    [
      {
        command: 'systemctl --user start comfyui-driver.service',
        source: 'driver_service',
        config: CONFIG,
      },
      'systemctl --user start comfyui-driver.service',
      `From the driver_service setting in ${CONFIG}. If it fails, read: journalctl --user -u comfyui-driver.service -n 50`,
    ],
    [
      {
        command: 'bash /home/user/bin/start-driver.sh --no-browser',
        source: 'driver_command',
        config: CONFIG,
      },
      'bash /home/user/bin/start-driver.sh --no-browser',
      `From the driver_command setting in ${CONFIG}. Studio reads it when it starts, so restart Studio after an edit, and edit cluster.toml when no render runs.`,
    ],
    [
      { command: LAUNCHER, source: 'launcher', config: null },
      LAUNCHER,
      'The repository launcher for this driver origin, with the Python that runs Studio; run it in a terminal you keep open. If you start ComfyUI another way, set driver_command in the [studio] table of cluster.toml.',
    ],
  ];
  for (const [start, command, source] of variants) {
    const body = panel(start);
    assert.match(
      body,
      /<section class="notice driver-setup" role="status" aria-live="polite"><h2>Driver not running<\/h2>/,
    );
    for (const text of [
      LEAD,
      ELSEWHERE,
      'Start it on the Studio host:',
      `<code>${command}</code><button>Copy command</button>`,
      `<p class="muted">${source}</p>`,
    ])
      assert.ok(body.includes(text), text);
    assert.ok(
      body.includes(
        'Studio checks every few seconds and connects by itself when the driver answers.',
      ),
    );
    assert.doesNotMatch(body, /Start your ComfyUI driver|start it from Operations/);
  }
});

test('the driver panel without a command says where the driver must listen', () => {
  const body = panel({ command: null, source: null, config: null });
  for (const text of [
    LEAD,
    ELSEWHERE,
    `Start your ComfyUI driver so it listens at ${ORIGIN}. To show your own start command here, set driver_command in the [studio] table of cluster.toml.`,
  ])
    assert.ok(body.includes(text), text);
  assert.doesNotMatch(body, /Copy command|Start it on the Studio host/);
});

test('the driver panel offers Operations only when driver.start is available', () => {
  const service = {
    command: 'systemctl --user start comfyui-driver.service',
    source: 'driver_service',
    config: CONFIG,
  };
  const available = panel(service, [{ id: 'driver.start', available: true }]);
  assert.ok(available.includes('You can also start it from Operations.'));
  assert.ok(available.includes(ELSEWHERE));
  const refused = panel(service, [
    { id: 'driver.start', available: false, reason: 'Restart the companion.' },
  ]);
  assert.doesNotMatch(refused, /start it from Operations/);
});

test('the driver panel shows a configured command as text', () => {
  const body = panel({ command: 'echo <b>driver</b>', source: 'driver_command', config: CONFIG });
  assert.ok(body.includes('<code>echo &lt;b>driver&lt;/b></code>'));
  assert.doesNotMatch(body, /<b>driver/);
});

test('no driver panel while the driver is connected or the tab is signed out', () => {
  const start = { command: null, source: null, config: null };
  const connected = panel(start, [], {
    driverNotRunning: false,
    snapshot: { driver: { state: 'connected' } },
  });
  assert.doesNotMatch(connected, /driver-setup|Driver not running/);
  assert.doesNotMatch(panel(start, [], { connected: false }), /driver-setup|Driver not running/);
  // StatusMessages mounts the panel, which shows only for a driver that is not running.
  assert.doesNotMatch(html(StatusMessages, statusProps()), /driver-setup/);
  const waiting = statusProps({
    connection: {
      ...statusProps().connection,
      driverNotRunning: true,
      capabilities: { driver_origin: ORIGIN },
    },
  });
  assert.match(html(StatusMessages, waiting), /<h2>Driver not running<\/h2>/);
});

test('Generation keeps Review closed while the driver is not running', () => {
  const body = html(Generation, generationProps({ driverState: 'not running' }));
  assert.match(tagBefore(body, 'Review generation'), /disabled/);
  assert.match(
    body,
    /<small class="warning">\s*Start the driver first; see Driver not running above\.<\/small>/,
  );
  assert.match(body, />Driver not running<\/span>/);
  assert.doesNotMatch(body, /Review effective settings and warnings before submitting/);
  const ready = html(Generation, generationProps());
  assert.doesNotMatch(ready, /Start the driver first/);
});
