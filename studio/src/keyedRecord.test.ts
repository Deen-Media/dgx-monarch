import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
import { compileModule } from 'svelte/compiler';

// Compile the shipped rune module and a small rune harness with the client runtime, so the test
// drives real Svelte reactivity: fresh snapshot objects, deriveds and effects.
const svelteClient = new URL('../node_modules/svelte/src/index-client.js', import.meta.url).href;
function runeModule(source: string, filename: string, imports: Record<string, string> = {}) {
  const plain = ts.transpile(source, {
    target: ts.ScriptTarget.ESNext,
    module: ts.ModuleKind.ESNext,
    verbatimModuleSyntax: true,
  });
  const compiled = compileModule(plain, { filename, generate: 'client' }).js.code;
  const resolved = compiled.replace(/from ['"]([^'"]+)['"]/g, (_match, specifier: string) => {
    const url =
      imports[specifier] ??
      (specifier === 'svelte' ? svelteClient : import.meta.resolve(specifier));
    return `from ${JSON.stringify(url)}`;
  });
  return `data:text/javascript;base64,${Buffer.from(resolved).toString('base64')}`;
}
const recordModule = runeModule(
  await readFile(new URL('./keyedRecord.svelte.ts', import.meta.url), 'utf8'),
  'keyedRecord.svelte.js',
);
const harnessModule = runeModule(
  `import { keyedRecord } from './keyedRecord.svelte';
import { isActive } from './queueRows';
export function mount(read, initial, options) {
  let summary = $state(initial);
  let record;
  const stop = $effect.root(() => {
    record = keyedRecord(read, () => ({
      id: summary.id,
      revision: summary.state + ':' + (summary.updated_at ?? ''),
      live: isActive(summary),
    }), options);
  });
  return { record, stop, set: (next) => { summary = next; } };
}`,
  'harness.svelte.js',
  {
    './keyedRecord.svelte': recordModule,
    './queueRows': new URL('./queueRows.ts', import.meta.url).href,
  },
);
const { KeyedRecord } = await import(recordModule);
const { mount } = await import(harnessModule);
const { flushSync } = await import(svelteClient);
const settle = async () => {
  flushSync();
  await new Promise((resolve) => setImmediate(resolve));
  flushSync();
};
function counter() {
  const calls: string[] = [];
  return {
    calls,
    read: async (id: string) => {
      calls.push(id);
      return { id, fetch: calls.length };
    },
  };
}

test('identical snapshot refreshes fetch the full record once per id and keep it loaded', async () => {
  const { calls, read } = counter();
  const row = { id: 'job-a', state: 'succeeded', updated_at: 10 };
  const view = mount(read, { ...row });
  await settle();
  assert.deepEqual(calls, ['job-a']);
  const loaded = view.record.value;
  for (let poll = 0; poll < 8; poll++) {
    view.set({ ...row });
    await settle();
  }
  assert.deepEqual(calls, ['job-a']);
  assert.equal(view.record.value, loaded);
  view.set({ id: 'job-b', state: 'succeeded', updated_at: 10 });
  await settle();
  assert.deepEqual(calls, ['job-a', 'job-b']);
  assert.equal(view.record.value.id, 'job-b');
  view.stop();
});

test('a changed revision reloads in place without clearing the loaded record', async () => {
  let finish!: (value: { id: string; fetch: number }) => void;
  let reads = 0;
  const view = mount(
    (id: string) =>
      ++reads === 1
        ? Promise.resolve({ id, fetch: 1 })
        : new Promise((resolve) => {
            finish = resolve;
          }),
    { id: 'job-a', state: 'failed', updated_at: 1 },
  );
  await settle();
  const first = view.record.value;
  view.set({ id: 'job-a', state: 'cancelled', updated_at: 2 });
  await settle();
  assert.equal(reads, 2);
  assert.equal(view.record.loading, true);
  assert.equal(view.record.value, first, 'the record stays mounted while it reloads');
  finish({ id: 'job-a', fetch: 2 });
  await settle();
  assert.equal(view.record.value.fetch, 2);
  assert.equal(view.record.loading, false);
  view.stop();
});

test('running jobs reload at most once per throttle period and always load the last change', async () => {
  let clock = 0;
  const timers: Array<{ run: () => void; at: number }> = [];
  const { calls, read } = counter();
  const view = mount(
    read,
    { id: 'job-a', state: 'running', updated_at: 0 },
    {
      throttleMs: 5000,
      now: () => clock,
      schedule: (run: () => void, delay: number) => {
        const timer = { run, at: clock + delay };
        timers.push(timer);
        return timer;
      },
      cancel: (timer: unknown) => timers.splice(timers.indexOf(timer as never), 1),
    },
  );
  await settle();
  for (let event = 1; event <= 10; event++) {
    clock = event * 300;
    view.set({ id: 'job-a', state: 'running', updated_at: event });
    await settle();
  }
  assert.equal(calls.length, 1, 'progress events inside the period do not refetch');
  assert.equal(timers.length, 1);
  assert.equal(timers[0].at, 5000);
  clock = 5000;
  timers.shift()!.run();
  await settle();
  assert.equal(calls.length, 2);
  view.set({ id: 'job-a', state: 'succeeded', updated_at: 11 });
  await settle();
  assert.equal(calls.length, 3, 'leaving a live state reloads immediately');
  view.stop();
});

test('a reply for an earlier id never replaces the current record', async () => {
  const pending = new Map<string, (value: { id: string }) => void>();
  const record = new KeyedRecord(
    (id: string) =>
      new Promise((resolve) => {
        pending.set(id, resolve);
      }),
  );
  record.follow('first');
  record.follow('second');
  pending.get('second')!({ id: 'second' });
  pending.get('first')!({ id: 'first' });
  await settle();
  assert.equal(record.value.id, 'second');
  record.refresh();
  record.dispose();
  pending.get('second')!({ id: 'late' });
  await settle();
  assert.equal(record.value.id, 'second', 'a disposed owner ignores late replies');
});

test('provenance, receipts and comparisons load through the keyed record, not a per-poll effect', async () => {
  const sources = Object.fromEntries(
    await Promise.all(
      ['JobProvenance', 'OperationReceipt', 'ComparisonDetails'].map(async (name) => [
        name,
        await readFile(new URL(`./components/${name}.svelte`, import.meta.url), 'utf8'),
      ]),
    ),
  );
  for (const [name, source] of Object.entries(sources)) {
    assert.match(source, /import \{ keyedRecord \} from '\.\.\/keyedRecord\.svelte';/, name);
    assert.match(source, /= keyedRecord<[^>]+>\(/, name);
    assert.doesNotMatch(
      source,
      /\$effect\(/,
      `${name} has no effect that clears the record per poll`,
    );
  }
  // RuntimeEvidence keeps its pages and choices only while it stays mounted across reloads.
  assert.match(
    sources.JobProvenance,
    /\{#if record\.value\}[^{]*\{@const job = record\.value\}[\s\S]*<RuntimeEvidence \{client\} \{job\} \/>\s*\{\/if\}/,
  );
  assert.match(sources.JobProvenance, /\{#if record\.loading && !record\.value\}/);
});
