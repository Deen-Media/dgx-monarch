import test from 'node:test';
import assert from 'node:assert/strict';
import { ApiError, type StudioClient } from './api.ts';
import type { RecordData, StoredRecord } from './contracts.ts';
import {
  persistRecord,
  putRecord,
  RECORD_SAVED_EARLIER,
  RecordSession,
} from './recordPersistence.ts';
import { componentFunctions } from './renderHarness.js';

/** A companion that commits a PUT and can lose the reply, as a client timeout does. */
function companion() {
  const rows = new Map<string, StoredRecord>();
  let loseReply = false;
  const client = {
    async request(path: string, body?: { revision: number; data: RecordData }, method?: string) {
      if (method === 'PUT') {
        const id = decodeURIComponent(path.split('/').at(-1)!);
        const old = rows.get(id);
        if ((old?.revision ?? 0) !== body!.revision)
          throw new ApiError(
            409,
            'A newer version exists; reload before saving.',
            'revision_conflict',
          );
        const value = {
          id,
          revision: body!.revision + 1,
          data: { ...body!.data, grid_snapshot: { cells: [] } },
        };
        rows.set(id, value);
        if (loseReply) {
          loseReply = false;
          throw new Error('The operation was aborted.');
        }
        return value;
      }
      return { records: [...rows.values()].reverse(), next_offset: null };
    },
  } as unknown as StudioClient;
  return {
    rows,
    client,
    loseNextReply: () => {
      loseReply = true;
    },
  };
}

test('a retry after a lost reply reads the committed record back instead of duplicating it', async () => {
  const server = companion();
  const session = new RecordSession();
  const data = { name: 'Alpha', grid_id: 'grid-1', project_ids: ['p'] };
  server.loseNextReply();
  await assert.rejects(
    putRecord(server.client, 'experiments', data, session.id('grid-1')),
    /aborted/,
  );
  const notices: string[] = [];
  const saved = await persistRecord(
    server.client,
    async () => {},
    (text) => notices.push(text),
    'experiments',
    data,
    session.id('grid-1'),
  );
  assert.equal(server.rows.size, 1);
  assert.equal(saved.data.name, 'Alpha');
  assert.deepEqual(notices, ['experiments record was already saved by an earlier attempt.']);
});

test('a refresh failure after a committed save is a notice, not a failed save', async () => {
  const server = companion();
  const notices: string[] = [];
  const saved = await persistRecord(
    server.client,
    async () => {
      throw new Error('This catalog exceeds the 8 MiB browser limit.');
    },
    (text) => notices.push(text),
    'presets',
    { name: 'kept' },
    'kept-id',
  );
  assert.equal(saved.id, 'kept-id');
  assert.match(notices[0], /^presets record saved\. The view did not refresh: .*8 MiB/);
});

test('a kept identity saved with other content, or any other refusal, is not treated as saved', async () => {
  const server = companion();
  const session = new RecordSession();
  const id = session.id('presets');
  server.loseNextReply();
  await assert.rejects(putRecord(server.client, 'presets', { name: 'first' }, id));
  await assert.rejects(
    putRecord(server.client, 'presets', { name: 'renamed' }, session.id('presets')),
    (cause: unknown) => {
      session.failed(cause);
      return cause instanceof ApiError && cause.code === RECORD_SAVED_EARLIER;
    },
  );
  assert.notEqual(session.id('presets'), id, 'the used identity starts a new form session');
  const limited = {
    request: async () => {
      throw new ApiError(409, 'Export or remove records before adding more.', 'record_limit');
    },
  } as unknown as StudioClient;
  await assert.rejects(putRecord(limited, 'presets', { name: 'x' }, 'new'), /Export or remove/);
  const edited = companion();
  edited.rows.set('r', { id: 'r', revision: 2, data: { name: 'theirs' } });
  await assert.rejects(
    putRecord(edited.client, 'presets', { name: 'theirs' }, 'r', 1),
    /newer version/,
  );
});

test('a form session keeps its identity until a confirmed save', () => {
  let minted = 0;
  const session = new RecordSession(() => `id-${++minted}`);
  assert.equal(session.id('grid-a'), 'id-1');
  session.failed(new Error('timeout'));
  assert.equal(session.id('grid-a'), 'id-1', 'an unknown outcome keeps the identity');
  assert.equal(session.id('grid-b'), 'id-2', 'naming another grid starts a new session');
  session.settle();
  assert.equal(session.id('grid-b'), 'id-3');
});

// Run the shipped ExperimentLibrary save handler against the companion fake.
const saveCode = await componentFunctions('./components/ExperimentLibrary.svelte', ['save']);
const libraryHarness = new Function(
  'client',
  'putRecord',
  'RecordSession',
  `'use strict';
  let gridId = 'grid-1', name = 'Alpha', projectId = 'p', selected = '', notice = '', refreshed = 0;
  const session = new RecordSession();
  async function refresh() { refreshed++; }
  ${saveCode}
  return { save, state: () => ({ name, selected, notice, refreshed }), rename: (value) => { name = value; } };`,
);

test('ExperimentLibrary keeps one record identity across a retried save', async () => {
  const server = companion();
  const library = libraryHarness(server.client, putRecord, RecordSession);
  server.loseNextReply();
  await assert.rejects(library.save(), /aborted/);
  assert.equal(library.state().name, 'Alpha', 'an unknown outcome keeps the form');
  await library.save();
  assert.equal(server.rows.size, 1);
  assert.equal(library.state().selected, [...server.rows.keys()][0]);
  assert.match(library.state().notice, /already saved this grid/);
  library.rename('Beta');
  await library.save();
  assert.equal(server.rows.size, 2, 'a confirmed save starts a new form session');
});
