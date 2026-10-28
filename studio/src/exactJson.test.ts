import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { Artifact, Job } from './contracts.ts';
import { parseExactJSON, readExactJSON, roundedNumber, writeExactJSON } from './exactJson.ts';
import { artifactDraft } from './artifactDraft.ts';
import { recordedSeeds, recordedSettings } from './recordedSettings.ts';
import { componentFunctions } from './renderHarness.js';

const MAX_SEED = '18446744073709551615';
// A record as the companion writes it: Python's json keeps every digit of each whole number.
const RECORD =
  '{"id":"job-1","state":"succeeded","created_at":1,"submission":{' +
  `"prompt":{"8":{"class_type":"KSampler","inputs":{"seed":${MAX_SEED},"steps":20,"model":["4",0]}},` +
  '"9":{"class_type":"RandomNoise","inputs":{"noise_seed":9007199254740993}},' +
  `"10":{"class_type":"Custom","inputs":{"big":-${MAX_SEED},"text":"a \\"quoted\\" 18446744073709551615"}}},` +
  `"effective":{"seeds":{"8.seed":${MAX_SEED},"9.noise_seed":9007199254740993}},` +
  `"workflow":{"nodes":[{"id":8,"type":"KSampler","widgets_values":[${MAX_SEED},"fixed",20]}]}}}`;

test('the exact parser reads JSON as JSON.parse does, with large whole numbers as BigInt', () => {
  const plain =
    '{"a":[1,-0,1.5,-1e-7,1e15,9007199254740991,true,false,null,"x\\u00e9\\n",{},[]],' +
    '"__proto__":{"b":2},"dup":1,"dup":2,"20":"x","1":"y","nested":[[{"c":[3]}]]}';
  assert.deepEqual(parseExactJSON(plain), JSON.parse(plain));
  const parsed = parseExactJSON(plain) as Record<string, unknown>;
  assert.equal(Object.getPrototypeOf(parsed), Object.prototype, 'a "__proto__" key stays a key');
  assert.deepEqual(Object.keys(parsed), Object.keys(JSON.parse(plain)));
  const exact = parseExactJSON(RECORD) as Job;
  assert.equal(exact.submission.prompt['8'].inputs.seed, 18446744073709551615n);
  assert.equal(exact.submission.prompt['9'].inputs.noise_seed, 9007199254740993n);
  assert.equal(exact.submission.prompt['10'].inputs.big, -18446744073709551615n);
  assert.equal(exact.submission.prompt['10'].inputs.text, `a "quoted" ${MAX_SEED}`);
  assert.equal(parseExactJSON('9007199254740992'), 9007199254740992n);
  // A whole float this large keeps the exact value of its double, so none reads rounded.
  assert.equal(parseExactJSON('1e21'), 1000000000000000000000n);
  assert.equal(parseExactJSON('-1.8446744073709552e19'), -18446744073709551616n);
  assert.equal(parseExactJSON(' 42 '), 42);
  assert.throws(() => parseExactJSON('{"a":1,}'), SyntaxError);
  assert.throws(() => parseExactJSON(''), SyntaxError);
});

// Texts whose every number a double holds exactly: the exact parse must equal JSON.parse.
const SAFE = [
  String.raw`"\" \\ \/ \b \f \n \r \t \u0000 \u001f \u00e9 \u00E9 \u2028 \u2029 \u007f"`,
  String.raw`["\ud83d\ude00", "\uD83D\uDE00", "\ud800", "\udfff", "é 中 😀", "", "{}[],:\"0 1"]`,
  '[0, -0, 1, -1, 10, -10, 0.5, -0.5, 3.14159, -273.15, 123.456e-2, 0e0, -0.0]',
  '[1e3, 1E3, 1e+3, 1E-3, -1e-7, 2.5e+10, -2.5E+10, 5e-324, -5e-324, 1e-400, 1e400, -1e400]',
  '[9007199254740991, -9007199254740991, 4503599627370496.5, 1.5e15, 8.99e15]',
  '[[], {}, [[]], [{}], {"a": {}}, {"a": []}, [[[[[1]]]]], {"a": {"b": {"c": {"d": null}}}}]',
  '[true, false, null, [true, [false, [null]]]]',
  '{"": 1, "a b": 2, "\\u0041": 3, "20": "x", "1": "y", "dup": 1, "dup": 2, "__proto__": {"x": 1}}',
  ' \t\n\r{ "a" \t:\n [ 1 ,\r 2 , { } , [ ] ] , "b" : { "c" : "d" } , "e" : -1.5e-3 }\n\t ',
  '"plain"',
  '-12.5',
  'null',
];

test('the exact parser and writer match JSON.parse and JSON.stringify on every safe text', () => {
  for (const text of SAFE) {
    const plain = JSON.parse(text);
    assert.deepEqual(parseExactJSON(text), plain, text);
    assert.equal(JSON.stringify(parseExactJSON(text)), JSON.stringify(plain), `key order: ${text}`);
    assert.equal(writeExactJSON(parseExactJSON(text)), JSON.stringify(plain, null, 2), text);
    assert.deepEqual(readExactJSON(text), plain, text);
  }
});

// Whole numbers beyond 2**53 - 1, which JSON.parse rounds, with the digits the writer must print.
const LARGE: [string, bigint, string][] = [
  ['9007199254740992', 9007199254740992n, '9007199254740992'],
  ['-9007199254740992', -9007199254740992n, '-9007199254740992'],
  ['9007199254740993', 9007199254740993n, '9007199254740993'],
  ['18446744073709551615', 18446744073709551615n, '18446744073709551615'],
  ['-18446744073709551615', -18446744073709551615n, '-18446744073709551615'],
  [
    '123456789012345678901234567890',
    123456789012345678901234567890n,
    '123456789012345678901234567890',
  ],
  ['1e21', 10n ** 21n, '1000000000000000000000'],
  ['-1.8446744073709552E+19', -(2n ** 64n), '-18446744073709551616'],
];

test('large whole numbers read as BigInt and write back as their digits, at any depth', () => {
  for (const [text, value, digits] of LARGE) {
    assert.equal(parseExactJSON(text), value, text);
    assert.equal(writeExactJSON(parseExactJSON(text)), digits, text);
    const nested = ` { "a" : [ 1.5 , { "b" : ${text} } , [ ${text} ] ] , "c" : ${text} } `;
    const tree = parseExactJSON(nested);
    assert.deepEqual(tree, { a: [1.5, { b: value }, [value]], c: value }, nested);
    const written = writeExactJSON(tree);
    assert.equal(written.split(digits).length, 4, `three copies of ${digits}`);
    assert.deepEqual(parseExactJSON(written), tree, 'the written text reads back the same');
    assert.doesNotThrow(() => JSON.parse(written), 'still JSON');
  }
});

// A leaf nested `depth` lists deep, and a way to find it again without recursion.
const nest = (depth: number, leaf: string) => '['.repeat(depth) + leaf + ']'.repeat(depth);
function leafOf(tree: unknown) {
  let depth = 0;
  for (; Array.isArray(tree); depth++) tree = tree[0];
  return { depth, leaf: tree };
}

test('readExactJSON reads whatever JSON.parse reads, however deep, and refuses the same text', () => {
  for (const depth of [1, 1000, 3000, 12000, 30000]) {
    const { depth: found, leaf } = leafOf(readExactJSON(nest(depth, MAX_SEED)));
    assert.equal(found, depth);
    // The exact walk keeps the digits; past its call stack, which shrinks while code runs
    // unoptimized, the read is JSON.parse's.
    if (depth <= 1000) assert.equal(leaf, 18446744073709551615n, `depth ${depth}`);
    else assert.ok(leaf === 18446744073709551615n || leaf === Number(MAX_SEED), `depth ${depth}`);
  }
  assert.throws(() => parseExactJSON(nest(30000, '7')), RangeError, 'deeper than the walk');
  for (const bad of ['{"a":1,}', '', '[1 2]', '{"a"}', "{'a':1}", '01', 'NaN', '"\\x"'])
    assert.throws(() => readExactJSON(bad), SyntaxError, bad);
});

test('writeExactJSON writes whatever JSON.stringify writes, however deep, and no rounded value', () => {
  let fallback = 0;
  for (let depth = 1000; depth <= 6000; depth += 500) {
    const plain = JSON.parse(nest(depth, '7'));
    let native: string | null = null;
    try {
      native = JSON.stringify(plain, null, 2);
    } catch {
      continue; // Too deep for JSON.stringify as well, so nothing wrote this before either.
    }
    assert.equal(writeExactJSON(plain), native, `depth ${depth}`);
    // A BigInt leaf shows which writer ran: past the walk's stack, the native one cannot write it.
    const leaf = (value: unknown) => {
      const tree = JSON.parse(nest(depth, '0'));
      let node = tree;
      for (let level = 1; level < depth; level++) node = node[0];
      node[0] = value;
      return tree;
    };
    try {
      assert.match(writeExactJSON(leaf(18446744073709551615n)), /\n {2,}18446744073709551615\n/);
    } catch (error) {
      assert.match(
        String(error),
        /0 holds 18446744073709551615 too deep in the record to write exactly/,
      );
      fallback ||= depth;
    }
  }
  // Node 24 runs out of stack in the walk before JSON.stringify does; the native writer then
  // still refuses a rounded number.
  if (fallback)
    assert.throws(
      () => writeExactJSON(JSON.parse(nest(fallback, MAX_SEED))),
      /^Error: 0 is a whole number above 9007199254740991 that this browser holds rounded, so Studio wrote nothing\.$/,
    );
});

test('the exact writer writes JSON.stringify text, with each BigInt as its digits', () => {
  const plain = JSON.parse(
    '{"a":[1,-0,1.5,-1e-7,1e15,true,null,"x\\u2028\\"q\\"",{},[]],"20":"x","1":{"b":[[]]}}',
  );
  assert.equal(writeExactJSON(plain), JSON.stringify(plain, null, 2));
  assert.equal(writeExactJSON([]), '[]');
  assert.equal(
    writeExactJSON({ a: undefined, b: [undefined], c: 1n }),
    '{\n  "b": [\n    null\n  ],\n  "c": 1\n}',
  );
  const exact = parseExactJSON(RECORD);
  const text = writeExactJSON(exact);
  assert.deepEqual(parseExactJSON(text), exact, 'the text reads back to the same values');
  assert.match(text, /"seed": 18446744073709551615,/);
  assert.equal(roundedNumber(2 ** 64), true);
  assert.equal(roundedNumber(Number.MAX_SAFE_INTEGER), false);
  // A number the browser already rounded is never written.
  assert.throws(
    () => writeExactJSON(JSON.parse(RECORD)),
    /^Error: submission\.prompt\.8\.inputs\.seed is a whole number above 9007199254740991 that this browser holds rounded, so Studio wrote nothing\.$/,
  );
});

// Run the shipped Gallery copy and export handlers against fakes.
const galleryCode = await componentFunctions('./views/Gallery.svelte', [
  'copy',
  'exportProvenance',
]);
function gallery(record: unknown) {
  return new Function(
    'recordedSeeds',
    'recordedSettings',
    'writeExactJSON',
    'record',
    `'use strict';
    const copied = [], notices = [], exported = [], fetched = [];
    const client = {};
    const navigator = { clipboard: { writeText: async (text) => copied.push(text) } };
    const setNotice = (text) => notices.push(text);
    const exactJob = async (_client, id) => (fetched.push(id), record);
    const jobSummary = async () => { throw new Error('no summary read'); };
    const exportText = (text, name) => exported.push([name, text]);
    ${galleryCode}
    return { copy, exportProvenance, state: () => ({ copied, notices, exported, fetched }) };`,
  )(recordedSeeds, recordedSettings, writeExactJSON, record) as {
    copy: (entry: unknown, kind: string) => Promise<void>;
    exportProvenance: (entry: unknown) => Promise<void>;
    state: () => {
      copied: string[];
      notices: string[];
      exported: [string, string][];
      fetched: string[];
    };
  };
}
const output = { id: 'out', job_id: 'job-1', kind: 'image', filename: 'out.png', created_at: 2 };

test('Copy named inputs and Copy API prompt keep every digit of every whole number', async () => {
  const view = gallery(parseExactJSON(RECORD));
  await view.copy({ artifact: output }, 'inputs');
  const inputs = view.state().copied[0];
  const expected = JSON.stringify(
    {
      '8': { seed: 1, steps: 20, model: ['4', 0] },
      '9': { noise_seed: 2 },
      '10': { big: 3, text: `a "quoted" ${MAX_SEED}` },
    },
    null,
    2,
  )
    .replace('"seed": 1', `"seed": ${MAX_SEED}`)
    .replace('"noise_seed": 2', '"noise_seed": 9007199254740993')
    .replace('"big": 3', `"big": -${MAX_SEED}`);
  assert.equal(inputs, expected);
  await view.copy({ artifact: output }, 'prompt');
  const prompt = view.state().copied[1];
  assert.deepEqual(parseExactJSON(prompt), (parseExactJSON(RECORD) as Job).submission.prompt);
  assert.deepEqual(view.state().notices, [
    'Copied named input values, recorded seeds included.',
    'Copied recorded API prompt.',
  ]);
  assert.deepEqual(view.state().fetched, ['job-1', 'job-1']);
});

test('Export provenance keeps every digit in the prompt, the seed list and the widget values', async () => {
  const view = gallery(parseExactJSON(RECORD));
  await view.exportProvenance({ artifact: output });
  const [[name, text]] = view.state().exported;
  assert.equal(name, 'out.json');
  const submission = (parseExactJSON(RECORD) as Job).submission;
  assert.deepEqual(parseExactJSON(text), { artifact: output, submission });
  assert.match(text, /"8\.seed": 18446744073709551615,\n {8}"9\.noise_seed": 9007199254740993\n/);
  assert.match(text, /"widgets_values": \[\n {12}18446744073709551615,\n {12}"fixed",/);
  assert.doesNotThrow(() => JSON.parse(text), 'the export is valid JSON');
  // Queue's provenance view shows and exports the same exact record.
  const queue = readFileSync(new URL('./components/JobProvenance.svelte', import.meta.url), 'utf8');
  assert.match(queue, /\(id\) => exactJob\(client, id\),/);
  assert.match(queue, /\{@const text = writeExactJSON\(job\)\}/);
  assert.match(queue, /exportText\(text, `\$\{job\.id\}\.json`\)\}>Export provenance</);
  assert.match(queue, /<pre>\{text\}<\/pre>/);
  // A record the browser rounded writes nothing and says which number.
  const rounded = gallery(JSON.parse(RECORD));
  await rounded.exportProvenance({ artifact: output });
  assert.deepEqual(rounded.state().exported, []);
  assert.deepEqual(rounded.state().notices, [
    'submission.prompt.8.inputs.seed is a whole number above 9007199254740991 that this browser holds rounded, so Studio wrote nothing.',
  ]);
});

const artifact = { id: 'out', job_id: 'job-1' } as Artifact;
const NOTE =
  'A draft field holds whole numbers exactly only up to 9007199254740991, so the draft is unchanged; Copy named inputs in the gallery keeps every digit.';

test('Reuse refuses a job with a large whole number, naming each input and its digits', () => {
  assert.throws(
    () => artifactDraft(artifact, parseExactJSON(RECORD) as Job),
    new Error(
      `Settings not loaded: 8.seed is ${MAX_SEED}, 9.noise_seed is 9007199254740993, 10.big is -${MAX_SEED}. ${NOTE}`,
    ),
  );
  // A record the browser parsed has no digits left; the inputs are still named.
  const above = 'is a whole number above 9007199254740991 that this browser holds rounded';
  assert.throws(
    () => artifactDraft(artifact, JSON.parse(RECORD) as Job, 'edit'),
    new Error(
      `Settings not loaded: 8.seed ${above}, 9.noise_seed ${above}, 10.big ${above}. ${NOTE}`,
    ),
  );
  // Without a prompt, the requested workflow loads; its own path names the number.
  const requested = parseExactJSON(RECORD) as Job;
  delete requested.submission.prompt;
  assert.throws(
    () => artifactDraft(artifact, requested),
    new Error(`Settings not loaded: workflow.nodes.0.widgets_values.0 is ${MAX_SEED}. ${NOTE}`),
  );
  // App reads the exact record for every reuse action.
  const app = readFileSync(new URL('./App.svelte', import.meta.url), 'utf8');
  assert.match(app, /const job = await exactJob\(api, artifact\.job_id\);/);
  assert.doesNotMatch(app, /fullJob/);
});
