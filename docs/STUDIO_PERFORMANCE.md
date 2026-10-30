# Studio measurement record

Asset sizes updated: 2026-10-30 (UTC). Observer timings updated: 2026-10-21
(UTC). Status: partial engineering measurements; performance acceptance remains
**NOT RUN**.

## Packaged asset sizes

These are measured file sizes of the current compiled Svelte/Vite bundle.
Gzip bytes are the output of `gzip -9 -c`, file name included. They measure
transfer size only, not JavaScript execution, browser responsiveness or Spark
overhead.

| Asset | Bytes | Gzip bytes |
|---|---:|---:|
| `index-R1VZxs78.js` | 324,175 | 113,985 |
| `index-Dzp3gAvP.css` | 40,456 | 9,432 |

## CPU observer microbenchmark

The reproducible harness in
[`studio/tests/measure_observer.py`](../studio/tests/measure_observer.py) exercises
the real `Journal` and `capture()` implementations without importing ComfyUI or
Torch, starting a service, or running a model. It measures disabled hooks,
unavailable-context capture, an explicitly synthetic stdlib context fixture,
event publication and replay. Timing excludes provenance hashing; retained
allocations are measured separately with `tracemalloc`.

```sh
python studio/tests/measure_observer.py \
  --output "$EVIDENCE/dgxm-observer-cpu.json" --iterations 10000 --repeats 7
```

`$EVIDENCE` is a durable directory outside the repository that only you can
read (mode 0700). Do not use a temporary directory that the host clears. The
harness refuses a path inside the repository, a directory that other accounts
can read or write, and an existing file, before it measures anything. It
prints the record's path and SHA-256. Cite the record by that SHA-256 and by
where it is kept; a public page names the store, not a private path. The record
holds raw repeat samples, Python/platform/CPU affinity, load averages, the
harness hash, the uncached package source manifest, and hashes of current
tracked and nonignored untracked file bytes. Git HEAD is not used as
the source identity. A runtime package change during measurement fails the run;
other concurrent workspace changes are recorded with both tree hashes.

Local CPU measurement at `2026-10-21T02:28:19Z`, Python 3.12.3, Linux aarch64,
20 CPUs in the process affinity set, seven repeats:

| Operation | Median wall time/call | Repeat minimum to maximum |
|---|---:|---:|
| Capture, observer disabled | 107.0 ns | 106.7 to 107.3 ns |
| Publish, observer disabled | 105.3 ns | 105.1 to 105.5 ns |
| Capture, enabled with synthetic context | 749.6 ns | 749.0 to 752.3 ns |
| Capture, enabled with ComfyUI unavailable | 6.389 us | 6.359 to 6.418 us |
| Publish enabled, small synthetic event | 4.408 us | 4.402 to 4.592 us |
| Replay first 128 retained events | 449.4 us | 449.3 to 449.7 us |

Each repeat used 10,000 calls, except replay which used 1,000. These include
Python call/loop overhead. The CPU governor and concurrent host work were not
controlled; the raw report retains load and repeat variance. Synthetic context
capture does not measure the real ComfyUI execution path or Worker fingerprint
collection.

The count-bound retention case published two waves of 16,384 events and retained
8,192 after each wave. Encoded payload was about 2.54 MB; traced retained Python
allocations were 14,100,640 and 14,101,012 bytes. The byte-bound case published
two waves of 2,048 events with 8,192-byte fixture payloads and retained 990 events
after each wave. Encoded payload remained below 8 MiB; traced allocations were
9,782,878 and 9,783,250 bytes. Both cases exposed eviction gaps and counted an
oversized-event omission. The second wave added only 372 traced bytes in each
case. Python object overhead exceeds encoded payload size; these are not RSS
measurements or a long-session database-growth test.

The raw record of this run is kept in the maintainer's private evidence store,
outside the repository and the packages. Its SHA-256 is
`41960ecff8e8a7793befa6a8cadafe00d17d08bb951d6cb9526a652438397102`.
The harness SHA-256 was
`8f0ae41d337a838d3af92f5b83c409e0ddd634921e4bd737c9a1ebd05df5cd99`.
The runtime package manifest was
`307615b607e6ae7e14ebfb34463c5a4cb84e3cd8ef1bc9d5562615bb3c414874`.
Both the runtime package manifest and the whole-tree hash stayed the same
during the run; the record holds both snapshots. These measurements supplement
correctness tests and do not establish Spark generation overhead, native return
performance, browser responsiveness or release regression budgets.

## Runtime acceptance

| Required comparison | Spark host | Browser client |
|---|---|---|
| Studio closed baseline | NOT RUN | NOT RUN |
| Studio open and idle | NOT RUN | NOT RUN |
| Active generation | NOT RUN | NOT RUN |
| Video viewing | NOT RUN | NOT RUN |
| Full telemetry and console | NOT RUN | NOT RUN |
| Long-session retention and growth | NOT RUN | NOT RUN |

The CPU-only HTTP fixture verifies contract behavior and artifact transport;
it is not a substitute for these measurements. The interface passed browser
checks against the synthetic fixture for workflow import, generation
review/submission, a two-cell grid, durable results, thumbnail gallery and image
viewer. Those checks ran on an earlier build of the bundle and recorded no
browser name, browser version or bundle hash. The
[browser procedure](../studio/tests/README.md#browser-checks) repeats them on
the shipped bundle. These checks do not establish video, accessibility or
long-session performance acceptance.

The [interface QA record](STUDIO_INTERFACE_QA.md) adds scoped responsive,
image geometry, video controls and reduced-motion checks. Its short disclosure
animation sample is interaction evidence, not a sustained performance budget.

A release report must record exact source, driver, model, workflow, topology,
client hardware/browser and baseline variance. Measure wall time, host CPU/GPU,
resident memory, network bytes, client interaction latency and retained memory
for every row. Set regression thresholds from repeatable baseline variance
before claiming acceptance. No runtime-overhead threshold has been fabricated
from bundle size or a mocked job.
