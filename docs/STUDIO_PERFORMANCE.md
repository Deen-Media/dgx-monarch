# Studio measurement record

Date: 2026-10-19. Status: partial engineering measurements; performance
acceptance remains **NOT RUN**.

## Packaged asset sizes

These are measured file sizes of the current compiled Svelte/Vite bundle.
Gzip uses Python gzip with timestamp zero. They measure transfer size only,
not JavaScript execution, browser responsiveness or Spark overhead.

| Asset | Bytes | Gzip bytes |
|---|---:|---:|
| `index-BVIZXrRA.css` | 27,287 | 6,855 |
| `index-eaqzXxUW.js` | 193,936 | 67,491 |

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
  --output /tmp/dgxm-observer-cpu.json --iterations 10000 --repeats 7
```

The output must be a new JSON file under `/tmp`; the harness refuses to replace
an existing file. It records raw repeat samples, Python/platform/CPU affinity,
load averages, the harness hash, the uncached package source manifest, and hashes
of current tracked and nonignored untracked file bytes. Git HEAD is not used as
the source identity. A runtime package change during measurement fails the run;
other concurrent workspace changes are recorded with both tree hashes.

Local CPU measurement at `2026-10-19T14:22:57Z`, Python 3.12.3, Linux aarch64,
20 CPUs in the process affinity set, seven repeats:

| Operation | Median wall time/call | Repeat minimum to maximum |
|---|---:|---:|
| Capture, observer disabled | 107.6 ns | 107.5 to 108.3 ns |
| Publish, observer disabled | 106.1 ns | 105.8 to 107.0 ns |
| Capture, enabled with synthetic context | 764.9 ns | 761.7 to 767.5 ns |
| Capture, enabled with ComfyUI unavailable | 6.306 us | 6.289 to 6.331 us |
| Publish enabled, small synthetic event | 4.379 us | 4.374 to 4.515 us |
| Replay first 128 retained events | 450.7 us | 450.3 to 451.2 us |

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

The local raw result is `/tmp/dgxm-observer-delivery-measurement.json` (ephemeral, not a
packaged artifact). Its runtime package manifest was unchanged throughout:
`d4df964b3450467791e1120ffb886f30ceecf9858fd734fa3722fe2c3c1e7ba0`.
The harness SHA-256 was
`461d5215dcedaa28ca3d34f5ad7d1159998fec457114b52f2e288ae357e6e250`.
Both the runtime package manifest and whole-tree hash remained unchanged
during this run; the JSON records both snapshots. These measurements supplement correctness tests and do
not establish Spark generation overhead, native return performance, browser
responsiveness or release regression budgets.

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
viewer. These checks do not establish video, accessibility or long-session
performance acceptance.

A release report must record exact source, driver, model, workflow, topology,
client hardware/browser and baseline variance. Measure wall time, host CPU/GPU,
resident memory, network bytes, client interaction latency and retained memory
for every row. Set regression thresholds from repeatable baseline variance
before claiming acceptance. No runtime-overhead threshold has been fabricated
from bundle size or a mocked job.
