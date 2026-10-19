# Studio measurement record

Date: 2026-10-19. Status: partial engineering measurements; performance
acceptance remains **NOT RUN**.

## Packaged asset sizes

These are measured file sizes of the current compiled Svelte/Vite bundle.
Gzip uses Python gzip with timestamp zero. They measure transfer size only,
not JavaScript execution, browser responsiveness or Spark overhead.

| Asset | Bytes | Gzip bytes |
|---|---:|---:|
| `index-yUzH7Qw5.js` | 124,490 | 43,772 |
| `index-zdB5ZhgI.css` | 25,183 | 6,435 |

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
it is not a substitute for these measurements. A browser smoke checked selected
workspace actions with the real companion and an unavailable driver. The final
component refactor passed static accessibility/type checks and HTTP regression
smoke; a final browser replay was unavailable because the browser tool exposed
no enabled browser.

A release report must record exact source, driver, model, workflow, topology,
client hardware/browser and baseline variance. Measure wall time, host CPU/GPU,
resident memory, network bytes, client interaction latency and retained memory
for every row. Set regression thresholds from repeatable baseline variance
before claiming acceptance. No runtime-overhead threshold has been fabricated
from bundle size or a mocked job.
