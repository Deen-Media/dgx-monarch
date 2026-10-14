# Benchmarks

Measured render times and peak memory for stock ComfyUI and Monarch.
[VALIDATION.md](VALIDATION.md) records output comparisons (NRMS);
[MODELS.md](MODELS.md) lists tested configurations and known limitations. Performance and fidelity
are separate results: neither establishes the other.

[Campaign results](CAMPAIGN_RESULTS.md) summarizes the published datasets.
[Column definitions](CAMPAIGN_COLUMNS.md) explains their timing, memory and
comparison fields. Times below are seconds unless marked otherwise.

## Selected results

| Workflow | One GPU | Two GPUs | Settings and comparison |
|---|--:|--:|---|
| Krea2 RAW | 74.1 s | 48.1 s | BF16, 1536x1536, 8 steps, CFG 1, CFG++ sampler, Ulysses |
| Ideogram4 | 36.0 s | 26.0 s | FP8-scaled dual-model, 1024x1024, 20 steps, CFG 7, Ulysses |
| MiniMax H3 video/audio | 42:03 | 23:07 | INT8 ConvRot, 1280x720, 241 frames, 20 steps, CFG 1, exact-attention Ulysses; times are minutes:seconds |
| Qwen Image 2.1 | Not timed | Not timed | Matching native output recorded for 25-step T2I and editing; no comparable timing published here |

Krea2 and Ideogram4 use the [Ulysses comparison tests](VALIDATION.md#ulysses-comparisons).
Their times are warm full renders of the same graph on one GPU and two GPUs. The first Spark had a 2100 MHz GPU clock cap; the second was uncapped. Separate one-step probes produced identical decoded pixels;
that does not establish equality at every step of the full render.

The [H3 comparison](VALIDATION.md#evidence-minimax-h3-t2va-fl2va-ref2va-av)
used stock ComfyUI for the one-GPU timing. One Monarch worker took 42:20,
and its output matched the exact-attention pair. This is a different recipe
from the shorter September H3 runs below, whose audio comparison was missing.

[Qwen Image 2.1 tests](VALIDATION.md#evidence-qwen-image-2-1) compare complete
latents and RGBA pixels against native ComfyUI. Qwen-Image 2512 timings below
belong to a different model and do not establish a Qwen Image 2.1 speedup.

## Method

- **Hardware**: two NVIDIA DGX Spark (GB10, 121.6 GiB unified memory each),
  paired over ConnectX-7 200G RoCE on both rails. Idle baseline ~7 GiB per box.
- **Warm renders only.** Every leg queues twice with the same graph, steps, cfg,
  and files; the first queue pays model loads, the first-use identity gate,
  and shape compiles, and is discarded. The second queue (seed changed so the
  execution cache cannot serve it) is the published elapsed time. Observed run-to-run
  spread on repeated legs was 0-9%.
- **Memory is peak total system usage** (`MemTotal - MemAvailable`, 1 Hz
  sampler on each box) during the warm sampler window. GB10 has one unified
  pool, so this number decides what else fits; CUDA-allocated bytes do not.
- **Attention backend:** measurements use the kernel reported by each worker,
  not just the requested setting.
- Solo legs run stock ComfyUI with this rig's validated launch flags and no
  Monarch code loaded. "Monarch 1 GPU" runs the full Monarch pack in local
  mode on one Spark, as the pack-overhead control. Dual legs run the family's
  tested topology across both Sparks.
- The unrestricted-clock stills tables were measured on August 21, 2026,
  using ComfyUI `76135e55`, dgx-monarch 1.0.0 and torch 2.12.0+cu132. Ideogram4 was remeasured
  on August 25. Later tables state their own clock and software conditions.

<a id="stills-warm-render-time-seconds-2026-08-21-neither-box-capped"></a>

## Stills: warm render time with unrestricted clocks

| Family (artifact, recipe) | Solo Comfy | Monarch 1 GPU | Dual exact | Dual sage | Dual vs solo |
|---|--:|--:|--:|--:|--:|
| Chroma 1 HD fp8mixed, 1536x1536, 26 steps, cfg 3.5, uly2 | 124.1 | 124.1 | 82.1 | 76.1 | **1.51x** (1.63x sage) |
| Krea2 RAW fp8, 1536x1536, 8 steps, cfg 1.0, uly2 | 64.1 | 70.2 | 48.0 | 46.1 | **1.34x** (1.39x sage) |
| Flux2 dev fp8mixed, 1024x1024, 20 steps, guidance 4.0, uly2 | 56.1 | 56.1 | 36.0 | n/a | **1.56x** |
| Boogu fp8, 1024x1024, 25 steps, cfg 4.0, uly2 | 48.1 | 48.1 | 34.0 | n/a | **1.41x** |
| Ideogram4 fp8 dual-model, 1024x1024, 20 steps, cfg 7.0, uly2 (re-timed 2026-08-25 after the residency correction) | 36.0 | 36.0 | 26.0 | n/a | **1.38x** |
| Ideogram4, same recipe, before the residency correction (2026-08-21) | 36.0 | 110.1 | 100.1 | n/a | 0.36x (slower) |

Reading the table:

- **Measured pack overhead.** On chroma, flux2, and boogu the Monarch
  1-GPU wall equals the solo wall to the second; krea2's short 8-step run
  takes 70.2 s against 64.1 s.
- **Sage** reduces wall time by ~7% on chroma at 1536 (76.1 vs 82.1). Its
  effect on krea2's 8-step recipe is within the run-to-run spread. It applies only where the auto table
  turns it on (fp8 + USP + high resolution); the exact-attention walls are
  the default claim.
- **Ideogram4:** the older slow result includes repeated checkpoint offloads.
  Correct residency reduced the one-GPU time from 110.1 to 36.0 s and the
  pair from 100.1 to 26.0 s. Both ranks still returned identical tensors.
- Sol-Attn is a video lever and is not a column here; the MiniMax H3 video
  [validation record](VALIDATION.md#evidence-minimax-h3-t2va-fl2va-ref2va-av)
  records its separate 2.58-2.69x comparisons against one GPU.

## Stills: peak memory (GiB, `used` = MemTotal - MemAvailable)

| Family | Solo (one box) | Monarch 1 GPU | Dual: head | Dual: sibling | Dual pair sum |
|---|--:|--:|--:|--:|--:|
| Chroma | 29.4 | 33.2 | 40.6 | 26.9 | 67.6 |
| Krea2 | 32.7 | 34.5 | 45.8 | 31.3 | 77.2 |
| Flux2 | 79.4 | 82.6 | 89.1 | 50.9 | 140.1 |
| Boogu | 30.3 | 31.3 | 43.4 | 29.5 | 72.9 |
| Ideogram4 (re-timed 2026-08-25 after the residency correction) | 37.8 | 39.2 | 51.4 | 34.6 | 86.0 |
| Ideogram4, before the residency correction (2026-08-21) | 37.8 | 49.2 | 73.8 | 62.2 | 136.0 |

The head box carries the driver (text encoders, VAE) plus its worker, so the
dual head peak sits above the solo peak; the sibling carries only its worker.
The actor memory overhead (Monarch 1 GPU minus solo) is 1-4 GiB on every
family once the Ideogram4 offload churn is gone (re-timed 2026-08-25: 39.2 GiB against
37.8 solo, and the dual pair sum fell from 136.0 to 86.0 GiB).

## Stills: cfg2 warm render time

On October 1, 2026, below 1.2 MP, automatic selection offered cfg2 for
Chroma BF16/FP8/INT8 and LongCat BF16 when CFG was not 1. Campaign K
(2026-10-14) later added Boogu FP8 and Anima BF16 below 1.2 MP, and cfg2 at
CFG 1 under a CFG++ sampler
([Campaign K](#campaign-k-cfg2-against-uly2-2026-10-14-head-capped)). These
October 1, 2026 measurements used
ComfyUI `a7169322` and torch 2.12.0+cu132. The first Spark was capped at 2100 MHz (2086 MHz median
SM clock); the second was uncapped. Compare within this table.

Recipes, the same graph and files in all four legs of a family:

- **Chroma1-HD bf16:** euler with the beta scheduler; the
  `t5xxl_fp8_e4m3fn_scaled` text encoder and the `ae` VAE. The prompt pair is
  "a lighthouse on a rocky coast at golden hour, dramatic clouds, oil
  painting" against "blurry, low quality, watermark, oversaturated, deformed,
  noisy frame".
- **LongCat-Image bf16:** euler with the simple scheduler; the
  `qwen_2.5_vl_7b_fp8_scaled` text encoder and the `ae` VAE. The prompt pair
  is the one in the shipped `example_workflows/dgx-monarch-longcat-t2i.json`.

| Family (artifact, recipe) | Solo Comfy | Monarch 1 GPU | Dual cfg2 | Dual uly2 | cfg2 vs solo |
|---|--:|--:|--:|--:|--:|
| Chroma1-HD bf16, 1024x1024, 26 steps, cfg 3.5 | 64.1 | 66.3 | 32.4 | 42.3 | **1.98x** |
| LongCat-Image bf16, 1024x1024, 20 steps, cfg 4.0 | 32.0 | 32.4 | 16.0 | 20.0 | **2.00x** |

Peak memory for the same legs (GiB, `used`):

| Family | Solo | Monarch 1 GPU | cfg2 head / sibling | uly2 head / sibling |
|---|--:|--:|--:|--:|
| Chroma1-HD bf16 | 33.1 | 37.8 | 44.8 / 37.4 | 48.2 / 36.9 |
| LongCat-Image bf16 | 30.9 | 34.4 | 41.4 / 30.9 | 45.1 / 32.0 |

Reading the table:

- **cfg2 roughly halves wall time on these recipes.** It splits a guided
  step's two model calls across the pair. uly2 on the same recipes gives
  speedups of 1.52x and 1.60x.
- **Automatic cfg2 selection uses output comparisons and timing.** With conditioning mask trimming,
  Chroma cfg2 produced identical tensors to one GPU on bf16, fp8 and int8, and the sweep
  harness's warm walls at 1024 square (2026-10-01, head capped) put it ahead
  of uly2 on every quant, so auto selects cfg2 for all three:

  | quant | one GPU | cfg2 | uly2 |
  |---|--:|--:|--:|
  | bf16 | 64.2 | 32.0 | 42.1 |
  | fp8_scaled | 46.1 | 24.0 | 32.0 |
  | fp8mixed | 46.0 | 24.0 | 32.2 |
  | mxfp8 | 52.3 | 26.2 | 34.1 |
  | int8_convrot | 48.6 | 26.4 | 34.0 |

  These are the harness's warm legs (one render after a cold one), not this
  file's two-queue method, so read them against each other, not against the
  table above.

<a id="campaign-k-cfg2-against-uly2-2026-10-14-head-capped"></a>

## Campaign K: cfg2 against uly2

Campaign K timed cfg2 and uly2 against one GPU on 2026-10-14, so the
automatic table could gain cfg2 rows above 1.2 MP. Every cell ran dgx-monarch
`d549f10d` except a09-anima-1024, which ran `65d14333`. Both Sparks ran
ComfyUI `e8665ca8`. The first Spark was capped at 2100 MHz, with a highest
sampled SM clock of 2093 MHz; the second was uncapped.

Method:

- **Instrument.** `benchmark/run_matrix.py` in cluster mode, one process per
  cell. Each leg ran one cold render and two warm renders, and the table gives
  the mean of the two warm sampler walls in seconds. A sampler wall leaves out
  text encoding and VAE decode, so these numbers read lower than the
  whole-prompt walls in the tables above. Compare within this table.
- **One GPU.** At world 2 the reference is a dp2 render of batch 2. Each rank
  renders one image alone, so batch row 0 carries one-GPU math and its wall is
  the capped first Spark's one-GPU wall.
- **uly2** ran on Torch Flash. **Auto SAGE uly2** is the route auto took
  before this campaign for Krea2 and Chroma FP8 above 1.2 MP, measured where
  it applies.
- **Fidelity.** Bit-identical means batch row 0 of the full-schedule latent
  and of a one-step latent both equal the one-GPU reference exactly.
  Otherwise the table gives the one-step NRMS and the full schedule's largest
  absolute latent difference.
- **Recipes.** Prompts, steps, CFG, sampler, scheduler and text encoders come
  from each family's shipped template. The one exception is the Krea2
  control, which runs CFG 4.5 under plain Euler. The Krea2 template runs CFG 1
  under `euler_cfg_pp` with an empty negative prompt; a CFG++ sampler keeps
  the unconditional pass, so cfg2 still splits the step.

| Cell | Checkpoint | Size | MP | Steps, CFG, sampler | One GPU | uly2 | cfg2 | cfg2 vs one GPU | cfg2 vs uly2 | Auto SAGE uly2 | cfg2 against one GPU |
|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|---|
| a09-anima-1024 | Anima BF16 | 1024 | 1.05 | 30, 4, `euler` | 19.6 | 12.7 | 10.0 | 1.96x | 1.27x | not run | bit-identical |
| a08-boogu-fp8-1024 | Boogu FP8-scaled | 1024 | 1.05 | 25, 4, `euler` | 48.1 | 34.0 | 24.3 | 1.98x | 1.40x | not run | bit-identical |
| b-chroma-bf16-1152 | Chroma1-HD BF16 | 1152 | 1.33 | 26, 3.5, `euler` | 77.3 | 51.6 | 39.3 | 1.97x | 1.31x | not run | bit-identical |
| b-chroma-fp8mixed-1152 | Chroma1-HD FP8-mixed | 1152 | 1.33 | 26, 3.5, `euler` | 58.2 | 39.2 | 29.1 | 2.00x | 1.34x | 38.0 (1-step NRMS 0.078) | bit-identical |
| b-chroma-bf16-1216 | Chroma1-HD BF16 | 1216 | 1.48 | 26, 3.5, `euler` | 87.2 | 56.6 | 43.6 | 2.00x | 1.30x | not run | bit-identical |
| b-chroma-fp8mixed-1216 | Chroma1-HD FP8-mixed | 1216 | 1.48 | 26, 3.5, `euler` | 65.0 | 43.8 | 32.6 | 1.99x | 1.34x | 42.0 (1-step NRMS 0.065) | bit-identical |
| b-chroma-bf16-1280 | Chroma1-HD BF16 | 1280 | 1.64 | 26, 3.5, `euler` | 97.9 | 64.6 | 49.2 | 1.99x | 1.31x | not run | bit-identical |
| b-chroma-fp8mixed-1280 | Chroma1-HD FP8-mixed | 1280 | 1.64 | 26, 3.5, `euler` | 75.4 | 49.8 | 37.8 | 2.00x | 1.32x | 47.5 (1-step NRMS 0.057) | bit-identical |
| c-chroma-int8-1152 | Chroma1-HD INT8 ConvRot | 1152 | 1.33 | 26, 3.5, `euler` | 61.2 | 40.3 | 30.6 | 2.00x | 1.32x | not run | bit-identical |
| c-chroma-int8-1216 | Chroma1-HD INT8 ConvRot | 1216 | 1.48 | 26, 3.5, `euler` | 70.7 | 46.9 | 35.1 | 2.01x | 1.33x | not run | bit-identical |
| c-chroma-int8-1280 | Chroma1-HD INT8 ConvRot | 1280 | 1.64 | 26, 3.5, `euler` | 87.8 | 51.7 | 44.0 | 2.00x | 1.17x | not run | bit-identical |
| a07-kandinsky5-1024 | Kandinsky5 Lite image BF16 | 1024 | 1.05 | 50, 3.5, `euler` | 81.0 | 49.4 | 40.5 | 2.00x | 1.22x | not run | bit-identical |
| b-krea2-turbo-fp8-1152 | Krea2 Turbo FP8-scaled | 1152 | 1.33 | 8, 1, `euler_cfg_pp` | 34.7 | 22.5 | 17.3 | 2.00x | 1.30x | 22.3 (1-step NRMS 0.030) | bit-identical |
| b-krea2-turbo-fp8-1216 | Krea2 Turbo FP8-scaled | 1216 | 1.48 | 8, 1, `euler_cfg_pp` | 39.1 | 25.3 | 19.5 | 2.01x | 1.30x | 24.9 (1-step NRMS 0.033) | bit-identical |
| b-krea2-turbo-fp8-1280 | Krea2 Turbo FP8-scaled | 1280 | 1.64 | 8, 1, `euler_cfg_pp` | 43.7 | 27.7 | 21.7 | 2.01x | 1.27x | 27.1 (1-step NRMS 0.058) | bit-identical |
| c-krea2-turbo-fp8-1408 | Krea2 Turbo FP8-scaled | 1408 | 1.98 | 8, 1, `euler_cfg_pp` | 53.5 | 33.8 | 26.9 | 1.99x | 1.25x | 32.8 (1-step NRMS 0.030) | bit-identical |
| c-krea2-raw-bf16-1152 | Krea2 RAW BF16 | 1152 | 1.33 | 8, 1, `euler_cfg_pp` | 40.4 | 25.3 | 20.1 | 2.01x | 1.26x | not run | bit-identical |
| c-krea2-raw-bf16-1216 | Krea2 RAW BF16 | 1216 | 1.48 | 8, 1, `euler_cfg_pp` | 45.5 | 27.9 | 22.6 | 2.02x | 1.24x | not run | bit-identical |
| c-krea2-raw-bf16-1280 | Krea2 RAW BF16 | 1280 | 1.64 | 8, 1, `euler_cfg_pp` | 51.0 | 31.1 | 25.2 | 2.02x | 1.23x | not run | bit-identical |
| a01-krea2-raw-bf16-1536 | Krea2 RAW BF16 | 1536 | 2.36 | 8, 1, `euler_cfg_pp` | 72.9 | 44.8 | 36.3 | 2.01x | 1.23x | not run | bit-identical |
| a02-krea2-turbo-bf16-1536 | Krea2 Turbo BF16 | 1536 | 2.36 | 8, 1, `euler_cfg_pp` | 72.6 | 46.3 | 36.1 | 2.01x | 1.28x | not run | bit-identical |
| a03-krea2-turbo-fp8-1536 | Krea2 Turbo FP8-scaled | 1536 | 2.36 | 8, 1, `euler_cfg_pp` | 62.9 | 39.8 | 31.4 | 2.00x | 1.27x | 38.9 (1-step NRMS 0.042) | bit-identical |
| c-krea2-raw-int8-1152 | Krea2 RAW INT8 ConvRot | 1152 | 1.33 | 8, 1, `euler_cfg_pp` | 40.6 | 19.6 | 20.2 | 2.01x | 0.97x | not run | bit-identical |
| c-krea2-raw-int8-1216 | Krea2 RAW INT8 ConvRot | 1216 | 1.48 | 8, 1, `euler_cfg_pp` | 47.8 | 23.4 | 23.8 | 2.01x | 0.98x | not run | bit-identical |
| c-krea2-raw-int8-1280 | Krea2 RAW INT8 ConvRot | 1280 | 1.64 | 8, 1, `euler_cfg_pp` | 52.5 | 27.2 | 26.3 | 1.99x | 1.03x | not run | bit-identical |
| a04-krea2-raw-int8-1536 | Krea2 RAW INT8 ConvRot | 1536 | 2.36 | 8, 1, `euler_cfg_pp` | 82.0 | 44.8 | 41.4 | 1.98x | 1.08x | not run | bit-identical |
| a05-krea2-turbo-fp8-1536-cfg45-euler | Krea2 Turbo FP8-scaled, control | 1536 | 2.36 | 8, 4.5, `euler` | 64.9 | 40.7 | 32.4 | 2.00x | 1.26x | 39.1 (1-step NRMS 0.041) | bit-identical |
| b-longcat-bf16-1152 | LongCat-Image BF16 | 1152 | 1.33 | 20, 4, `euler` | 37.8 | 23.5 | 18.6 | 2.03x | 1.26x | not run | 1-step NRMS 0.007; full schedule max abs 7.62 |
| b-longcat-bf16-1216 | LongCat-Image BF16 | 1216 | 1.48 | 20, 4, `euler` | 42.1 | 25.9 | 20.4 | 2.06x | 1.27x | not run | 1-step NRMS 0.007; full schedule max abs 2.65 |
| b-longcat-bf16-1280 | LongCat-Image BF16 | 1280 | 1.64 | 20, 4, `euler` | 46.1 | 28.6 | 22.5 | 2.05x | 1.27x | not run | 1-step NRMS 0.006; full schedule max abs 4.54 |
| b-qwen-fp8-1152 | Qwen-Image 2512 FP8 | 1152 | 1.33 | 50, 4, `euler` | 214.4 | 169.8 | 102.3 | 2.10x | 1.66x | not run | bit-identical |
| b-qwen-fp8-1216 | Qwen-Image 2512 FP8 | 1216 | 1.48 | 50, 4, `euler` | 235.4 | 179.9 | 113.8 | 2.07x | 1.58x | not run | bit-identical |
| b-qwen-fp8-1280 | Qwen-Image 2512 FP8 | 1280 | 1.64 | 50, 4, `euler` | 253.1 | 196.3 | 122.7 | 2.06x | 1.60x | not run | bit-identical |
| a06-qwen-fp8-1328 | Qwen-Image 2512 FP8 | 1328 | 1.76 | 50, 4, `euler` | 271.2 | 211.2 | 132.1 | 2.05x | 1.60x | not run | bit-identical |

Reading the table:

- **cfg2 beat uly2 in every cell but Krea2 RAW INT8 below 1536.** Elsewhere
  it ran 1.17x (Chroma INT8 at 1280) to 1.66x (Qwen-Image at 1152) faster
  than uly2 and 1.96x to 2.10x faster than one GPU, and its slowest warm leg
  beat uly2's fastest. Krea2 RAW INT8 read 0.97x, 0.98x and 1.03x against
  uly2 at 1152, 1216 and 1280, and 1.08x at 1536: its uly2 leg ran 1.93x to
  2.07x faster than one GPU, against 1.59x to 1.64x on BF16, so the split had
  less to gain.
- **cfg2 was bit-identical to one GPU in every cell but LongCat.** LongCat
  read 1-step NRMS 0.006 to 0.007, and its full-schedule latent differed,
  while uly2 read bit-identical on the same cells. LongCat keeps Ulysses above
  1.2 MP.
- **The SAGE route bought little.** It ran 1 to 5 percent faster than Torch
  Flash uly2 and read 1-step NRMS 0.030 to 0.078 against one GPU, while Torch
  Flash uly2 read bit-identical.
- **New automatic rows.** Krea2 FP8 and BF16 from 1.2 to 2.36 MP (row 15),
  Chroma BF16, FP8 and INT8 from 1.2 to 1.64 MP (row 19), Qwen-Image FP8 from
  1.2 to 1.77 MP (row 68), Boogu FP8 below 1.2 MP (row 79) and Anima BF16
  below 1.2 MP (row 47). Each upper band ends at the largest size measured.
  The two lower bands follow rows 20 and 28 and rest on one size, 1.05 MP.
  Each grant covers only the precisions that read bit-identical and beat uly2
  at every measured size in its band, so Krea2 RAW INT8 keeps Ulysses (row
  12). Kandinsky5's image and video checkpoints share one family, so it has
  no row.

<a id="campaign-render-times-stills-2026-09-12-to-2026-09-28-head-capped"></a>

## Campaign render times: stills

Measured September 12-28, 2026 with one dgx-monarch revision and matching
ComfyUI `3216c62e` installations, torch 2.12.0+cu132 and NCCL 2.30.7. The dataset's
NCCL 2.29.7 label records torch's build version rather than the loaded library.

Each value is one warm run after a cold run, using a changed seed to avoid
cached execution. Matching one- and two-GPU runs use the same graph and seed.
The first Spark was capped at 2100 MHz; the second was uncapped. Polling every
2 seconds limits precision, especially for short jobs: 12 versus 10 seconds
is a small difference. Compare within this table, not with unrestricted-clock
results above.

All rows use Torch Flash with residency optimizations and LoRA off. `C`
means the one-step comparison returned CHECK; the time is valid but the
comparison did not pass. A dash means no measurement exists.

| Family (artifact, recipe) | One GPU | uly2 | cfg2 | ring2 | Fastest PASS vs one GPU |
|---|--:|--:|--:|--:|--:|
| Anima bf16, 1024x1024, 30 steps, CFG 4.0 | 20.0 | 14.0 | 12.0 | 14.0 | 1.67x (cfg2) |
| Boogu fp8_scaled, 1024x1024, 25 steps, CFG 4.0 | 50.1 | 34.0 | 26.0 | 34.0 | 1.93x (cfg2) |
| Chroma1-HD fp8mixed, 1024x1024, 26 steps, CFG 3.5, shipped prompts | 48.1 | 32.0 | 24.0 C | 32.0 C | 1.50x (uly2) |
| Chroma1-HD bf16, same recipe, equal-token testing prompts | 66.1 | 42.0 | 32.4 | 42.1 | 2.04x (cfg2) |
| Ernie-Image bf16, 1024x1024, 20 steps, CFG 4.0 | 50.1 | 34.0 | 36.0 | 30.0 | 1.67x (ring2) |
| Flux 1 Schnell bf16 in the Flux 1 graph, 1024x1024, 20 steps, CFG 1.0 | 26.0 | 18.0 | - | 18.0 | 1.44x (uly2) |
| HunyuanImage 2.1 fp8_e4m3fn, 2048x2048, 20 steps, CFG 3.5 | 82.1 | 52.1 | 44.1 | 52.1 | 1.86x (cfg2) |
| Ideogram4 fp8_scaled dual-model, 1024x1024, 20 steps, CFG 7.0 | 36.1 | 24.0 | 20.0 | 24.0 | 1.81x (cfg2) |
| Ideogram4 bf16 dual-model, same recipe | 56.1 | 36.0 | 30.0 | 40.0 | 1.87x (cfg2) |
| Kandinsky5 Image Lite bf16, 1024x1024, 50 steps, CFG 3.5 | 84.1 | 52.1 | 42.0 | 56.0 | 2.00x (cfg2) |
| Krea2 RAW fp8_scaled, 1536x1536, 8 steps, CFG 1.0 (cfg++ sampler) | 68.2 | 44.0 | 36.0 | 44.0 | 1.89x (cfg2) |
| Lens Turbo bf16, 1344x1344, 20 steps, CFG 5.0 | 38.0 | 22.0 | 20.0 | 20.1 C | 1.90x (cfg2) |
| LongCat-Image bf16, 1024x1024, 20 steps, CFG 4.0 | 32.0 | 20.0 | 16.0 | 20.1 | 2.00x (cfg2) |
| Omnigen2 fp16, 1024x1024, 20 steps, CFG 5.0 | 26.0 | - | 14.0 | 18.0 | 1.86x (cfg2) |
| PixelDiT 1300M bf16, 1024x1024, 30 steps, CFG 4.0 | 12.0 | 10.0 | - | 10.0 | 1.20x (uly2) |
| PiD 1024-to-4096 bf16, 4 steps, CFG 1.0 | 34.2 | 34.0 | - | 34.1 | 1.01x (uly2) |
| Qwen-Image 2512 fp8_e4m3fn, 1328x1328, 50 steps, CFG 4.0 | 266.3 | 184.2 | 136.1 | 192.2 | 1.96x (cfg2) |
| Radiance x0 bf16, 1024x1024, 30 steps, CFG 3.5 | 94.1 | 64.1 | 46.0 | 64.1 | 2.05x (cfg2) |
| Z-Image bf16, 1536x1536, 8 steps, CFG 1.0 | 20.0 | 14.0 | - | 14.0 | 1.43x (uly2) |

Reading the table:

- **cfg2 is the fastest split on most rows that ran it.** It runs 1.67x to
  2.05x everywhere except Ernie, where it runs 1.39x and is slower than uly2
  and ring2 on this graph. Krea2 splits at CFG 1.0 because its template uses
  a cfg++ sampler.
- **Sequence splits give smaller speedups.** uly2 runs 1.43x to 1.73x, except
  PixelDiT (1.20x on a 12 s wall) and PiD's 4-step 1024-to-4096 graph (1.01x).
- **What is missing.** Flux2 has no row. Its BF16 single-GPU references
  refused class C on the capped head, and every lever-free fp8mixed `auto`,
  uly2 and ring2 cell on Torch Flash refused class C beside the driver
  (capacity admission corrected 2026-10-01). The one lever-free fp8mixed cell in those
  topologies that rendered, `auto` on SAGE_AUTO, read 34.0 s
  warm against 56.1 s on one GPU. With the first-use gate off, the three
  lever-free fp8mixed `uly2+fsdp` cells rendered at 66.1 s, slower than one
  GPU; with the gate on they refused the clean-reload proof. The
  HunyuanImage BF16 references ran on the uncapped sibling after the head
  refused them, so that row is left out. The sweep's artifact matcher did not
  match Flux 1 Dev's file, so the Flux 1 graph ran with Schnell.
- **Other quants and topologies.** Chroma, Ideogram4, Krea2 and Lens each ran
  more quants than this table shows. They are in
  [CAMPAIGN_RESULTS.md](CAMPAIGN_RESULTS.md), with every other topology,
  kernel and template the campaign ran.

<a id="campaign-render-times-video-2026-09-27-to-2026-09-28-head-capped"></a>

## Campaign render times: video

Same records, method and caveats as the stills above. Each pair cell is the
`auto` preset on Torch Flash with no levers and no LoRA, against the local
reference it names. `auto` resolved to uly2 on every row whose record names
its topology; the records of the LTX rows and the H3 testing graph name none.
Verdicts are the cells' own. Every CHECK row lacked the `soundfile` audio
comparator at run time (VALIDATION.md), and the LTX 2.5 two-stage graph also
read one-step NRMS 0.106, over the 0.10 floor. Their elapsed times remain valid measurements.

| Template, artifact, shape | One GPU | Pair (auto) | Speedup | Verdict |
|---|--:|--:|--:|---|
| HunyuanVideo 1.5 720p T2V fp16, 1280x720, 121 frames, 20 steps | 3581.0 | 1935.4 | 1.85x | PASS |
| Kandinsky5 Video Lite bf16, 768x512, 121 frames, 50 steps | 940.9 | 534.5 | 1.76x | PASS |
| Wan 2.1 I2V 480p fp8_scaled, 832x480, 81 frames, 20 steps | 964.9 | 560.4 | 1.72x | PASS |
| Wan 2.1 T2V 14B fp8_scaled, 832x480, 81 frames, 20 steps | 952.9 | 556.5 | 1.71x | PASS |
| Wan Bernini-R high/low fp8_scaled, 832x480, 81 frames, 40 steps | 1611.5 | 942.9 | 1.71x | PASS |
| Wan 2.2 I2V high/low fp8_scaled, 640x640, 81 frames, 20 steps | 850.8 | 498.5 | 1.71x | PASS |
| CogVideoX 1.5 I2V bf16, 1360x768, 45 frames, 50 steps | 1027.1 | 612.6 | 1.68x | PASS |
| Wan 2.2 T2V high/low fp16, 832x480, 81 frames, 20 steps | 982.9 | 596.6 | 1.65x | PASS |
| CogVideoX 1.5 T2V bf16, 720x480, 48 frames, 50 steps | 258.3 | 164.2 | 1.57x | PASS |
| LTX 2.3 T2V fp8_scaled, 1280x704, 121 frames | 80.1 | 56.3 | 1.42x | PASS |
| Wan SCAIL2 fp8_scaled, testing graph, 896x512, 81 frames, 40 steps | 3010.5 | 2369.9 | 1.27x | PASS |
| Wan FlowRVS bf16, testing graph, 832x480, 17 frames, 1 step | 4.0 | 4.0 | 1.00x | PASS |
| MiniMax H3 guide int8_convrot, 1344x768, 124 frames, 20 steps | 1161.3 | 636.6 | 1.82x | CHECK |
| MiniMax H3 guides int8_convrot, testing graph, same shape | 1197.2 | 668.8 | 1.79x | CHECK |
| MiniMax H3 t2va fp8_scaled, 1344x768, 124 frames, 20 steps | 924.9 | 564.5 | 1.64x | CHECK |
| LTX 2.5 multishot int8_convrot, testing graph, 1280x704, 121 frames | 106.1 | 82.1 | 1.29x | CHECK |
| LTX 2.5 T2V int8_convrot, 1280x704, 121 frames | 102.3 | 80.1 | 1.28x | CHECK |
| LTX 2.5 guide mask int8_convrot, testing graph, 1280x704, 121 frames | 122.1 | 96.1 | 1.27x | CHECK |
| LTX 2.5 I2V guide int8_convrot, 1280x704, 121 frames | 114.1 | 90.3 | 1.26x | CHECK |
| LTX 2.5 I2V int8_convrot, 1280x704, 121 frames | 110.1 | 88.1 | 1.25x | CHECK |
| LTX 2.5 two-stage int8_convrot, testing graph, 640x352, 121 frames | 80.1 | 68.1 | 1.18x | CHECK |
| LTX 2.3 I2V AV fp8_scaled, 1280x704, 121 frames | 100.1 | 88.1 | 1.14x | CHECK |
| LTX 2.5 FLF2V int8_convrot, 1280x704, 121 frames | 122.1 | 108.1 | 1.13x | CHECK |

Reading the table:

- **Long video gains most.** HunyuanVideo, Kandinsky5 Lite, the Wan 2.1 and
  2.2 graphs, Bernini-R, CogVideoX and MiniMax H3 run 1.57x to 1.85x.
- **LTX gains less.** The LTX graphs run 1.13x to 1.42x.
- **Two PASS rows gain little.** SCAIL2's testing graph runs 1.27x, and
  FlowRVS's one-step synthetic graph takes 4.0 s either way.
- **Missing rows.** The one-GPU references for Kandinsky5 Video Pro and SCAIL
  Preview refused class C on the capped head, and WanDancer's testing graph
  refused its pair warm leg for capacity, so none of the three has a speedup.

<a id="the-head-clock-caps-cost-2026-09-02"></a>

## GPU clock cap comparison

A September 2 control repeated three one-GPU recipes on the uncapped second
Spark. ComfyUI `3216c62e` and graphs matched; dgx-monarch revisions differed, so this
was an approximate control rather than an isolated clock experiment.

| Template and recipe | Head, capped at 2100 MHz | Sibling, uncapped | Cost of the cap |
|---|--:|--:|--:|
| Krea2 RAW bf16, 1536x1536, 8 steps | 76.1 | 74.1 | +3% |
| Ideogram4 bf16 dual-model, 1024x1024, 20 steps | 56.1 | 54.1 | +4% |
| Boogu fp8_scaled, 1024x1024, 25 steps | 50.1 | 48.1 | +4% |

Each gap is one polling interval. The observed difference was at most about
4 percent, and a later capped Krea2 repeat took 74.1 s, equal to the uncapped
result. Software and artifact changes also affect timing, so these observations
do not provide a universal clock adjustment for other tables.

<a id="the-fsdp-round-capacity-path-not-speed"></a>

## FSDP memory capacity and render time

The 60-GiB BF16 Flux2 file rendered with `uly2+fsdp` beside the ComfyUI
driver. Stock residency did not fit. These August 2026 tests used torch
2.12.0+cu132; the August 25 rows use a different ComfyUI version from the
August 21 row and should not be compared as a topology-only change.
The table records both ComfyUI revisions; dgx-monarch was version 1.0.0.

| leg | warm wall | used: head | used: sibling |
|---|--:|--:|--:|
| Flux2 dev **bf16 60 GiB**, 1024x1024, 20 steps, `uly2+fsdp` (2026-08-21, ComfyUI `76135e55`) | **104.2 s** | 92.3 | 56.3 |
| same recipe, 2026-08-25, ComfyUI `7d9d0c39`, without root reshard | 114.1 s | - | - |
| same recipe, 2026-08-25, ComfyUI `7d9d0c39`, with root reshard | 116.2 s | - | - |

Reading the row:

- FSDP lets the 60 GiB bf16 artifact fit beside the driver. Stock residency
  does not fit; the sharded render takes 104.2 s, compared with 107.7 s at
  the 2026-07-13 acceptance. The fp8mixed resident path in the tables above
  stays the production Flux2 route.
- The `used` gauge overstates what a sharded rank costs, because the mmap'd
  checkpoint sits inside it as reclaimable page cache. A separate leg on
  2026-08-25 decomposed it on a 32.5 GiB bf16 HunyuanImage
  `uly2+fsdp` round: a settled rank held 16.2 GiB of shard plus about 1 GiB
  of root-level parameters allocated on the device (17.3 GiB, near 0.53x the
  checkpoint), against 21.2 GiB reserved. That 3.9 GiB gap is allocator
  slack: a soft Clear VRAM handed 3.6 GiB of it back without unloading
  anything. Host anonymous pages stayed near 7 GiB on the head box and 3 GiB
  on the sibling, nothing checkpoint-sized. Judge co-residency headroom by
  the device figure.
- The root wrap reshards after forward. On the
  HunyuanImage leg the reshard dropped per-rank allocated memory between
  renders from 17.3 to 16.4 GiB, with 21.2 GiB reserved before and after.
  The saving is at rest, not at peak: the root group re-gathers at each
  pre-forward, so a rank holds those parameters at full size while a render
  runs, and each step pays one more exposed all-gather. On the same ComfyUI
  nightly the reshard costs this row 1.8% (116.2 s against 114.1 s, inside
  run spread), while the nightly itself, on the same dgx-monarch source,
  moved the row by 9.5%. Re-time the whole ledger on one ComfyUI commit
  before quoting cross-date ratios.
- Run with the first-use gate off. On 2026-10-13 the gate's clean-reload proof
  on this file refused at its initial price. The driver's 33 GiB text encoder
  was resident, and the driver host read 39.1 GiB available, with its 30.0 GiB
  share resident, against a 39.8 GiB price. The proof unloads each rank's
  shards before every load it makes, so the price now credits that resident
  shard pool, capped at one share. That state now reads 39.1 plus 29.9 GiB.
  The Flux2 proof has not been rerun on hardware with the credit. A refusal
  names the GiB available and the credit it applied. If the load still does
  not fit, the refusal names `auto_gate=off` as the intended path
  ([TROUBLESHOOTING #88](TROUBLESHOOTING.md#88-the-first-render-of-an-fsdp-topology-refuses-with-a-capacity-figure-and-no-proof)).

## Updating results

Record the date, software versions, hardware settings, recipe and comparison
method with each new result. Keep older measurements when they explain a
change or limitation; identify superseded results clearly.
