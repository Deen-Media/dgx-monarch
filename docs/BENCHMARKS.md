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
| MiniMax H3 image to video, 15 s | refused | 53:45 | INT8 ConvRot with the Triton GEMM, 1344x768, 362 frames, 20 steps, CFG 1, exact-attention Ulysses, stock residency; a first render of its session, gate proof and model load inside; one Spark cannot hold the activations; times are minutes:seconds; head capped at 2100 MHz; [details](#a-15-second-clip-on-the-pair-2026-10-29) |
| Qwen Image 2.1 | Not timed | Not timed | Matching native output recorded for 25-step T2I and editing; no comparable timing published here |
| FLUX.3 Klein | 10.2 s | 6.7 s | Step-distilled BF16, 1024x1024, 16 steps, CFG 1, Euler, Ulysses; warm sampler walls, bit-identical to one GPU; head capped at 2100 MHz |
| FLUX.3 Klein Base | 29.8 s | 20.0 s | BF16, 1024x1024, 25 steps, CFG 3.5, Euler, Ulysses; warm sampler walls, bit-identical to one GPU; head capped at 2100 MHz |
| MOSS-Image | 90.0 s | 60.5 s | SFT BF16, 1024x1024, 50 steps, CFG 3, Euler/simple, layer-22 skip, Ulysses; warm sampler walls, bit-identical to one GPU; head capped at 2100 MHz |
| Kromoss | 88.7 s | 59.9 s | BF16, 1024x1024, 50 steps, CFG 3, Euler/simple, layer-22 skip, Ulysses; warm sampler walls, bit-identical to one GPU; head capped at 2100 MHz |

Krea2 and Ideogram4 use the [Ulysses comparison tests](VALIDATION.md#ulysses-comparisons).
Their times are warm full renders of the same graph on one GPU and two GPUs. The first Spark had a 2100 MHz GPU clock cap; the second was uncapped. Separate one-step probes produced identical decoded pixels;
that does not establish equality at every step of the full render.

The Krea2 row times the shipped template's graph, whose DGX Monarch KSampler
runs `euler_cfg_pp`, with RAW BF16 loaded. On a pair, `auto` takes cfg2 (row
15) for that graph, not the Ulysses it was timed on.
[Campaign K](#campaign-k-cfg2-against-uly2-2026-10-14-head-capped) cell a01
timed both at 1536: sampler walls of 44.8 s on uly2 and 36.3 s on
cfg2, with cfg2 bit-identical to one GPU. Sampler walls leave out text
encoding and VAE decode, so they do not compare with the table's times.

The [H3 comparison](VALIDATION.md#evidence-minimax-h3-t2va-fl2va-ref2va-av)
used stock ComfyUI for the one-GPU timing. One Monarch worker took 42:20,
and its output matched the exact-attention pair. This is a different recipe
from the shorter September H3 runs below, whose audio comparison was missing.

The
[FLUX.3 Klein times](#flux3-klein-one-gpu-uly2-and-cfg2-2026-10-15-to-2026-10-16-head-capped)
are warm sampler walls, which leave out text encoding and VAE decode, from
2026-10-15 and 2026-10-16. The first Spark was capped at 2100 MHz; the second
was uncapped.

The [MOSS-Image and Kromoss times](#moss-image-one-gpu-and-uly2-2026-10-21-to-2026-10-22-head-capped) are warm sampler walls from
2026-10-21 on ComfyUI's `simple` scheduler, with the same cap; the shipped
CFG 3 graphs now run `normal`, which takes the same work per step.

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
  uly2 at 1152, 1216 and 1280, and 1.08x at 1536: its uly2 leg ran 1.83x to
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

<a id="flux3-klein-one-gpu-uly2-and-cfg2-2026-10-15-to-2026-10-16-head-capped"></a>

## FLUX.3 Klein: one GPU, uly2 and cfg2

The FLUX.3 hardware ladder timed FLUX.3 Klein on 2026-10-15 and 2026-10-16
with dgx-monarch `70fb8997e3ea` and ComfyUI `29e28e2b4` on both Sparks. The
first Spark was capped at 2100 MHz, with a highest sampled SM clock of
2093 MHz; the second was uncapped.
[VALIDATION](VALIDATION.md#evidence-flux-3-klein) records the ladder and its
comparisons, and
[Campaign K](#campaign-k-cfg2-against-uly2-2026-10-14-head-capped) describes
the method this section follows. These walls predate the fused-norm parity
change, which gives each uly2 rank stock's full normalization work: op-level
timings put it at 1 to 31 ms per model call, about 2 to 4 percent of the uly2
walls below at 1024 square, until the hardware rerun times it.

Method:

- **Instrument.** `benchmark/run_matrix.py` in cluster mode, one process per
  cell. Each full-schedule leg ran one cold render and two warm renders, and
  the tables give the mean of the two warm sampler walls in seconds; one-step
  legs ran once, for fidelity. A sampler wall leaves out text encoding and VAE
  decode. Compare within each table.
- **One GPU.** Batch row 0 of a dp2 render of batch 2: each rank renders one
  image alone, so its wall is the capped first Spark's one-GPU wall.
- **Recipes.** The base model ran 25 Euler/simple steps at CFG 3.5 with an
  empty negative, and the step-distilled model 16 steps at CFG 1, both with
  the BF16 Qwen3-VL encoder and Torch Flash. The one-block prompt encodes to
  80 text rows, the size of the empty negative, so one GPU batches the two
  calls and cfg2 takes the slice path. The vendor-length prompt encodes to 480
  rows, so one GPU runs two calls and cfg2 dispatches one to each rank.
- **Fidelity.** Bit-identical means batch row 0 of the full-schedule latent
  and of a one-step latent both equal the one-GPU reference exactly.
  Otherwise the table gives the one-step NRMS and the full schedule's largest
  absolute latent difference.

Base model (K1), head capped at 2100 MHz:

| Cell | Checkpoint | Size | MP | Prompt (text rows) | One GPU | uly2 | cfg2 | uly2 vs one GPU | cfg2 vs one GPU | cfg2 vs uly2 | cfg2 against one GPU |
|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|---|
| k1-base-bf16-1024 | Base BF16 | 1024 | 1.05 | One block (80) | 29.8 | 20.0 | 15.5 | 1.49x | 1.92x | 1.29x | 1-step NRMS 0.009; full schedule max abs 1.36 |
| k1-base-bf16-1024 | Base BF16 | 1024 | 1.05 | Vendor length (480) | 31.6 | 20.5 | 16.6 | 1.54x | 1.90x | 1.23x | bit-identical |
| k1-base-bf16-1152 | Base BF16 | 1152 | 1.33 | One block (80) | 38.4 | 25.8 | 20.1 | 1.49x | 1.91x | 1.28x | 1-step NRMS 0.010; full schedule max abs 0.94 |
| k1-base-bf16-1152 | Base BF16 | 1152 | 1.33 | Vendor length (480) | 40.8 | 26.5 | 21.2 | 1.54x | 1.93x | 1.25x | bit-identical |
| k1-base-bf16-1216 | Base BF16 | 1216 | 1.48 | One block (80) | 43.3 | 28.7 | 22.3 | 1.51x | 1.94x | 1.29x | 1-step NRMS 0.010; full schedule max abs 2.34 |
| k1-base-bf16-1216 | Base BF16 | 1216 | 1.48 | Vendor length (480) | 46.0 | 29.5 | 24.1 | 1.56x | 1.91x | 1.23x | bit-identical |
| k1-base-bf16-1280 | Base BF16 | 1280 | 1.64 | One block (80) | 49.0 | 32.2 | 25.2 | 1.52x | 1.95x | 1.28x | 1-step NRMS 0.009; full schedule max abs 1.76 |
| k1-base-bf16-1280 | Base BF16 | 1280 | 1.64 | Vendor length (480) | 51.2 | 32.8 | 26.6 | 1.56x | 1.93x | 1.24x | bit-identical |
| k1-base-bf16-1536 | Base BF16 | 1536 | 2.36 | One block (80) | 76.5 | 49.5 | 39.0 | 1.54x | 1.96x | 1.27x | 1-step NRMS 0.009; full schedule max abs 3.48 |
| k1-base-bf16-1536 | Base BF16 | 1536 | 2.36 | Vendor length (480) | 79.5 | 50.7 | 41.1 | 1.57x | 1.94x | 1.23x | bit-identical |
| k1-base-nvfp4-1024 | Base NVFP4 | 1024 | 1.05 | One block (80) | 20.3 | 15.3 | 10.8 | 1.33x | 1.88x | 1.41x | 1-step NRMS 0.031; full schedule max abs 4.24 |
| k1-base-nvfp4-1024 | Base NVFP4 | 1024 | 1.05 | Vendor length (480) | 21.9 | 15.2 | 11.6 | 1.44x | 1.89x | 1.31x | bit-identical |
| k1-base-nvfp4-1152 | Base NVFP4 | 1152 | 1.33 | One block (80) | 27.5 | 19.1 | 13.9 | 1.44x | 1.97x | 1.37x | 1-step NRMS 0.027; full schedule max abs 5.57 |
| k1-base-nvfp4-1152 | Base NVFP4 | 1152 | 1.33 | Vendor length (480) | 28.6 | 19.8 | 15.1 | 1.45x | 1.90x | 1.31x | bit-identical |
| k1-base-nvfp4-1216 | Base NVFP4 | 1216 | 1.48 | One block (80) | 30.8 | 22.0 | 15.8 | 1.40x | 1.95x | 1.39x | 1-step NRMS 0.029; full schedule max abs 4.38 |
| k1-base-nvfp4-1216 | Base NVFP4 | 1216 | 1.48 | Vendor length (480) | 32.4 | 22.5 | 17.0 | 1.44x | 1.90x | 1.32x | bit-identical |
| k1-base-nvfp4-1280 | Base NVFP4 | 1280 | 1.64 | One block (80) | 35.5 | 25.0 | 17.9 | 1.42x | 1.98x | 1.39x | 1-step NRMS 0.026; full schedule max abs 3.14 |
| k1-base-nvfp4-1280 | Base NVFP4 | 1280 | 1.64 | Vendor length (480) | 36.4 | 25.4 | 18.9 | 1.43x | 1.93x | 1.34x | bit-identical |
| k1-base-nvfp4-1536 | Base NVFP4 | 1536 | 2.36 | One block (80) | 57.2 | 39.1 | 29.2 | 1.46x | 1.96x | 1.34x | 1-step NRMS 0.023; full schedule max abs 3.76 |
| k1-base-nvfp4-1536 | Base NVFP4 | 1536 | 2.36 | Vendor length (480) | 59.5 | 38.7 | 30.7 | 1.54x | 1.94x | 1.26x | bit-identical |

uly2 was bit-identical to one GPU in every cell above.

Base INT8 ConvRot (K1), rerun on 2026-10-16, head capped at 2100 MHz. The INT8
files pin an approximate attention kernel, so the one-GPU and cfg2 legs
rendered under the Sol-Attn waiver. These walls are recorded only and never
make a row. uly2 drops the pins and runs the exact kernel, so it differs from
the pinned one-GPU render by construction
([VALIDATION](VALIDATION.md#evidence-flux-3-klein)).

| Cell | Checkpoint | Size | MP | Prompt (text rows) | One GPU | uly2 | cfg2 | uly2 vs one GPU | cfg2 vs one GPU | cfg2 vs uly2 | Fidelity against one GPU |
|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|---|
| k1-base-int8-1024 | Base INT8 ConvRot | 1024 | 1.05 | One block (80) | 24.9 | 15.1 | 10.0 | 1.66x | 2.49x | 1.51x | uly2 1-step NRMS 0.021, full schedule max abs 2.50; cfg2 1-step NRMS 0.016, full schedule max abs 1.72 |
| k1-base-int8-1024 | Base INT8 ConvRot | 1024 | 1.05 | Vendor length (480) | 20.5 | 15.1 | 11.0 | 1.36x | 1.86x | 1.37x | uly2 1-step NRMS 0.031, full schedule max abs 6.29; cfg2 bit-identical |
| k1-base-int8-1152 | Base INT8 ConvRot | 1152 | 1.33 | One block (80) | 34.7 | 19.8 | 13.3 | 1.75x | 2.60x | 1.49x | uly2 1-step NRMS 0.020, full schedule max abs 5.60; cfg2 1-step NRMS 0.015, full schedule max abs 1.64 |
| k1-base-int8-1152 | Base INT8 ConvRot | 1152 | 1.33 | Vendor length (480) | 27.7 | 19.8 | 14.9 | 1.40x | 1.86x | 1.33x | uly2 1-step NRMS 0.027, full schedule max abs 6.58; cfg2 bit-identical |
| k1-base-int8-1216 | Base INT8 ConvRot | 1216 | 1.48 | One block (80) | 39.0 | 23.1 | 15.9 | 1.69x | 2.46x | 1.46x | uly2 1-step NRMS 0.022, full schedule max abs 2.43; cfg2 1-step NRMS 0.016, full schedule max abs 3.66 |
| k1-base-int8-1216 | Base INT8 ConvRot | 1216 | 1.48 | Vendor length (480) | 33.3 | 22.4 | 18.3 | 1.49x | 1.82x | 1.22x | uly2 1-step NRMS 0.028, full schedule max abs 5.37; cfg2 bit-identical |
| k1-base-int8-1280 | Base INT8 ConvRot | 1280 | 1.64 | One block (80) | 43.5 | 28.1 | 19.2 | 1.55x | 2.27x | 1.46x | uly2 1-step NRMS 0.020, full schedule max abs 3.20; cfg2 1-step NRMS 0.015, full schedule max abs 1.88 |
| k1-base-int8-1280 | Base INT8 ConvRot | 1280 | 1.64 | Vendor length (480) | 39.7 | 25.1 | 20.8 | 1.58x | 1.91x | 1.21x | uly2 1-step NRMS 0.028, full schedule max abs 5.04; cfg2 bit-identical |
| k1-base-int8-1536 | Base INT8 ConvRot | 1536 | 2.36 | One block (80) | 68.7 | 46.2 | 32.5 | 1.49x | 2.11x | 1.42x | uly2 1-step NRMS 0.020, full schedule max abs 3.66; cfg2 1-step NRMS 0.016, full schedule max abs 1.74 |
| k1-base-int8-1536 | Base INT8 ConvRot | 1536 | 2.36 | Vendor length (480) | 66.9 | 38.7 | 34.2 | 1.73x | 1.96x | 1.13x | uly2 1-step NRMS 0.027, full schedule max abs 4.83; cfg2 bit-identical |

Step-distilled model (K2, one-block prompt) and the step-distilled model at
CFG 3.5 (K3), head capped at 2100 MHz:

| Cell | Checkpoint | Size | MP | Steps, CFG | One GPU | uly2 | cfg2 | uly2 vs one GPU | cfg2 vs one GPU | cfg2 vs uly2 | Fidelity against one GPU |
|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|---|
| k2-sd-bf16-1024 | Step-distilled BF16 | 1024 | 1.05 | 16, 1 | 10.2 | 6.7 | not run | 1.53x | | | uly2 bit-identical |
| k2-sd-bf16-1152 | Step-distilled BF16 | 1152 | 1.33 | 16, 1 | 13.2 | 8.7 | not run | 1.51x | | | uly2 bit-identical |
| k2-sd-bf16-1216 | Step-distilled BF16 | 1216 | 1.48 | 16, 1 | 14.6 | 9.6 | not run | 1.53x | | | uly2 bit-identical |
| k2-sd-bf16-1280 | Step-distilled BF16 | 1280 | 1.64 | 16, 1 | 16.4 | 10.8 | not run | 1.53x | | | uly2 bit-identical |
| k2-sd-bf16-1536 | Step-distilled BF16 | 1536 | 2.36 | 16, 1 | 25.4 | 16.4 | not run | 1.55x | | | uly2 bit-identical |
| k2-sd-nvfp4-1024 | Step-distilled NVFP4 | 1024 | 1.05 | 16, 1 | 7.2 | 5.1 | not run | 1.41x | | | uly2 bit-identical |
| k2-sd-nvfp4-1152 | Step-distilled NVFP4 | 1152 | 1.33 | 16, 1 | 9.3 | 6.5 | not run | 1.43x | | | uly2 bit-identical |
| k2-sd-nvfp4-1216 | Step-distilled NVFP4 | 1216 | 1.48 | 16, 1 | 10.5 | 7.4 | not run | 1.42x | | | uly2 bit-identical |
| k2-sd-nvfp4-1280 | Step-distilled NVFP4 | 1280 | 1.64 | 16, 1 | 11.8 | 8.3 | not run | 1.42x | | | uly2 bit-identical |
| k2-sd-nvfp4-1536 | Step-distilled NVFP4 | 1536 | 2.36 | 16, 1 | 19.0 | 12.5 | not run | 1.52x | | | uly2 bit-identical |
| k3-sd-bf16-1024-cfg35 | Step-distilled BF16 | 1024 | 1.05 | 16, 3.5 | 19.4 | 13.1 | 10.2 | 1.48x | 1.91x | 1.29x | uly2 bit-identical; cfg2 1-step NRMS 0.061, full schedule max abs 3.61 |
| k3-sd-bf16-1536-cfg35 | Step-distilled BF16 | 1536 | 2.36 | 16, 3.5 | 49.2 | 31.8 | 25.2 | 1.55x | 1.95x | 1.26x | uly2 bit-identical; cfg2 1-step NRMS 0.059, full schedule max abs 3.40 |

The step-distilled edit with one 4:3 reference (8,110 joint rows, 16 steps)
ran through the driver API, head capped at 2100 MHz, one cold and two warm
renders on each side, with `mode=local` as the one-GPU side. Its sampler
walls read 13.91, 13.92 and 13.96 s on uly2 against 21.28, 20.94 and 20.95 s
on one GPU, a 1.50x warm speedup.

Reading the tables:

- **cfg2 was the fastest topology in every base BF16 and NVFP4 cell but earned
  no row.** It ran 1.88x to 1.98x faster than one GPU and 1.23x to 1.41x
  faster than uly2, and was bit-identical on the vendor-length prompt. On the one-block prompt
  one GPU batches the two calls and cfg2 cannot, so cfg2 read 1-step NRMS 0.009
  to 0.031 and no full schedule matched. A row needs both prompts
  bit-identical at each size, so auto keeps uly2
  ([DESIGN](DESIGN.md#54-topology--auto-mode)).
- **uly2 ran 1.33x to 1.57x faster than one GPU** on the base model and 1.41x
  to 1.55x on the step-distilled model, bit-identical in every BF16 and NVFP4
  cell.
- **INT8 walls are recorded only.** One GPU and cfg2 run the pinned kernel
  under the waiver and uly2 runs the exact one. cfg2 ran 1.82x to 2.60x faster
  than one GPU on the same kernel; uly2 ran 1.36x to 1.75x faster, a ratio
  across two kernels.
- **NVFP4 ran faster than BF16 here**, but it is a capacity choice: these
  cells compare NVFP4 with NVFP4 on one GPU, not with BF16.

The first two-Spark cells, run on 2026-10-15 before the hardware ladder,
timed one render each of the shipped recipes through
`benchmark/run_matrix.py`, head capped at 2100 MHz. Their walls run higher
than the warm means above.

| Cell | One GPU | uly2 | cfg2 | Speedup |
|---|--:|--:|--:|---|
| Step-distilled BF16, 1024, 16 steps, CFG 1 | 11.82 | 10.19 | n/a (CFG 1) | 1.16x uly2 |
| Step-distilled BF16, 1536, 16 steps, CFG 1 | 27.04 | 19.89 | n/a | 1.36x uly2 |
| Base BF16, 1024, 25 steps, CFG 3.5 | 31.61 | 23.77 | 15.58 | 1.33x uly2, 2.03x cfg2 |

Stock ComfyUI on one Spark (2026-10-15, head capped at 2100 MHz) ran the
vendor template settings: the step-distilled model at 15 steps with
`er_sde`, the base model at 25 Euler steps and CFG 3.5, the BF16 encoder for
the BF16 cell and the INT8 ConvRot encoder for the others. Walls are
ComfyUI's whole-prompt times in one process, cold (with loads) and warm (a
new seed). The recipe differs from the tables above, so compare within this
table only.

| Stock cell | Steps | CFG | Cold s | Warm s | Warm s per step |
|---|--:|--:|--:|--:|--:|
| Step-distilled INT8 ConvRot, 1024 | 15 | 1.0 | 9.39 | 6.48 | 0.43 |
| Step-distilled BF16, 1024 | 15 | 1.0 | 15.05 | 9.90 | 0.66 |
| Step-distilled NVFP4, 1024 | 15 | 1.0 | 10.52 | 7.00 | 0.47 |
| Base INT8 ConvRot, 1024 | 25 | 3.5 | 26.18 | 25.16 | 1.01 (two passes per step) |
| Step-distilled INT8 ConvRot, 1536 | 15 | 1.0 | 21.57 | 21.18 | 1.41 |

<a id="moss-image-one-gpu-and-uly2-2026-10-21-to-2026-10-22-head-capped"></a>

## MOSS-Image: one GPU and uly2

The MOSS-Image hardware ladder timed MOSS-Image SFT, Base and Kromoss on
2026-10-21 with dgx-monarch `ff8dab8300aa` and ComfyUI `d52a7a5ef` on both
Sparks, and timed INT8 again on 2026-10-21 and 2026-10-22 at `9caa6aae660b`,
the release with the INT8 dispatch fix. The first Spark was capped at
2100 MHz, with a highest sampled SM clock of 2093 MHz; the second was
uncapped. [VALIDATION](VALIDATION.md#evidence-moss-image-sft) records the
ladder and its comparisons, and
[Campaign K](#campaign-k-cfg2-against-uly2-2026-10-14-head-capped) describes
the method this section follows.

Method:

- **Instrument.** `benchmark/run_matrix.py` in cluster mode, one process per
  cell. Each Campaign K leg ran one cold render and two warm renders, and the
  BF16 and INT8 Campaign K tables give the cold wall and the mean of the two
  warm walls in seconds; the other tables say how many renders each wall
  holds. A sampler wall leaves out text encoding and VAE decode. Compare
  within each table.
- **One GPU.** Batch row 0 of a dp2 render of batch 2: each rank renders one
  image alone, so its wall is the capped first Spark's one-GPU wall.
- **Recipe.** 50 Euler steps on ComfyUI's `simple` scheduler, CFG 3, layer 22
  skipped on the negative pass over the whole schedule, the vendor's tag
  negative, the BF16 Qwen3.5-9B encoder and Torch Flash. The shipped graphs
  now run `normal`. One schedule computes as much as the other: in the
  scheduler record at 1024 square, one render each with `simple` first in
  the process, uly2 read 64.36 s on `simple` and 64.10 s on `normal`, and one
  GPU 94.51 s and 92.46 s.
- **Fidelity.** Bit-identical means batch row 0 of the full-schedule latent
  and of a one-step latent both equal the one-GPU reference exactly.

BF16, head capped at 2100 MHz:

| Checkpoint | Size | MP | Joint rows | One GPU cold / warm | uly2 cold / warm | uly2 vs one GPU | Fidelity |
|---|--:|--:|--:|---|---|--:|---|
| SFT | 1024 | 1.05 | 4,233 | 92.4 / 90.0 | 64.0 / 60.5 | 1.49x | bit-identical |
| SFT | 1152 | 1.33 | 5,321 | 121.5 / 119.1 | 82.7 / 79.3 | 1.50x | bit-identical |
| SFT | 1216 | 1.48 | 5,913 | 136.9 / 135.1 | 89.7 / 86.4 | 1.56x | bit-identical |
| SFT | 1280 | 1.64 | 6,537 | 153.4 / 150.9 | 102.3 / 99.0 | 1.52x | bit-identical |
| SFT | 1408 | 1.98 | 7,881 | 190.4 / 188.0 | 125.3 / 122.1 | 1.54x | bit-identical |
| SFT | 1536 | 2.36 | 9,353 | 233.0 / 230.6 | 153.9 / 150.9 | 1.53x | bit-identical |
| SFT | 2048 | 4.19 | 16,521 | 486.3 / 482.4 | 307.0 / 302.7 | 1.59x | bit-identical |
| Base | 1024 | 1.05 | 4,195 | 92.0 / 89.7 | 63.7 / 60.8 | 1.48x | bit-identical |
| Base | 1536 | 2.36 | 9,315 | 233.1 / 230.5 | 155.5 / 151.6 | 1.52x | bit-identical |
| Kromoss | 1024 | 1.05 | 4,125 | 91.8 / 88.7 | 63.3 / 59.9 | 1.48x | bit-identical |
| Kromoss | 1536 | 2.36 | 9,245 | 233.7 / 230.9 | 154.3 / 152.0 | 1.52x | bit-identical |
| SFT, empty negative | 1024 | 1.05 | 4,233 | 93.3 / 91.2 | 63.0 / 59.7 | 1.53x | bit-identical |
| SFT, empty negative | 1536 | 2.36 | 9,353 | 235.0 / 232.7 | 152.5 / 150.2 | 1.55x | bit-identical |

The joint rows are the positive pass, the salted prompt's text rows plus the
image rows. One step, one GPU against uly2: 1.86 s against 1.26 s at 1024,
4.70 s against 3.09 s at 1536 and 9.76 s against 6.14 s at 2048.

INT8 ConvRot, SFT, head capped at 2100 MHz, at `9caa6aae660b`. The INT8 files
pin an approximate attention kernel, so the one-GPU leg ran it under the
Sol-Attn waiver; uly2 drops the pins and runs the exact kernel. These walls
are recorded only and never make a row. A ratio between the two divides by
the pinned kernel, not by the exact one-GPU path, so the table gives none.

| Size | One GPU, pinned kernel, waived (warm) | uly2 (warm) | uly2 per step | uly2 against the pinned one-GPU render |
|--:|--:|--:|--:|---|
| 1024 | 48.1 | 39.6 | 0.79 s | one step NRMS 0.0396; full schedule NRMS 0.166, max abs 3.76 |
| 1536 | 162.9 | 105.6 | 2.11 s | one step NRMS 0.0344; full schedule NRMS 0.250, max abs 5.21 |

The exact one-GPU path is a copy of the file with its pins removed. At 1024
square, 50 steps and the vendor's empty negative, head capped at 2100 MHz,
renders of that copy took 52.96 to 53.48 s on one GPU (1.06 s a step) and
41.84 to 42.02 s on uly2 (0.84 s a step) across SFT, Base and Kromoss, one
render each, bit-identical. The INT8 dispatch fix made INT8 uly2 faster:

| Measurement, uly2, 50 steps, head capped at 2100 MHz | `ff8dab8300aa` | `9caa6aae660b` |
|---|--:|--:|
| INT8 testing graph (pinned file, auto picks uly2) through the API at 1024, sampler s: SFT with the INT8 encoder / Base / Kromoss | 51.66 / 51.52 / 50.62 | 39.58 / 39.55 / 39.10 |
| The same, per step | 1.01 to 1.03 s | 0.78 to 0.79 s |
| INT8 Campaign K cell at 1024, warm, SFT | 51.5 | 39.6 |
| INT8 Campaign K cell at 1536, warm, SFT | 126.0 | 105.6 |

BF16 uly2 runs 1.24 s a step at 1024 through the same API path, so INT8 uly2
now runs faster than BF16 uly2.

Other topologies, BF16, 1024 square, 50 steps, run_matrix walls in seconds,
head capped at 2100 MHz:

| Topology | Wall | Against |
|---|---|---|
| `uly2+fsdp`, prefetch depth 1 | SFT 113.70, Kromoss 115.24 | uly2 63.45 and 62.49 |
| `uly2+fsdp`, prefetch depth 2 | SFT 131.35 | uly2 72.32 in the same cell |
| cfg2 with the skip off, prompt and negative of different lengths (bit-identical) | 47.02 | one GPU 93.46, uly2 62.37 |
| cfg2 with the skip off, equal lengths (measured, not identical) | 44.52 | one GPU 91.08 |
| The same at 512 square | 13.83 | one GPU 23.65 |
| uly2 with the skip, and without it | 63.56 and 64.07 | one GPU 94.67 and 94.56 |
| `mode=local` on one GPU, no Worker | 90.12 | stock warm 92.4 to 93.9 |

Each wall in this table is one render, except in the two cfg2 rows at 1024
square, where every wall is the median of three renders, the first of them
cold.

cfg2 refuses on the shipped graphs, which carry the skip, so it has no row
([DESIGN](DESIGN.md#54-topology--auto-mode)).

Kromoss with the DMD2 LoRA at 1.0, 8 steps, CFG 1, head capped at 2100 MHz.
Finish-record walls in seconds, sampler seconds in brackets; the cold render
follows a `/free`, and the warm renders take new seeds. One GPU is the local
driver.

| Topology | 1024 cold | 1024 warm | 1536 cold | 1536 warm |
|---|---|---|---|---|
| uly2 | 6.0 (4.81) | 4.8 (4.79), 4.9 (4.84); 0.60 s a step | 13.5 (12.35) | 12.6 (12.37), 12.5 (12.46); 1.55 s a step |
| One GPU | 7.3 (7.26) | 7.3 (7.25), 7.3 (7.27); 0.91 s a step | 20.0 (19.16) | 18.8 (18.73), 18.8 (18.75); 2.34 s a step |

uly2 ran 1.51x faster than one GPU at both sizes.

Stock ComfyUI on one Spark (2026-10-21, head capped at 2100 MHz) ran the
shipped 50-step recipe on `simple`. Walls are ComfyUI's whole-prompt times,
encoder and decode included, so compare within this list only.

- SFT at 1024, cold: 95.99 s, at 1.81 s an iteration, which is 0.905 s a
  model call, because the skip's window runs the two passes as separate
  batch-1 calls.
- Warm at 1024: 92.4 to 93.9 s across SFT, Base and Kromoss, the layout graph
  and the skip off; 90.82 s on one Kromoss render.
- Warm at 1040: 95.58 s. Batch 2 at 1024: 181.75 s. Warm at 1536: 233.01 and
  233.18 s.
- One step at 1024: 8.74 to 9.08 s cold and 3.01 to 3.02 s warm; at 2048,
  12.54 s warm.
- Kromoss with a DMD LoRA, 8 steps, warm: 8.04 to 8.54 s.
- On the scheduler record's graph at 1024: 97.05 s on `simple` (cold) and
  91.13 s on `normal` (warm), in one process.

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

## INT8 GEMM: CUTLASS against the Triton kernel

Warm legs of the 2026-10-28 and 29 kernel comparison (docs/VALIDATION.md, "INT8
GEMM kernel comparison"): the same INT8 ConvRot cells on the same build, stock
comfy-kitchen CUTLASS against kan11's Triton fused-dequant GEMM, every frame
pixel-identical. Head capped at 2100 MHz; harness timing at 2 s granularity;
cold legs left out because they include model loads whose disk state differed
between the passes.

| model (INT8 ConvRot file) | layout | CUTLASS warm s | Triton warm s | change |
|---|---|---|---|---|
| MiniMax H3 guide (124 frames) | single | 1125.0 | 884.7 | +21% |
| MiniMax H3 guide (124 frames) | uly2 | 656.7 | 546.5 | +17% |
| MiniMax H3 image to video (mp4) | single | 1053.0 | 838.7 | +20% |
| MiniMax H3 image to video (mp4) | uly2 | 628.6 | 518.5 | +18% |
| Wan Animate 2 | single | 314.3 | 264.3 | +16% |
| Wan Animate 2 | uly2 | 266.3 | 200.2 | +25% |
| LTX 2.5 distilled text to video | single | 108.1 | 98.1 | +9% |
| LTX 2.5 distilled text to video | uly2 | 82.1 | 78.1 | +5% |
| Ideogram4 | single | 44.1 | 38.0 | +14% |
| Ideogram4 | uly2 | 26.0 | 26.0 | 0% |
| Ideogram4 | cfg2 | 24.0 | 22.0 | +8% |
| Chroma1-HD | single | 50.0 | 48.1 | +4% |
| Chroma1-HD | uly2 | 32.1 | 32.0 | 0% |
| Chroma1-HD | cfg2 | 26.0 | 24.0 | +8% |
| Krea2 RAW | single | 94.1 | 94.1 | 0% |
| Krea2 RAW | uly2 | 46.1 | 46.2 | 0% |
| Krea2 RAW | cfg2 | 48.1 | 48.1 | 0% |
| Qwen Image 2.1 | single | 18.0 | 18.0 | 0% |
| Qwen Image 2.1 | uly2 | 12.2 | 12.0 | +2% |
| FLUX.3 Klein | uly2 | 6.0 | 6.0 | 0% |
| FLUX.3 Klein Base | uly2 | 16.0 | 16.0 | 0% |
| MOSS-Image SFT | uly2 | 40.0 | 40.1 | 0% |
| MOSS-Image Base | uly2 | 40.0 | 40.0 | 0% |
| Kromoss | uly2 | 40.0 | 42.0 | -5% |
| Mage Flow turbo | single | 2.0 | 2.0 | 0% |
| Mage Flow turbo | uly2 | 2.0 | 2.0 | 0% |
| Mage Flow edit turbo | single | 6.0 | 6.0 | 0% |
| Mage Flow edit turbo | uly2 | 4.0 | 4.0 | 0% |

### A 15 second clip on the pair (2026-10-29)

The pair rendered a 15 second H3 image-to-video clip, the mp4 template of the
rows above at its 362-frame ceiling: 1344x768 at 24 fps, 20 steps, CFG 1,
exact attention on uly2, the pruned INT8 ConvRot DiT with the Triton kernel,
dev `50ad5e9` on both Workers, head capped at 2100 MHz. Two sessions rendered
it once each, 53:45 and 53:58 for the prompt: the identity gate's proof render
took 4:43 and 4:42 with the model load inside, sampling 45:10 and 45:11, and
text encoding, the two VAE decodes and the save the remaining 3:50. The
two-point fit from the 5 and 10 second rows had predicted about 45 minutes of
sampling on the pair.

What the cell showed about capacity:

- One Spark refuses it before loading. The loader footprint preflight charges
  112.3 GiB of activations for the 109062 packed rows on one rank and finds no
  budget for any DiT file: 44.1 GiB more free memory than the box has.
- The pair renders it once per session. After the first render the loaded
  stack leaves 56.5 GiB on the head once the driver has unloaded its VAEs, and
  the activation preflight needs 65.1 GiB (60.1 GiB per rank plus the 5 GiB
  reserve), so it refuses the second request 8.7 GiB short. A new session
  renders it cold again, as the second session did.
- The gate cannot conclude at this length. Its slab-path preflight came up
  3.8 GiB short after the proof render, so the gate forced stock residency,
  which the 124-frame rows above ran on as well, and the session paid the
  proof render for nothing. #93 carries the follow-ups.

Reported, not reproduced: a single RTX 5090 renders the same 362-frame clip at
1344x768 in 518.0 s, 8.6 minutes
([ai-muninn.com, 2026-09-03](https://ai-muninn.com/en/blog/minimax-h3-rtx5090-config-stack)),
with a different recipe: the pruned NVFP4 DiT and an NVFP4 text encoder, 14
steps, the KJNodes head-group attention and chunked feed-forward patches
(exact), torch 2.13.0+cu130 on Windows and SageAttention. The same post gives
99.6 s for 124 frames, 2208 s at coarse chunking and an out-of-memory failure
without the chunking patches. Per step that is about 37 s against the pair's
136 s at 362 frames, and 7 s against the pair's 26 s and one Spark's 42 s at
124 frames: the 5090 holds about 3.6 times the pair's pace at both lengths.

GEMM throughput across M on the families' own linears, same box and cap
(ratio is Triton over CUTLASS; TOPS at 29184 rows, the H3 per-rank count):

| linear (N x K) | ratio at M 1024 | at 4096 | at 8192 | at 29184 | CUTLASS TOPS at 29184 | Triton TOPS at 29184 |
|---|---|---|---|---|---|---|
| krea2 mlp-in (12288 x 3072) | 1.02 | 1.08 | 1.73 | 1.90 | 91.3 | 173.4 |
| krea2 attn-out (3072 x 3072) | 0.97 | 1.04 | 1.04 | 1.04 | 168.6 | 175.3 |
| krea2 mlp-out (3072 x 12288) | 1.06 | 1.42 | 1.36 | 1.36 | 130.9 | 178.4 |
| ideogram4 adaln (18432 x 512) | 1.18 | 1.01 | 0.96 | 1.50 | 52.4 | 78.4 |
| ideogram4 qkv (9216 x 3072) | 0.92 | 1.03 | 1.68 | 2.13 | 80.6 | 172.2 |
| ideogram4 attn-out (3072 x 3072) | 0.97 | 1.05 | 1.03 | 1.02 | 171.0 | 174.3 |
| chroma linear1 (15360 x 3072) | 0.95 | 1.06 | 1.78 | 2.05 | 86.1 | 176.1 |
| chroma attn-out (3072 x 3072) | 0.97 | 1.14 | 1.04 | 1.03 | 170.2 | 175.9 |
| qwen2.1 mlp-in (12288 x 3072) | 1.13 | 1.03 | 1.74 | 1.92 | 90.7 | 173.8 |
| qwen2.1 attn-out (3072 x 3072) | 0.97 | 1.13 | 1.04 | 1.05 | 169.7 | 177.8 |
| h3 mlp-in (21504 x 5376) | 0.96 | 1.39 | 2.16 | 2.51 | 74.1 | 185.6 |
| h3 attn-out (5376 x 5376) | 0.96 | 1.45 | 1.84 | 1.78 | 100.2 | 178.6 |
| h3 mlp-out (5376 x 14336) | 1.20 | 1.71 | 1.69 | 1.68 | 103.5 | 173.8 |

Below 4096 rows the two routes are within 8 percent of each other either way,
the worst reading 0.92 on the Ideogram4 qkv linear at 1024 rows; from 4096 rows
up the kernel is at or above CUTLASS on every shape but one reading (Ideogram4
adaln at 8192 rows, 0.96), and from 6144 rows up CUTLASS loses more than half
its throughput on the wide linears while the Triton kernel holds about 175 TOPS.

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

<a id="community-four-node-results-2026-10-27"></a>

## Community results: four Sparks (reported, not reproduced here)

A four-Spark owner, barichardson on the NVIDIA Developer Forums, ran DGX
Monarch v1.3.0 on four Sparks and published a full four-node matrix on
2026-10-27 in
[post 17 of the DGX Monarch thread](https://forums.developer.nvidia.com/t/dgx-monarch-dual-spark-rendering-w-comfyui/385450/17).
This section records their report with credit. This project's maintainer has
two Sparks and has not reproduced any of it, so no number here is a verified
result and no `auto` row rests on it. Their clocks, fabric and software differ
from the tables above, so compare within this section.

Setup, as reported:

| Item | Value |
|---|---|
| Sparks | Four, one of them the driver host |
| Fabric | Two 100GbE links per Spark to a Mellanox SN2100 switch; RoCE with four RoCE functions per Spark and lossless priority 3 (`NCCL_IB_TC=104`) |
| DGX Monarch | v1.3.0 (`5433e05`), native venv |
| ComfyUI | 0.42.0 (`a071826f`), clean checkout |
| torch | 2.12.0+cu132, torchvision 0.27.0+cu132 |
| NCCL | 2.29.7 |
| Python | 3.12.3 (system) |
| NVIDIA driver | 580.178.04, CUDA 13.0 ptxas |
| Clocks and swap | No GPU clock cap; swap left on |

The shipped `dgx-spark-pair` profile assumes a direct cable. On the switch,
they extended it so NCCL uses all four RoCE functions and the switch's
lossless class:

```toml
[fabric.dgx-spark-pair]
NCCL_IB_HCA = "rocep1s0f0,roceP2p1s0f1,roceP2p1s0f0,rocep1s0f1"
NCCL_IB_TC = 104
```

With it, they report balanced traffic on all four rails, about 10 to 17 GB
per rail per render. On v1.3.0, `dgxm setup --apply` wrote a new cluster.toml
without `[fabric.*]` tables, so they added the table back after each setup
run; from 1.4.0 a re-apply keeps it
([Re-running setup](CLUSTER.md#re-running-setup)).

Method, as reported:

- **Cells.** Workflows and settings came from this project's sweep,
  `benchmark/sweep`, whose cell IDs they report match the
  [campaign tables](#campaign-render-times-stills-2026-09-12-to-2026-09-28-head-capped)
  one to one. They report no public files for the Chroma1-HD fp8mixed and
  Ideogram4 bf16 cells, so their Chroma1-HD row is the bf16 cell and their
  Ideogram4 row the fp8_scaled cell.
- **Timing.** Each time is the warm leg's wall in seconds; the cold leg runs
  at the seed and the warm leg at the seed plus one. Times come in 2 s steps,
  so a 2 s gap on a short still is within noise.
- **One GPU and pair.** These columns are their own runs of the published
  cells, the pair on the driver host and one other Spark.
- **Four Sparks.** They ran `auto`, uly4, ring4 and uly2+cfg2, the layouts
  that fill world 4 at batch 1. uly2+ring2 is reachable only through `auto`.
  Where `auto` resolved to one of the explicit layouts, they reused that
  layout's result.
- **Quality check.** Each run also rendered a one-step probe and compared it
  with the one-GPU probe by NRMS. Above 0.10 counts as a failed render.

In the tables, `refused` means Monarch rejected the layout before rendering,
and `crash` means it accepted the layout and the render failed. `FAIL` after
a time marks a render whose probe missed the one-GPU probe (NRMS above 0.10);
it does not count as fastest. `not run` means no published cell exists to
reproduce, and the published video table lists only the pair's `auto`. The
Fastest on four column names the fastest four-Spark layout that rendered
correctly, and the vs one GPU and vs pair columns give its speedups over one
Spark and over the best pair result.

Their findings:

- **Models that use CFG: uly2+cfg2.** It was the fastest four-Spark layout
  for every CFG model it rendered correctly: 1.77x to 1.97x faster than the
  best pair result on video and 1.25x to 1.62x on stills. On it the large
  video models ran 2.78x to 3.18x faster than on one Spark.
- **MiniMax H3 and LTX: uly4.** Monarch refused every other four-Spark layout
  for the MiniMax H3 rows and for every LTX row but LTX 2.3 T2V, which also
  rendered on `auto` and ring4. MiniMax H3 gained about 1.6x over the pair
  and LTX 1.14x to 1.44x, because LTX renders are short.
- **Avoid `auto` at world 4.** It was refused on nine still-image rows, on
  every MiniMax H3 row and on every LTX row but LTX 2.3 T2V, and on Wan
  SCAIL2 it picked the layout with the worst output. It beat every explicit
  layout only on Ideogram4, where it resolved to uly2+ring2 and uly4 crashed.
- **Small or fast models gain nothing.** Flux Schnell, PixelDiT, PiD and Wan
  FlowRVS gained nothing from four Sparks. Flux Schnell took 20 s on four
  against 18 s on the pair, and PixelDiT's drop from 10 s to 8 s is one
  timing step.
- **Quality.** Every four-Spark render that finished matched the one-GPU
  render within the 0.10 NRMS limit, except Wan SCAIL2 on `auto`, ring4 and
  uly2+cfg2. Monarch gave no warning for those three.

The matrix covered 40 rows on four layouts, 160 results, in about 14.5 hours.

Still images, warm seconds:

| Model | One GPU | Pair uly2 | Pair cfg2 | Pair ring2 | Four auto | Four uly4 | Four ring4 | Four uly2+cfg2 | Fastest on four | vs one GPU | vs pair |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|
| Anima | 20.0 | 14.0 | 10.0 | 14.0 | refused | 8.0 | 10.0 | 8.0 | uly4 / uly2+cfg2 | 2.50x | 1.25x |
| Boogu | 46.1 | 32.2 | 24.0 | 30.1 | refused | 20.0 | 22.0 | 18.0 | uly2+cfg2 | 2.56x | 1.33x |
| Chroma1-HD | 62.1 | 40.0 | 30.0 | 40.0 | refused | 24.0 | 26.0 | 20.0 | uly2+cfg2 | 3.10x | 1.50x |
| Ernie | 48.1 | 30.0 | 26.0 | 30.0 | 18.0 | 16.0 | 18.0 | 16.0 | uly4 / uly2+cfg2 | 3.01x | 1.62x |
| Flux Schnell | 26.0 | 18.0 | not run | 18.0 | 20.0 | 20.0 | 20.0 | refused | any (tie) | 1.30x | 0.90x |
| HunyuanImage 2048 | 76.1 | 52.1 | 42.0 | 50.1 | 32.1 | 36.0 | 34.0 | 32.0 | uly2+cfg2 | 2.38x | 1.31x |
| Ideogram4 | 34.0 | 24.0 | 20.0 | 22.2 | 14.0 | crash | 16.0 | refused | auto (uly2+ring2) | 2.43x | 1.43x |
| Kandinsky5 Image | 78.1 | 50.1 | 40.1 | 48.1 | 32.0 | 30.0 | 34.0 | 26.0 | uly2+cfg2 | 3.00x | 1.54x |
| Krea2 1536 | 64.1 | 40.1 | 34.0 | 38.1 | refused | 24.0 | refused | 22.0 | uly2+cfg2 | 2.91x | 1.55x |
| Lens Turbo | 34.0 | 24.0 | 18.0 | 20.0 | refused | 16.0 | refused | 14.0 | uly2+cfg2 | 2.43x | 1.29x |
| LongCat | 30.0 | 20.0 | 16.0 | 20.0 | refused | 12.0 | 12.1 | 10.0 | uly2+cfg2 | 3.00x | 1.60x |
| OmniGen2 | 24.0 | not run | 14.0 | 16.0 | refused | refused | refused | refused | none | n/a | n/a |
| PixelDiT | 10.0 | 10.0 | not run | 10.0 | 8.0 | 8.0 | 8.0 | refused | auto / uly4 / ring4 | 1.25x | 1.25x |
| PiD 4K upscale | 32.0 | 34.1 | not run | 32.0 | 32.0 | 32.0 | 32.0 | refused | any (tie) | 1.00x | 1.00x |
| Qwen-Image | 252.3 | 196.2 | 130.1 | 178.2 | refused | 196.2 | 186.2 | 100.1 | uly2+cfg2 | 2.52x | 1.30x |
| Radiance | 88.1 | 62.1 | 44.1 | 60.1 | refused | 42.0 | 42.0 | 32.0 | uly2+cfg2 | 2.75x | 1.38x |
| Z-Image | 18.0 | 14.0 | not run | 14.0 | 10.0 | refused | 10.0 | refused | auto / ring4 | 1.80x | 1.40x |

Video, warm seconds:

| Model | One GPU | Pair auto | Four auto | Four uly4 | Four ring4 | Four uly2+cfg2 | Fastest on four | vs one GPU | vs pair |
|---|--:|--:|--:|--:|--:|--:|---|--:|--:|
| HunyuanVideo 720p | 3229.1 | 1907.5 | 1050.7 | 1060.8 | 1066.9 | 1016.8 | uly2+cfg2 | 3.18x | 1.88x |
| Wan SCAIL2 | 2754.3 | 1629.1 | 920.8 FAIL | 892.7 | 939.0 FAIL | 838.7 FAIL | uly4 | 3.09x | 1.82x |
| Wan Bernini-R | 1443.1 | 906.8 | 524.4 | 508.5 | 528.5 | 460.4 | uly2+cfg2 | 3.13x | 1.97x |
| MiniMax H3 guides | 1074.9 | 660.7 | refused | 402.4 | refused | refused | uly4 | 2.67x | 1.64x |
| MiniMax H3 guide | 1050.9 | 640.6 | refused | 388.4 | refused | refused | uly4 | 2.71x | 1.65x |
| CogVideoX 1.5 I2V | 977.1 | 616.6 | 362.4 | 374.4 | 356.4 | 336.3 | uly2+cfg2 | 2.91x | 1.83x |
| Wan 2.2 T2V fp16 | 928.6 | 604.6 | 392.3 | 346.2 | 402.4 | 334.3 | uly2+cfg2 | 2.78x | 1.81x |
| Wan 2.1 I2V | 888.5 | 542.5 | 314.3 | 318.3 | 322.3 | 288.3 | uly2+cfg2 | 3.08x | 1.88x |
| Kandinsky5 Video | 882.7 | 546.5 | 306.3 | 314.2 | 314.3 | 296.3 | uly2+cfg2 | 2.98x | 1.84x |
| Wan 2.1 T2V 14B | 872.7 | 542.4 | 314.3 | 310.3 | 322.4 | 280.2 | uly2+cfg2 | 3.11x | 1.94x |
| MiniMax H3 t2va | 838.8 | 538.4 | refused | 332.4 | refused | refused | uly4 | 2.52x | 1.62x |
| Wan 2.2 I2V | 748.7 | 468.4 | 276.3 | 266.3 | 280.2 | 246.2 | uly2+cfg2 | 3.04x | 1.90x |
| CogVideoX 1.5 T2V | 246.3 | 156.2 | 100.1 | 96.1 | 106.1 | 88.1 | uly2+cfg2 | 2.80x | 1.77x |
| LTX 2.5 guide mask | 114.1 | 90.1 | refused | 70.1 | refused | refused | uly4 | 1.63x | 1.29x |
| LTX 2.5 FLF2V | 112.1 | 88.1 | refused | 66.1 | refused | refused | uly4 | 1.70x | 1.33x |
| LTX 2.5 I2V guide | 108.2 | 86.1 | refused | 64.1 | refused | refused | uly4 | 1.69x | 1.34x |
| LTX 2.5 I2V | 104.1 | 82.1 | refused | 64.1 | refused | refused | uly4 | 1.62x | 1.28x |
| LTX 2.5 multishot | 100.1 | 78.1 | refused | 62.1 | refused | refused | uly4 | 1.61x | 1.26x |
| LTX 2.5 T2V | 96.2 | 74.2 | refused | 60.1 | refused | refused | uly4 | 1.60x | 1.23x |
| LTX 2.3 I2V AV | 94.1 | 78.1 | refused | 54.1 | refused | refused | uly4 | 1.74x | 1.44x |
| LTX 2.5 two-stage | 76.1 | 64.1 | refused | 56.1 | refused | refused | uly4 | 1.36x | 1.14x |
| LTX 2.3 T2V | 74.1 | 52.0 | 40.0 | 38.1 | 40.0 | refused | uly4 | 1.94x | 1.36x |
| Wan FlowRVS (1 step) | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | refused | any (tie) | 1.00x | 1.00x |

Wan SCAIL2's uly2+cfg2 render was the fastest, at 838.7 s, but failed the
quality check, so its speedups use uly4. LTX 2.5 two-stage on uly4 read NRMS
0.089 against one GPU, a pass, and 0.112 against the pair render, which was
itself the one borderline result of their reproduction.

Their one-GPU and pair runs reran the published campaign cells:

| Measure | Result |
|---|---|
| Cells run | 69 on the pair and 40 on one GPU, on a third Spark; none wedged, none blocked |
| Published cells compared | 109 of 117; they report no public files for the four Chroma1-HD fp8mixed cells and the four Ideogram4 bf16 cells |
| Quality | 108 PASS and 1 CHECK: LTX 2.5 two-stage on the pair's `auto`, NRMS 0.120 against the 0.10 limit |
| Speed against published | Median 5.5 percent faster; 94 of 109 cells within 10 percent and 104 within 15 percent |
| Outliers | All faster than published: Wan SCAIL2 `auto` by 31 percent, Ernie cfg2 by 28 percent and LTX 2.5 FLF2V `auto` by 19 percent |

They name their uncapped clocks as the likely reason for the faster times;
the published campaign capped the first Spark at 2100 MHz. The maintainer's
reply in the thread
([post 19](https://forums.developer.nvidia.com/t/dgx-monarch-dual-spark-rendering-w-comfyui/385450/19))
named the uncapped clocks and some slightly outdated published cells as
possible causes. The published record of the LTX 2.5 two-stage cell also
read over the floor, at one-step NRMS 0.106
([campaign video table](#campaign-render-times-video-2026-09-27-to-2026-09-28-head-capped)).

## Updating results

Record the date, software versions, hardware settings, recipe and comparison
method with each new result. Keep older measurements when they explain a
change or limitation; identify superseded results clearly.
