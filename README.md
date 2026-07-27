# temporal_extractor

Extracts a small number of high-quality stills from a low-quality video, for use
as LoRA training data. Each still is reconstructed from a **window of
neighbouring frames** rather than from a single frame — that is the entire point
of the tool. There is no single-image fallback.

## The two-venv split

| | interpreter | may import torch |
|---|---|---|
| the tool | `..\.venv` | **no** |
| the restore worker | `..\.venv-seedvr2` | yes |

The restorer sits behind a narrow interface — `restore(frames) -> ndarray` — and
runs as a subprocess under its own venv. The model's dependency pins therefore
cannot dictate what the tool is allowed to use. `temporal_extractor.restore`
exports only the client; `worker.py` is never imported in-process.

Only `restore/worker.py` imports torch.

## Stages

1. **scan** — decode, score every frame, detect scene boundaries, write metadata. CPU only. **built**
2. **select** — pick the best N frames, deduped and spread across scenes. **built**
3. **restore** — decode `[i-k .. i+k]`, run SeedVR2 on the whole window, keep the centre. **built**
4. **sheet** — contact sheet + manifest. **built**

Each stage is runnable on its own from the CLI, and `run` chains all four.

## Usage

```
..\.venv\Scripts\python.exe -m temporal_extractor.cli run <video>
```

Output is one self-contained directory, named after the video unless `--out`
says otherwise:

```
<video_stem>/
  stills/            the deliverable
  contact_sheet.jpg  for review
  manifest.json      what came from where
  work/              scan.json, select.json
```

One directory to move or delete, and nothing orphaned if a run dies partway.

### Resume

Re-running picks up where it stopped. Resume state *is* the artifacts: a still
that exists is a still that is done, a scan that exists need not be redone. There
is no progress file to fall out of sync with reality. Everything is written to a
temporary name and renamed into place, so a process killed mid-write leaves no
half-file that a later run would mistake for finished work.

The worker is only started when there is restoring left to do, so a fully
resumed run does not pay ~10s of model materialisation to discover it has
nothing to do — a complete re-run of a finished job takes **0.8s**. Use `--force`
to redo everything.

## Still filename convention

```
<video_stem>_s<scene:03d>_f<frame:06d>[_seed<n>].png
```

Defined once in `sheet.still_name()` and parsed back by `sheet.parse_still_name()`.
Stage 4 recovers scene, source frame and seed from the name, so a folder of
stills is self-describing even without the selection JSON — though with it, the
sheet and manifest also carry timestamps and full source-frame provenance.

## Stage 1: scan

```
..\.venv\Scripts\python.exe -m temporal_extractor.cli scan <video> [--out <json>]
```

One decode pass produces everything downstream needs, so nothing re-decodes the
video until stage 3 wants real pixels. Per frame it records index, timestamp,
variance-of-Laplacian sharpness, scene-change delta, dHash and scene id; per
scene, the bounds and the sharpest member.

**Scene detection** is the mean absolute difference of the HSV channels between
consecutive frames — the same idea as PySceneDetect's `ContentDetector`, and
`--scene_threshold` is on the same scale (default 27.0) — computed inline on
frames we are already decoding rather than pulling in `scenedetect` and a second
decode pass. Hue is compared circularly so a red-to-red transition does not read
as a cut. On the sample footage the metric is sharply bimodal (real cuts at
46–69, everything else below 9), so any threshold from 10 to 45 gives the same
answer; 27.0 sits in the middle of that gap. Two encodes of the same film at
480p and 1080p yield identical scene boundaries.

**Content box** detection is on by default, taking the union of non-matte area
over 60 frames spread through the video — one frame is not enough, because fades
and dark shots under-report it. Sharpness is then measured on picture only. This
is not merely a compute saving: measured against a full-frame scan, rank
correlation is 0.981, and the sharpest frame of one scene out of four changed.
The bars carry per-frame compression noise, so they are not a constant offset.

**dHash** is what gives stage 2 diversity, and scene id alone would not be
enough: within a single 35-second take, two frames 9 seconds apart scored a
Hamming distance of 41, further apart than two frames from *different* scenes
(27). Adjacent frames score 0, so slow pans dedupe cleanly.

Scan cost is decode-bound: ~6s for 1517 frames at 848×480, ~25s for 1536 frames
at 1920×1080.

### A warning about the sharpness numbers

Variance of Laplacian is **not** scale-invariant and **rises with invented
noise**. The same film scored a peak of 470 at 480p and 53 at 1080p. Use it to
rank frames within one video at one resolution; never threshold on an absolute
value, and never rank a generation sweep by it — at `cfg_scale` 3.0 it reports
the *noisiest* output as the sharpest.

## Stage 2: select

```
..\.venv\Scripts\python.exe -m temporal_extractor.cli select <scan.json> [--count 15]
```

Reads the scan and writes a selection JSON. Never touches the video, so it runs
instantly and is cheap to re-run while tuning.

### How many stills

**There is no global count.** Each scene earns one still per
`--seconds_per_still` (default 4.0) of its own duration, with a floor of one and
a ceiling of `--per_scene_max` (default 8). The total is whatever the video's
structure justifies.

A global ceiling was deliberately rejected: set it below the scene count and
whole scenes vanish silently, losing looks that exist nowhere else in the film.
Scene length has to matter too — a 35-second take in which a character turns
through several angles is worth more stills than a 7-second insert, and those
angles are the point of the exercise.

Quota is computed from the scene's **real duration**, not from how many of its
frames survive filtering. They answer different questions: a scene with heavy
motion blur is still as long as it is, it simply has fewer frames to choose from.
Sizing quota off the survivors collapsed a whole scene to a single still.

### The filters

- **window fits the scene** — a pick is only eligible if `[i-k .. i+k]` lies
  inside one scene. Blending across a cut is meaningless, so stage 3 must never
  be handed a window that straddles one.
- **weak-frame rejection** — the output is training data, so frames dimmer than
  `--min_luma` (default 24) or softer than `--min_sharpness_frac` (default 0.35)
  of **their own scene's** best are dropped outright. Judged per scene, never
  against an absolute number: a dim scene on a long lens has its own scale.
- **spread** — each scene is divided into as many equal **time** segments as it
  has picks, and the sharpest surviving frame is taken from each.
- **dedupe** — `--hash_distance` (default 8), deliberately loose. Two frames a
  second apart in a close-up may differ by a collar coming into view or a slight
  head turn: visually "the same shot", but genuinely different training
  examples. This filter exists to catch frames that are *actually* redundant —
  static shots, held frames — not to enforce variety.
- **adaptive gap** — a floor of `--gap_fraction` (default 0.2) of the scene's own
  natural spacing (`span / picks`), never below `--min_gap` (default 0.4s). A
  backstop against two adjacent segments both picking at their shared boundary,
  scaled per scene rather than one number imposed on a 7s insert and a 35s take
  alike.

Each of these earned its place by fixing an observed failure:

- Ranking a whole scene by sharpness put **six of seven picks inside one 5-second
  span** of a 35-second take.
- Segmenting the *surviving-frame list* rather than the scene's **time span** put
  two picks **half a second apart**, because weak-frame rejection had left the
  survivors bunched into one part of the scene. Equal slices of the list were
  not equal slices of the scene.
- When a segment is starved, the shortfall is filled *within that scene* by
  maximin — the candidate furthest in time from everything chosen, ties broken by
  sharpness. Filling by sharpness alone put the replacement right beside an
  existing pick. `filled` reports how often this fires.

Cross-check: two encodes of the same film at 480p and 1080p, scanned and
selected independently, agree on **14 of 15** picks to within one second.

## Stage 3: restore

```
..\.venv\Scripts\python.exe -m temporal_extractor.cli restore <png_dir> [options]
```

Sweep three seeds at 1440p on a pillarboxed source:

```
..\.venv\Scripts\python.exe -m temporal_extractor.cli restore samples\window_480_37s ^
    --window 9 --resolution 1440 --crop_pillarbox --vae_encode_tiled --seeds 3
```

The worker is started once and reused for every window and every seed. Model
materialisation dominates a single call (~10s of a ~13s 1080p window), so a warm
worker is worth roughly 25% per subsequent window and far more across a sweep.

## Window sizes

Windows must be **4n+1** (5, 9, 13, …) because the VAE downsamples temporally by
4, and at least 5 frames so there is temporal information to exploit.

## Generation parameters

`resolution` (target **short** side), `seed`, `cfg_scale`, `input_noise_scale`,
`latent_noise_scale`, `color_correction` — all per call, none hardcoded.

### cfg_scale

Defaults to **1.0 (off)**, and is **not clamped** — because only the centre frame
of each window is kept, temporal flicker across the window is irrelevant, so
still-image values are legitimate to explore.

Two things worth knowing before sweeping it:

- The reference repo wires `cfg_scale` all the way to the model, then
  `upscale_all_batches()` overwrites `config.diffusion.cfg.scale = 1.0` on entry
  and never passes `cfg_scale` to `inference()`. Setting the config is therefore
  silently discarded; the worker injects the argument into `runner.inference`
  instead, which is the only route that reaches the model.
- Measured on the 7B one-step distilled checkpoint, `cfg > 1.0` adds invented
  high-frequency speckle rather than recovering detail. Noise on a flat
  background patch climbed monotonically (0.775 → 1.259 from cfg 1.0 → 3.0) while
  true sharpness fell. **Laplacian sharpness rises at cfg 3.0 purely because of
  that speckle** — ranking a sweep by sharpness will pick the worst image.

Any `cfg_scale != 1.0` runs both DiT branches and roughly doubles phase 2.

## Stage 4: sheet

```
..\.venv\Scripts\python.exe -m temporal_extractor.cli sheet <stills_dir> --selection <select.json>
```

Writes a contact sheet JPEG and a manifest JSON.

The **contact sheet** is for manual review, so every cell is labelled with what
you need to act on it: source frame, timestamp, scene, seed, the window it came
from, and output dimensions. Cells are ordered by source frame, and sized from
the median aspect so one odd-shaped still cannot stretch the grid.

The **manifest** records, per still, the exact source frames that produced it —
the one fact you cannot recover by looking at the PNG later — plus the video,
content box, selection parameters and scene table.

Both degrade gracefully. Without `--selection` you still get a sheet, minus
timestamps and provenance; stills whose names do not match the convention are
laid out but flagged as carrying no provenance.

It is worth actually looking at the sheet rather than trusting the selection
metrics — the first 15-still run put a nearly black frame in the set, which no
sharpness number flagged. That is what drove weak-frame rejection into stage 2.

`--seeds N` (default 1) produces N variants per still with consecutive seeds, all
appearing on the one sheet. `pipeline.choose_variants()` is the seam for a future
"keep only the sharpest variant" mode: it receives a pick's variants and returns
those to keep, and the sheet and manifest follow whatever it returns.

## Memory

Peaks scale with output pixels × window length. Measured on a 32GB RTX 5090:

| window | output | time | peak VRAM |
|---|---|---|---|
| 5 | 1920×1080 | 13.5s | 11.3 GB |
| 9 | 1914×1440 | 22.9s | 18.0 GB |

Above ~1080p or with long windows you need `--vae_encode_tiled`; encode and
decode are two separate peaks. At 1440p with a 9-frame window, encode tiling plus
`--crop_pillarbox` was the difference between running and a hard OOM.

`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` is set by the worker before
torch is imported. Without it the allocator fragments during VAE decode and
spills to system RAM — measured at ~8GB paged, turning a 3s decode into 100s.

## Configuration

Paths come from `config.py` and can be overridden by environment variable:
`VIDSTILLS_SEEDVR2_PYTHON`, `VIDSTILLS_SEEDVR2_REPO`, `VIDSTILLS_MODEL_DIR`,
`VIDSTILLS_DIT_MODEL`, `VIDSTILLS_VAE_MODEL`.
