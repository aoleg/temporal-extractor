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
2. **select** — pick the best N frames, deduped and spread across scenes. *(not built yet)*
3. **restore** — decode `[i-k .. i+k]`, run SeedVR2 on the whole window, keep the centre. **built**
4. **sheet** — contact sheet + manifest. *(not built yet)*

Each stage is runnable on its own from the CLI.

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
