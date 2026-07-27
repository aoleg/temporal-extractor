# temporal_extractor — command reference

Everything is invoked through `extract.bat`, which runs the project's own
virtualenv:

```
extract.bat <command> [options]
```

Commands: [`doctor`](#doctor) · [`run`](#run) · [`scan`](#scan) ·
[`select`](#select) · [`restore`](#restore) · [`sheet`](#sheet)

`run` chains scan → select → restore → sheet. The individual stages exist so each
can be inspected, tuned and swept on its own.

---

## Configuration

All machine-specific paths live in `.env` in the project root. Exactly one
variable is required.

| Variable | Required | Default |
|---|---|---|
| `SEEDVR2_REPO` | **yes** | — |
| `SEEDVR2_PYTHON` | no | `<root>/.venv-seedvr2/Scripts/python.exe` |
| `MODEL_DIR` | no | `<root>/models/seedvr2` |
| `DIT_MODEL` | no | `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors` |
| `VAE_MODEL` | no | `ema_vae_fp16.safetensors` |

`<root>` is two levels above `SEEDVR2_REPO`, matching the usual layout where the
checkout, its virtualenv and the model folder are siblings. Set the optional
variables explicitly if yours differs. Real environment variables override
`.env`.

`SEEDVR2_PYTHON` is **not** the tool's own interpreter. The tool runs without
torch on purpose; the restorer runs in a separate virtualenv so its dependency
pins cannot constrain the tool.

---

## doctor

Prints the resolved configuration, checks every path, then starts the restore
worker to confirm it actually loads. Run this after install, and any time a run
fails in a way that smells like setup.

```
extract.bat doctor
```

Exit code 1 if anything is wrong. Cheap insurance before a long job.

---

## run

The whole pipeline.

```
extract.bat run <video> [--out DIR] [--force] [stage options...]
```

| Option | Default | Meaning |
|---|---|---|
| `--out DIR` | folder beside the video, named after it | output directory |
| `--force` | off | redo everything, ignoring existing scan, selection and stills |

Accepts every option from `scan`, `select`, `restore` and `sheet` below, except
those that only make sense for a single stage (`--show_scenes`, `--title`,
`--crop_pillarbox`, the per-stage `--out` paths). Cropping to the content box is
automatic in `run` — the scan already measured it.

### Output layout

```
<out>/
  stills/            the deliverable
  contact_sheet.jpg  for review
  manifest.json      what came from where
  work/              scan.json, select.json
```

### Resume

Re-running continues where it stopped. The artifacts *are* the state: a still
that exists is done, a scan that exists is not redone. There is no progress file
to fall out of sync. Writes go to a temporary name and are renamed into place, so
a process killed mid-write leaves nothing that a later run would mistake for
finished work.

The worker only starts when there is restoring left to do, so re-running a
finished job costs under a second rather than a model load.

---

## scan

Stage 1. One decode pass; scores every frame, finds scene cuts, measures the
content box. CPU only, no model.

```
extract.bat scan <video> [--out scan.json]
```

| Option | Default | Meaning |
|---|---|---|
| `--out PATH` | `<video>.scan.json` | metadata output |
| `--scene_threshold N` | `27.0` | HSV content delta counting as a cut. Lower = more scenes |
| `--min_scene_len N` | `15` | discard cuts producing a scene shorter than N frames |
| `--no_content_box` | off | skip pillar/letterbox detection, score the full frame |
| `--show_scenes N` | `20` | how many scenes to print |
| `--quiet` | off | no progress output |

`--scene_threshold` is on the same scale as PySceneDetect's `ContentDetector`.
On typical footage the metric is strongly bimodal — real cuts score 45+, ordinary
motion under 10 — so anything in the 10–45 range gives the same answer, and the
default sits in the middle of that gap.

Cost is decode-bound: roughly 6s for 1500 frames at 480p, 25s at 1080p.

**Output** — per frame: index, timestamp, sharpness, scene delta, mean luma,
dHash, scene id. Per scene: bounds, frame count, sharpest member, mean sharpness.
Plus the video properties and content box.

---

## select

Stage 2. Decides which frames to restore, reading only the scan. Instant, so it
is cheap to re-run while tuning.

```
extract.bat select <scan.json> [--out select.json]
```

| Option | Default | Meaning |
|---|---|---|
| `--out PATH` | `<video>.select.json` | selection output |
| `--window N` | `5` | restore window size, must be 4n+1 |
| `--seconds_per_still N` | `4.0` | scene time earning one still |
| `--per_scene_max N` | `8` | ceiling per scene |
| `--hash_distance N` | `8` | minimum dHash Hamming distance between picks |
| `--min_gap SECONDS` | `0.4` | absolute floor between picks |
| `--gap_fraction F` | `0.2` | gap floor as a fraction of a scene's natural spacing |
| `--min_sharpness_frac F` | `0.35` | reject frames below this fraction of their own scene's best |
| `--min_luma N` | `24.0` | reject frames dimmer than this mean luminance; 0 disables |

### How many stills you get

There is **no global count**. Each scene earns one still per
`--seconds_per_still` of its own duration, with a floor of one and a ceiling of
`--per_scene_max`. The total is whatever the video's structure justifies.

A global ceiling is deliberately absent: set below the scene count, it silently
drops whole scenes, losing looks that exist nowhere else in the film.

Quota comes from the scene's real duration, not from how many of its frames
survive filtering — a scene full of motion blur is still as long as it is, it
simply has fewer frames to choose from.

To get more stills, lower `--seconds_per_still` or raise `--per_scene_max`.

### The filters

- **window fits the scene** — a pick is eligible only if `[i-k .. i+k]` lies
  inside one scene. Blending across a cut is meaningless.
- **weak-frame rejection** — frames dimmer than `--min_luma`, or softer than
  `--min_sharpness_frac` of **their own scene's** best, are dropped. Judged per
  scene, never absolutely: a dim scene on a long lens has its own scale.
- **spread** — each scene is divided into as many equal *time* segments as it has
  picks; the sharpest surviving frame in each is taken.
- **dedupe** — `--hash_distance`, deliberately loose. Two frames a second apart
  in a close-up may differ by a collar coming into view or a slight head turn:
  the same shot, but genuinely different training examples. This catches frames
  that are *actually* redundant — static or held — not merely similar ones.
- **adaptive gap** — floor of `--gap_fraction` of the scene's natural spacing
  (`span / picks`), never below `--min_gap`. Stops two adjacent segments both
  picking at their shared boundary, scaled per scene.

If a segment yields nothing acceptable, the shortfall is filled *within that
scene* by maximin — the candidate furthest in time from everything chosen, ties
broken by sharpness. The `filled` count in the output reports how often that
happened; a high number means the scene has less variety than its length suggests.

---

## restore

Stage 3. Restores one window and keeps the centre frame. Mostly for sweeping
generation parameters on a single window — `run` handles whole videos.

```
extract.bat restore <png_dir> [--out out.png] [--window N]
```

| Option | Default | Meaning |
|---|---|---|
| `--out PATH` | `<dir>_restored_centre.png` | output PNG; seed suffix added when `--seeds > 1` |
| `--window N` | all | use only the centred N frames (4n+1, min 5) |
| `--crop_pillarbox` | off | crop to the non-black content box first |

### Generation parameters

| Option | Default | Meaning |
|---|---|---|
| `--resolution N` | `1080` | target **short** side in px |
| `--seed N` | `42` | base seed |
| `--seeds N` | `1` | produce N variants with consecutive seeds |
| `--cfg_scale F` | `1.0` | classifier-free guidance; 1.0 is off |
| `--input_noise_scale F` | `0.0` | noise added before VAE encode |
| `--latent_noise_scale F` | `0.0` | noise added in latent space |
| `--color_correction M` | `lab` | `lab`, `wavelet`, `wavelet_adaptive`, `hsv`, `adain`, `none` |

`--resolution` sets the **short** side, so a 480p source at `--resolution 1080`
is a genuine upscale, while a 1080p source at the same setting is a 1:1
restoration pass.

**On `--cfg_scale`.** It is not clamped: only the centre frame of each window is
kept, so temporal flicker across the window is irrelevant and still-image values
are legitimate to explore. Two things to know before sweeping it:

- The reference repo wires `cfg_scale` to the model, then overwrites
  `config.diffusion.cfg.scale = 1.0` inside `upscale_all_batches` and never
  forwards the argument, so the config route is a dead end. The worker injects
  it into `runner.inference` instead — the only path that reaches the model.
- On the 7B one-step distilled checkpoint, `cfg > 1.0` measurably adds invented
  high-frequency speckle rather than detail. Noise on a flat background patch
  rose monotonically from 0.775 at cfg 1.0 to 1.259 at cfg 3.0, while true
  sharpness fell. **Laplacian sharpness rises at cfg 3.0 purely because of that
  speckle**, so ranking a sweep by sharpness will select the worst image.

Any `cfg_scale != 1.0` runs both DiT branches and roughly doubles phase 2.

### Performance and memory

| Option | Default | Meaning |
|---|---|---|
| `--attention_mode M` | `sdpa` | `sdpa`, `flash_attn_2`, `flash_attn_3`, `sageattn_2`, `sageattn_3` |
| `--vae_encode_tiled` | off | tile VAE encoding |
| `--no_decode_tiling` | off | disable VAE decode tiling |
| `--blocks_to_swap N` | `0` | offload N DiT blocks to CPU (0–36) |
| `--quiet` | off | reduce the worker's logging |

Peaks scale with output pixels × window length. Measured on a 32 GB RTX 5090:

| window | output | time | peak VRAM |
|---|---|---|---|
| 5 | 1920×1080 | 13.5s | 11.3 GB |
| 9 | 1914×1440 | 22.9s | 18.0 GB |

Encode and decode are **two separate peaks**. Decode tiling is on by default;
above ~1080p or with long windows you also need `--vae_encode_tiled`. At 1440p
with a 9-frame window, encode tiling plus content-box cropping was the difference
between running and a hard CUDA OOM.

`--blocks_to_swap` trades speed for VRAM and is a last resort; nothing in testing
has needed it on 32 GB.

`--quiet` reduces rather than removes output: the reference code emits some lines
that bypass its own debug flag.

---

## sheet

Stage 4. Contact sheet and manifest from a folder of stills.

```
extract.bat sheet <stills_dir> [--selection select.json]
```

| Option | Default | Meaning |
|---|---|---|
| `--selection PATH` | none | selection JSON; without it there are no timestamps, scenes or provenance |
| `--out PATH` | `<dir>_sheet.jpg` | contact sheet |
| `--manifest PATH` | `<dir>_manifest.json` | manifest |
| `--columns N` | `5` | grid columns |
| `--thumb_width N` | `420` | cell width in px |
| `--quality N` | `92` | JPEG quality |
| `--title TEXT` | video name + count | sheet title |

Cells are ordered by source frame and labelled with frame, timestamp, scene,
seed, originating window and output size. Cell size comes from the median aspect
so one odd-shaped still cannot stretch the grid.

The manifest records, per still, the exact source frames that produced it — the
one fact you cannot recover from the PNG later — plus the video, content box,
selection parameters and scene table.

### Still filenames

```
<video_stem>_s<scene:03d>_f<frame:06d>[_seed<n>].png
```

Defined once in `sheet.still_name()` and parsed back by `parse_still_name()`, so
a folder of stills is self-describing. Stills not matching the convention are
laid out but flagged as carrying no provenance.

---

## Notes and limits

**Window sizes** must be 4n+1 (5, 9, 13, …) because the VAE downsamples time by
4, and at least 5 so there is temporal information to exploit. There is no
single-image fallback by design.

**Sharpness numbers are not portable.** Variance of Laplacian is not
scale-invariant and rises with invented noise. The same film scored a peak of 470
at 480p and 53 at 1080p. Use it to rank frames within one video at one
resolution; never threshold on an absolute value, and never rank a generation
sweep by it.

**Pillar/letterboxing** is detected automatically and cropped before restoring.
Beyond saving the compute the bars would consume, it changes which frames get
selected — bars carry per-frame compression noise, so they are not a constant
offset. Against a full-frame scan, sharpness rank correlation was 0.981 and the
sharpest frame of one scene in four changed.
