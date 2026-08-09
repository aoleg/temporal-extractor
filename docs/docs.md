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

All machine-specific settings live in `.env` in the project root.

| Variable | Required | Meaning |
|---|---|---|
| `SEEDVR2_REPO` | **yes** | your SeedVR2 checkout — the folder containing `src/` and `configs_7b/` |
| `SEEDVR2_PYTHON` | **yes** | the `python.exe` of the virtualenv that has torch |
| `MODEL_DIR` | **yes** | the folder holding the DiT checkpoint and the VAE |
| `DIT_MODEL` | no | DiT checkpoint filename |
| `VAE_MODEL` | no | VAE checkpoint filename |

The three paths are **required and never guessed**. They are three independent
locations, and nothing about installing SeedVR2 implies they sit near each other
— deriving one from another would only produce confident wrong answers.

`SEEDVR2_PYTHON` is **not** the tool's own interpreter. The tool runs without
torch on purpose; the restorer runs in a separate virtualenv so its dependency
pins cannot constrain the tool.

The two checkpoint names are optional and default to the filenames the common
SeedVR2 release ships:

```
DIT_MODEL=seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors
VAE_MODEL=ema_vae_fp16.safetensors
```

Set them whenever yours differ, which is often. Quantisation variants — fp16,
fp8, int8, nvfp4, and whatever later releases add — all carry different
filenames, and files get renamed in practice. If a named checkpoint is not
present, `doctor` lists what it actually found in `MODEL_DIR`:

```
PROBLEMS:
  - DIT_MODEL 'seedvr2_ema_7b_nvfp4.safetensors' is not in D:\models\seedvr2.
        Set DIT_MODEL in .env to the filename you actually have.
        available: ema_vae_fp16.safetensors
                   seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors
```

Real environment variables override `.env`.

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
usage: temporal_extractor run [-h] [--out OUT] [--force] [--no_restore]
                              [--scene_threshold SCENE_THRESHOLD] [--min_scene_len MIN_SCENE_LEN]
                              [--no_content_box] [--workers WORKERS] [--segment FROM TO]
                              [--window WINDOW] [--seconds_per_still SECONDS_PER_STILL]
                              [--per_scene_max PER_SCENE_MAX] [--hash_distance HASH_DISTANCE]
                              [--min_gap SECONDS] [--gap_fraction GAP_FRACTION]
                              [--min_sharpness_frac MIN_SHARPNESS_FRAC] [--min_luma MIN_LUMA]
                              [--max_highlight_frac MAX_HIGHLIGHT_FRAC] [--resolution RESOLUTION]
                              [--seed SEED] [--seeds SEEDS] [--cfg_scale CFG_SCALE]
                              [--input_noise_scale INPUT_NOISE_SCALE]
                              [--latent_noise_scale LATENT_NOISE_SCALE]
                              [--color_correction {lab,wavelet,wavelet_adaptive,hsv,adain,none}]
                              [--attention_mode {sdpa,flash_attn_2,flash_attn_3,sageattn_2,sageattn_3}]
                              [--vae_encode_tiled] [--no_decode_tiling]
                              [--blocks_to_swap BLOCKS_TO_SWAP] [--quiet] [--columns COLUMNS]
                              [--thumb_width THUMB_WIDTH] [--quality QUALITY]
                              video
```

`run` accepts almost every option from the individual stages — all of `scan`'s
and `select`'s knobs, all the generation and memory knobs from `restore`, and the
sheet's layout options. Each behaves exactly as documented in its own section
below.

| Option | Default | Meaning |
|---|---|---|
| `--out DIR` | folder beside the video, named after it | output directory |
| `--force` | off | redo everything, ignoring existing scan, selection and stills |
| `--no_restore` | off | capture the picks straight from the video instead of restoring them |

The few that are not forwarded only make sense for a single stage:
`--show_scenes` and `--title` (display only), the per-stage `--out`, `--manifest`
and `--selection` paths (the run layout defines them), and `--crop_pillarbox` —
cropping to the content box is automatic in `run`, since the scan already
measured it.

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

### `--no_restore` — review the picks before paying for them

Restoring is the expensive part: seconds of GPU per still, minutes for a whole
video. `--no_restore` replaces stage 3 with a straight capture of each pick's
centre frame — the same frame the restorer would have been centred on, cropped to
the same content box, written under the same filename.

```
extract.bat run <video> --no_restore
```

Everything else is produced exactly as usual: `work/scan.json`,
`work/select.json`, `stills/`, `contact_sheet.jpg` and `manifest.json`. What you
get is the tool's choice of frames, at source resolution, in seconds instead of
minutes, and without needing a working SeedVR2 install or a GPU at all. It is the
fastest way to answer "are these the right frames?" before committing to the
restore, and to tune the selection knobs against something you can look at.

The stills are byte-identical to the source frames, so the contact sheet shows
the footage as it is — soft, noisy, small. Judge the *choice* of frames on it,
not their quality.

The manifest records, per still, whether it was actually restored:

```json
{ "file": "clip_s000_f000088.png", "frame": 88, "restored": false }
```

That record — not the flag you passed — is what later runs trust, because both
modes write to the same filenames. A restore run that finds un-restored stills in
its output directory stops with a message rather than quietly overwriting the
previews you are reviewing; delete them or pass `--force`. A `--no_restore` run
over a directory of real stills leaves them alone and keeps them marked restored.
Manifests written before this field existed read as restored, which is what they
were.

`--seeds` is ignored under `--no_restore`: decoding a frame is deterministic, so N
variants would be N identical files.

Normal resume rules still apply, which is worth knowing if you delete previews you
did not like: the picks live in `work/select.json`, so re-running `--no_restore`
captures the missing ones again, identically. Deleting is not yet a way to narrow
what gets restored.

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
| `--workers N` | `4` | processes to scan with |
| `--segment FROM TO` | none (whole video) | only scan this time range; repeatable |
| `--show_scenes N` | `20` | how many scenes to print |
| `--quiet` | off | no progress output |

`--scene_threshold` is on the same scale as PySceneDetect's `ContentDetector`.
On typical footage the metric is strongly bimodal — real cuts score 45+, ordinary
motion under 10 — so anything in the 10–45 range gives the same answer, and the
default sits in the middle of that gap.

### `--segment` — scan only part of the video

For a feature-length source where only a handful of scenes are usable for LoRA
training, scanning the full runtime to keep three of them is wasted decode.
`--segment FROM TO` restricts the scan (and everything downstream — select,
restore, sheet all just work with whatever the scan produced) to one or more
explicit time ranges, given as a closed interval `[FROM, TO]`:

```
extract.bat scan movie.mkv --segment 1:15:36 1:20:00 --segment 25:10 27:00
```

Repeat `--segment` for multiple ranges; each accepts `H:MM:SS[.sss]`,
`MM:SS[.sss]` or bare `SS[.sss]` (`36` means 36 seconds, same as `00:00:36`).
Omitting `--segment` entirely scans the whole video, exactly as before this
option existed.

Each segment gets its own independent set of scenes — a scene never spans a
segment boundary, because the frames on either side are not adjacent in the
source and a restore window blending across that gap would be meaningless.
Frame numbers and timestamps in the output (`scene.start_i`/`end_i`,
`frame.i`/`t`, pick windows) are the real numbers from the source video, so a
pick can still be traced back and decoded correctly by later stages. Overlapping
or adjacent segments are merged automatically. A segment reaching past the
video's actual end is not an error by itself — only a segment that decodes zero
frames is, since the container's declared length is a hint and is regularly
wrong.

### Speed and `--workers`

Decoding is not the bottleneck — decode alone runs at 368 fps on 1080p, while
the stage once managed 53.5. The cost was our own per-frame arithmetic, and
OpenCV threads it barely at all (1.25× from 1 to 24 threads), so the cores are
used by splitting the video into frame ranges across processes.

Measured on 4816 frames of 1920×1080:

| workers | time | speedup |
|---|---|---|
| 1 | 47.6s | 1.00× |
| 2 | 36.6s | 1.30× |
| **4** (default) | **23.1s** | **2.06×** |
| 8 | 17.4s | 2.73× |
| 12 | 16.5s | 2.88× |

Against the original implementation the same scan went from 90.1s to 23.1s —
**3.9× end to end** at the default. The default of 4 assumes an ordinary
machine; raise it if you have the cores. Returns flatten past about 12, and 24
was slower than 12 through decode contention, so there is little point going
higher. Short videos fall back to a single process automatically, and the value
is clamped to your CPU count.

Chunked output is verified **bit-identical** to a serial scan — frame indices,
sharpness, scene deltas across every chunk boundary, and every dHash. Each worker
decodes one frame of lead-in before its range so the scene delta at a boundary is
computed against its true predecessor; without that, every boundary would report
a delta of zero and a cut landing there would be missed.

The content-box pre-pass is a separate, unparallelised cost — about 3.7s for 61
sampled seeks — because the box must be known before any frame can be scored.

### Progress

A bar reports percent, frames, throughput and time remaining, refreshed on a
time interval rather than a frame count so its cost is constant at any speed:

```
  scanning  [###############-------------]  56%  2,704/4,816  243 fps  eta 0:08
```

It degrades in two situations. Container frame counts are hints and are regularly
wrong, so if the count is missing or the scan passes it, the bar stops quoting a
percentage and becomes a spinner with a running frame count rather than claiming
104%. And when output is redirected rather than going to a terminal, it switches
from `\r` rewriting to sparse newline updates so logs stay readable.

**Output** — per frame: index, timestamp, sharpness, scene delta, mean luma,
highlight fraction (share of the frame that is blown-out near-white), dHash,
scene id. Per scene: bounds, frame count, sharpest member, mean sharpness.
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
| `--max_highlight_frac F` | `0.20` | reject frames with more than this fraction blown out; 0 disables |

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
- **weak-frame rejection** — frames dimmer than `--min_luma`, blown out beyond
  `--max_highlight_frac`, or softer than `--min_sharpness_frac` of **their own
  scene's** best, are dropped. Sharpness is judged per scene, never absolutely:
  a dim scene on a long lens has its own scale.
- **`--max_highlight_frac`** judges exposure by the *fraction of clipped
  pixels*, not mean brightness — measured on real footage, a frame with a
  genuinely overexposed backdrop and a frame with an intact, merely bright one
  had almost the same mean luma (the overexposed one was not even the
  brighter of the two), while the clipped-pixel fraction separated them
  cleanly (~40% vs ~0%). A `--min_luma`-style mean ceiling would have missed
  this or misfired on legitimately bright shots.
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
