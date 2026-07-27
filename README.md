# temporal_extractor

An automated tool for extracting high-quality stills from video, for use as
**LoRA training data**.

The stills come out at higher quality and higher resolution than the source video
provides. That is possible because each still is reconstructed from a **window of
neighbouring frames** rather than from a single frame: detail that is destroyed
in any one frame by compression, noise or motion often survives in its
neighbours, and a multi-frame restoration model can recover it. This is
essentially the same temporal restoration technique behind many commercial
upscales.

The tool also does the tedious part: it watches the whole video, finds the scene
boundaries, scores every frame, throws away the blurred and the near-black, and
picks a spread of sharp, genuinely different shots — then hands you a contact
sheet to choose from.

## What it works best on

**Original, non-upscaled sources.** A genuine 480p or 720p transfer, heavily
compressed, is the ideal input: the temporal information is real, and there is
plenty of headroom to recover. Measured on a 480p source, a 3× upscale produced
detail no single-frame method can reach — hair resolving into separate strands,
irises gaining structure, skin gaining texture rather than the usual plastic
smear.

**Already-upscaled video still works, but the benefit shifts.** If the source has
already been through an upscaler, the detail the model would recover has largely
been synthesised or destroyed already, and there is little headroom left. On a
1080p transfer the same pipeline gained 2.8× sharpness against 7.5× on the 480p
one. On such material the value is mostly in the *other* half of the tool —
scene detection, frame scoring and picking the best, most varied shots — rather
than in enhancing a poor source.

If you have both an original and an upscale of the same footage, feed it the
original.

## Requirements

- Windows, a recent Python on `PATH` (no specific version required)
- An NVIDIA GPU. Comfortable at 1080p on 12 GB; 1440p with long windows wants
  closer to 20 GB
- A [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) checkout,
  its model files, and **a separate virtualenv with torch installed** for it

That last point is deliberate. The restorer runs as a subprocess in its own
virtualenv, behind a narrow `restore(frames) -> ndarray` interface, so the
model's dependency pins cannot dictate what this tool is allowed to use. Nothing
in the tool's own process imports torch.

## Install

```
install.bat
```

Creates a virtualenv in `.venv`, installs numpy and OpenCV, and writes a `.env`
from the template. Then open `.env` and fill in three paths:

```
SEEDVR2_REPO=D:\path\to\ComfyUI-SeedVR2_VideoUpscaler
SEEDVR2_PYTHON=E:\envs\seedvr2\Scripts\python.exe
MODEL_DIR=F:\checkpoints\seedvr2
```

All three are required and none is guessed from the others — they are
independent locations, and installing SeedVR2 says nothing about where you keep
its virtualenv or its checkpoints.

Checkpoint **filenames** are optional and default to the names the common SeedVR2
release ships. Set `DIT_MODEL` / `VAE_MODEL` whenever yours differ, which is
often: quantisation variants (fp16, fp8, int8, nvfp4, and whatever comes next)
all have different filenames, and files get renamed in practice.

Then check your setup:

```
extract.bat doctor
```

It validates every path, lists what it actually finds in `MODEL_DIR` if a
checkpoint name does not match, and starts the restore worker to confirm it
loads.

## Use

```
extract.bat run myvideo.mp4
```

That is the whole thing. Output lands in a folder beside the video:

```
myvideo/
  stills/            the deliverable
  contact_sheet.jpg  for review
  manifest.json      which source frames produced each still
  work/              intermediates
```

Re-running continues where it left off, so an interrupted job costs nothing to
resume and a finished one re-runs in under a second.

### Common adjustments

```
extract.bat run video.mp4 --seconds_per_still 2      more stills
extract.bat run video.mp4 --per_scene_max 3          fewer from long takes
extract.bat run video.mp4 --resolution 1440          bigger output
extract.bat run video.mp4 --window 9                 more temporal context
extract.bat run video.mp4 --out D:\dataset\clip01    choose the output folder
extract.bat run video.mp4 --workers 12               faster scan on a big CPU
```

There is no global "give me N stills" setting. Each scene earns stills in
proportion to its own length, so the total is whatever the video's structure
justifies — a film with forty scenes will not silently drop thirty of them to
satisfy a number.

### The individual stages

Each stage runs on its own, which is how the pipeline is meant to be tuned and
debugged:

| | | |
|---|---|---|
| 1 | `scan` | decode once, score every frame, find scenes, measure the content box |
| 2 | `select` | choose which frames to restore — sharp, spread out, not redundant |
| 3 | `restore` | run a window through the model, keep the centre frame |
| 4 | `sheet` | contact sheet and manifest |

```
extract.bat scan video.mp4
extract.bat select video.scan.json
extract.bat restore path\to\window\ --resolution 1440
extract.bat sheet stills\ --selection video.select.json
```

Stage 1 is CPU-only and runs across several processes — about 23 seconds for
three minutes of 1080p at the default 4 workers, with a progress bar and ETA.
Raise `--workers` if you have the cores. Stage 2 is instant and reads only the
scan, so it is cheap to re-run while tuning. Stage 3 is the expensive one —
roughly 9–14 seconds per still at 1080p on an RTX 5090, with the model loaded
once and reused.

## Full command reference

**[docs/docs.md](docs/docs.md)** — every command, every option, the memory and
tiling guidance, and the measured behaviour behind the defaults.

## Notes worth knowing

**Window sizes** must be 4n+1 (5, 9, 13, …) and at least 5. SeedVR2 is a
multi-frame model and the VAE downsamples time by 4. There is no single-image
fallback: without neighbouring frames the tool has no reason to exist.

**Pillar/letterboxing** is detected and cropped automatically. Beyond saving the
compute, it changes which frames get picked — the bars carry per-frame
compression noise, so they are not a constant offset.

**Sharpness scores rank frames within one video at one resolution.** Variance of
Laplacian is not scale-invariant and rises with invented noise. Don't compare the
numbers across videos, and don't rank a parameter sweep by them.

**`cfg_scale` defaults to 1.0 (off)** and is not clamped. On the one-step
distilled checkpoint, raising it adds high-frequency speckle rather than detail.
See [docs/docs.md](docs/docs.md#generation-parameters) for the measurements.
