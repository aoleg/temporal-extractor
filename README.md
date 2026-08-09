# Temporal Extractor

Pulls high-quality stills out of low-quality video, for LoRA training data. Each still is reconstructed from a window of neighbouring frames, not a single frame - so it carries detail no single frame in the source has.

What it does:

- Scans a video, breaks it down into scenes - automatically
- For each scene, finds the best stills in it - automatically
- For each still, extracts its neighboring frames, and applies temporal reconstruction to produce a single, sharp, denoised (as much as possible) center frame - again, automatically
- So it's basically a one-liner, `extract run myvideo.mp4`, that automatically (again - automatically; no other tool exists that does it, and I really looked) outputs a number of still images for you to train on
- End result: clean, high-resolution, dataset-ready images for training your LoRA

## Quick start

Setup is [Install](#install), below. Every command writes one output folder beside the video:

```
myvideo/
  stills/            the deliverable
  contact_sheet.jpg  for review - open this first
  manifest.json      which source frames produced each still
  work/              intermediates
```

Re-running continues where it left off. An interrupted job costs nothing to resume; a finished one re-runs in under a second.

### Preview Mode - no SeedVR2 needed

Finds the frames and extracts them straight from the video, bypassing the upscaler entirely. Seconds instead of minutes, no GPU, and **it runs without SeedVR2 installed at all** - so this half of the tool works on a machine that can't run the model.

With scene detection (the default): splits the video into scenes and takes the best frames from each, in proportion to how long each scene runs.

```
extract.bat run myvideo.mp4 --preview
```

Without scene detection: the sharpest frame of every N seconds, evenly across the whole video, nothing thrown away for being dark or soft. Minimum 0.1s, in steps of 0.1.

```
extract.bat run myvideo.mp4 --interval 2 --preview
```

Either way you get the full output folder - `stills/`, a labelled contact sheet, the manifest. The stills are source frames as they are: soft, noisy, small. Use the sheet to judge *which frames you want*, not how they look.

Point it at a folder instead of a file and it previews every video inside, one after another:

```
extract.bat run D:\raw_clips --preview
```

Every video directly in `D:\raw_clips` gets its own output folder beside it, matched by extension (`.mp4`, `.mkv`, `.mov`, `.avi`, `.mpg`/`.MPG`, `.mpeg`/`.MPEG`, and a few others - case doesn't matter). Subfolders aren't descended into. One bad file doesn't stop the rest - it's reported and skipped, and the others still run.

**Folder input only works with `--preview`.** Point `run` at a folder without it and the tool refuses rather than guessing: running a whole folder through SeedVR2 unattended would mean spending GPU time on every video before you've looked at any of them, which is the exact mistake `--preview` exists to prevent. Preview the folder, look at the sheets, then restore what you want one video at a time.

### You'll need SeedVR2 for this

```
extract.bat run myvideo.mp4
```

The real thing. Same selection, but each still is reconstructed from a window of neighbouring frames through SeedVR2 instead of copied out of the video. Minutes rather than seconds, and it needs the model, the GPU and the three paths in `.env`.

`--interval` works here too, and so does every option below.

### Cherry-pick what gets upscaled

The reason Preview Mode exists. Three steps:

```
extract.bat run myvideo.mp4 --preview           1. candidates, in seconds
                                                2. open contact_sheet.jpg and
                                                   delete the stills you don't
                                                   want from stills/
extract.bat run myvideo.mp4 --upscale-stills    3. upscale only the survivors
```

Step 3 restores whatever is left in `stills/` and nothing else. Each one is written as `<name>_upscaled.png` beside its original, which stays put - so you end up with before/after pairs, and the new contact sheet shows them side by side, labelled `source` and `upscaled`.

Your deletions are the instruction. Nothing else needs saying: a still you removed is a still that doesn't get upscaled. Re-running is safe - anything already upscaled is skipped, so an interrupted job picks up where it stopped.

### Common adjustments

```
extract.bat run video.mp4 --interval 0.5             sample every half-second instead of by scene
extract.bat run video.mp4 --seconds_per_still 2      more stills
extract.bat run video.mp4 --per_scene_max 3          fewer from long takes
extract.bat run video.mp4 --resolution 1440          bigger output
extract.bat run video.mp4 --window 9                 more temporal context
extract.bat run video.mp4 --out D:\dataset\clip01    choose the output folder
extract.bat run video.mp4 --workers 12               faster scan on a big CPU
extract.bat run movie.mkv --segment 1:15:36 1:20:00 --segment 25:10 27:00
                                                      only scan these ranges, not the whole movie
```

No global "give me N stills" setting. Each scene earns stills in proportion to its length. Forty scenes, forty scenes' worth of stills. (`--interval` is the exception - there the count is just the video's length divided by the interval.)

Full command reference, memory/tiling guidance, measurements behind the defaults: [docs/docs.md](./docs/docs.md).

## The problem

`ffmpeg` plus a single-image upscaler doesn't work for this. One frame from a low-bitrate H.264 stream has blocking, smeared motion, destroyed texture. The upscaler has one damaged frame to work from and invents the rest. Train on a hundred such frames and the LoRA learns the upscaler's idea of skin and fabric, not the subject's.

The detail isn't gone, it moved. Compression destroys different information per frame - a hair strand smeared in frame 40 is often intact in frame 38. Recovering it needs several frames at once, not one.

Second problem: a three-minute clip is ~4,500 frames. Most are useless - blurred, mid-blink, black on a cut, near-duplicate. Picking fifteen good, distinct ones by hand doesn't happen in practice.

## What I looked for

No single open-source tool does this. Restoration is solved; the rest is glue.

Right architecture class: sliding-window video super-resolution, 2N+1 low-res frames in, one restored centre frame out.

- [EDVR](https://github.com/XPixelGroup/BasicSR) - ships inside BasicSR, unmaintained since 2024, depends on a torchvision module removed in 0.17. Dead end.
- [Shift-Net](https://github.com/dasongli1/shift-net) - trained for motion blur and Gaussian noise, not compression. Targets PyTorch 1.8.
- [VRT](https://github.com/JingyunLiang/VRT) - OOMs readily.
- [Bringing Old Films Back to Life](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life) - scratches and flicker on scanned celluloid. Wrong damage model.
- [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR) - right degradation pipeline, compact enough for CPU - except mmcv has no prebuilt wheels for Blackwell (sm_120), so it isn't anymore.
- [SwiftVR](https://github.com/H-oliday/SwiftVR) - well engineered, right API, but a 5B backbone built for throughput (26 FPS at 1080p on a 5090). Don't need throughput. Repo is a month old, under thirty stars.

Also: lucky imaging. Astronomers solved this exact problem in the 1970s. [Siril](https://siril.org/) / AutoStakkert grade every frame, keep the best few percent, align subpixel, stack. Invents nothing. Falls apart the moment the subject moves independently of the camera.

Landed on [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler): one-step diffusion transformer, needs five frames minimum to use temporal info at all, conservative about preserving structure, mature (quantised variants, standalone CLI). `numz` fork for practical use, [upstream](https://github.com/ByteDance-Seed/SeedVR) for reference.

## The solution

Four stages. Model restores, everything else selects and stays out of its way.

|      |           |                                                              |
| ---- | --------- | ------------------------------------------------------------ |
| 1    | `scan`    | decode once, score every frame, find scenes, measure content box |
| 2    | `select`  | choose which frames to restore                               |
| 3    | `restore` | run a window through the model, keep the centre frame        |
| 4    | `sheet`   | contact sheet and manifest                                   |

Restorer runs as a subprocess in its own virtualenv, behind `restore(frames) -> ndarray`. Nothing in the tool's own process imports torch. These models churn and their dependency trees conflict; SeedVR2's pins don't get to decide what the rest of the tool can use. Swap restorers by writing one adapter.

That separation is also why Preview Mode can run with no SeedVR2 at all: stage 3 is the only stage that touches it, and skipping it skips the entire dependency.

Stage 4 exists because the quality gate can't be automated. Diffusion restorers occasionally produce something sharp, plausible, wrong - no no-reference metric catches that reliably. Ten minutes with the contact sheet does.

## Why the stills come out better

Compression loses different detail per frame; so does motion blur. A five-frame window lets the model align and fuse them - detail in any one frame can end up in the output.

SeedVR2 is generative, not purely fusing. It rebuilds texture from a learned prior, window constraining what it's allowed to build. Some of the output is recovered, some is invented, no marker for which. Still better than single-frame: the window narrows the space of plausible reconstructions, where a single-frame upscaler has one damaged observation and has to guess harder.

## What it works best on

Original, non-upscaled sources. Real 480p/720p, heavily compressed: real temporal information, plenty of headroom.

Already-upscaled video still works, benefit shifts. Detail this tool would recover is mostly already synthesised or destroyed. On a 480p source, 3x produced hair resolving into strands, irises gaining structure, against the usual plastic smear. On a 1080p transfer of the same material, gain was visibly smaller. On that input, value is mostly the other half of the tool - scene detection, scoring, picking a varied set.

It doesn't mean you have to feed it the worst transfer you have. A good studio upscale made with proper temporal restoration techniques or a proper high-resolution digital transfer of the original film will come out better. What I mean is don't feed it with dumb upscales. The kind you are likely to find in abundance on t\*\*\*\*\*t sites.

Caveat: those numbers are variance-of-Laplacian ratios, not scale-invariant, inflated by upscaling regardless of whether real detail appeared. Direction, not measurement. Proper test: downscale a known 4K source to 480p, run the pipeline, compare against the original at matched resolution with a metric that has ground truth. Haven't done that yet.

Have both an original and an upscale of the same footage? Feed it the original.

## Requirements

- Windows, recent Python on `PATH`
- For Preview Mode, that's it - no GPU, no SeedVR2
- For restoring: an NVIDIA GPU - comfortable at 1080p on 12 GB, 1440p with long windows wants ~20 GB
- Plus a SeedVR2 checkout, its model files, and a separate virtualenv with torch for it. Pull it from here: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler, and the models from here https://huggingface.co/numz/SeedVR2_comfyUI

## Install

```
install.bat
```

Creates `.venv`, installs numpy and OpenCV, writes `.env` from the template. Fill in three paths:

```
SEEDVR2_REPO=D:\path\to\ComfyUI-SeedVR2_VideoUpscaler
SEEDVR2_PYTHON=E:\envs\seedvr2\Scripts\python.exe
MODEL_DIR=F:\checkpoints\seedvr2
```

All three are required to restore. Preview Mode never reads them, so you can leave them alone until you have SeedVR2 set up.

Checkpoint filenames default to the common SeedVR2 release names. Set `DIT_MODEL` / `VAE_MODEL` when yours differ - quantisation variants (fp16, fp8, int8, nvfp4) all have different filenames, and files get renamed in practice.

Then:

```
extract.bat doctor
```

Validates paths, lists what it finds in `MODEL_DIR` on a name mismatch, starts the restore worker to confirm the model loads.

## Running the stages separately

```
extract.bat scan video.mp4
extract.bat select video.scan.json
extract.bat restore path\to\window\ --resolution 1440
extract.bat sheet stills\ --selection video.select.json
```

Stage 1: CPU-only, multi-process, ~23s for three minutes of 1080p at 4 workers. Stage 2: instant, reads only the scan output. Stage 3: the expensive one, ~9-14s per still at 1080p on an RTX 5090, model loaded once and reused.

## Notes worth knowing

Window sizes must be 4n+1 (5, 9, 13...) and at least 5 - SeedVR2's VAE downsamples time by 4. No single-image fallback; without neighbouring frames the tool has no reason to exist.

Pillarboxing/letterboxing detected and cropped automatically. Changes which frames get picked, too - the bars carry per-frame compression noise, not a constant offset.

Sharpness scores rank frames within one video at one resolution, nothing more. Variance of Laplacian isn't scale-invariant, rises with invented noise as readily as recovered detail. Don't compare across videos, don't rank a parameter sweep by it.

`cfg_scale` defaults to 1.0 (off), not clamped. On the one-step distilled checkpoint, raising it adds high-frequency speckle, not detail - measurements in [docs/docs.md](./docs/docs.md#generation-parameters). Common advice for still images says 2.0-3.0 for richer texture; that advice looks written for the multi-step configuration. Anyone reproducing a benefit from raised `cfg_scale` on the distilled checkpoint, I'd like to see it.

## License

[MIT](LICENSE).
