"""
Quick preview: N frames spread across a video, a contact sheet, and nothing else.

This is a deliberately separate path from `run`'s four stages, not a mode of
them. `run --preview` with no other options never scans the whole video, never
writes `work/`, and never touches `stills/` -- it decodes one second of footage
at each of N points, keeps the sharpest frame of each, and lays them out. On a
feature film that is a few hundred frames decoded instead of a few hundred
thousand.

The convention it follows is the one every video-contact-sheet tool uses (vcsi
defaults to a 4x4 grid, HandBrake's scan to 10 previews): a *fixed* number of
frames spread across the whole duration, so a 3-minute clip and a 3-hour film
both produce one screenful. That is the opposite of what the real pipeline does
-- there, output scales with the material, because it is training data and
missing a scene loses a look that exists nowhere else. Here the goal is only
"show me what this video looks like", and one screenful is the whole point.

Output goes in its own directory:

    <out>/preview/
      <stem>_f000123.png    the frames
      contact_sheet.jpg     the deliverable
      manifest.json         which frame and timestamp each came from

Separate from `stills/` on purpose. A preview is not an extraction: mixing the
two would put frames nobody selected in the folder `--upscale-stills` reads, and
would force a caching story onto a path whose whole appeal is that it is cheap
enough to just redo. It is redone every time, and the directory is cleared
first, so what you see is always this run's output.

Note also that the filenames here carry no `_s<scene>_` part, so
`parse_still_name()` does not match them: even if one were copied into `stills/`
it could not be mistaken for a pick.
"""

from pathlib import Path

import cv2
import numpy as np

from .frames import content_box_from_projection, write_png_rgb
from .scan import CONTENT_BOX_SAMPLES
from .sheet import build_contact_sheet, build_manifest, write_manifest, write_sheet

# What `--preview` alone means. In the range every comparable tool sits in
# (vcsi 16, HandBrake 10), chosen at the top of it because the contact sheet
# defaults to 5 columns and 20 fills four rows exactly -- 16 would leave a
# ragged last row of one.
DEFAULT_PREVIEW_COUNT = 20

# How much footage to look at per sample point. Long enough that the sharpest
# frame in it is meaningfully better than an arbitrary one (25-30 frames to
# choose from), short enough that N of them is still a trivial amount of decode.
PREVIEW_WINDOW_S = 1.0


def _sharpness_bgr(frame) -> float:
    """Variance of the Laplacian, computed exactly as scan.py's `_measure` does.

    Not `frames.sharpness()`: that one takes RGB and CV_64F, and these frames
    come off VideoCapture as BGR. CV_32F is the same measurement for half the
    cost -- scan.py measured Spearman 1.00000 against CV_64F, same sharpest
    frame, same top ten -- and matching it keeps the two definitions of "sharp"
    in this codebase from drifting apart.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _sample_windows(frame_count: int, fps: float, count: int) -> list:
    """
    N windows of PREVIEW_WINDOW_S, spread evenly, as (start, stop) frame pairs.

    Windows sit at the *centre* of each of N equal slices rather than at the
    start: a video's first frame is very often black or mid-fade, and starting
    slice 0 at frame 0 would put that on every sheet.

    When the slices are shorter than the window (a short video, or a large N),
    the window shrinks to the slice so the samples stay disjoint -- overlapping
    windows would pick the same frame twice and quietly return fewer than N.
    """
    span = frame_count / count
    width = min(PREVIEW_WINDOW_S * fps, span)
    windows = []
    for k in range(count):
        centre = (k + 0.5) * span
        start = int(max(0, min(frame_count - 1, centre - width / 2)))
        stop = int(min(frame_count, max(start + 1, centre + width / 2)))
        windows.append((start, stop))
    return windows


def quick_preview(video, out_dir=None, *, count: int = DEFAULT_PREVIEW_COUNT,
                  sheet_opts=None, log=print) -> dict:
    """
    Decode N one-second windows, keep the sharpest frame of each, sheet them.

    Returns the manifest. `count` is exactly how many frames end up on the sheet,
    short of the video not having that many distinct ones to give.
    """
    video = Path(video).resolve()
    out_dir = Path(out_dir) if out_dir else video.parent / video.stem
    preview_dir = out_dir / "preview"

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # A container hint and regularly wrong, which is why a window that decodes
    # nothing is skipped rather than treated as an error below.
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise OSError(f"cannot determine length of {video.name}")

    windows = _sample_windows(frame_count, fps, count)

    # Pass 1: find the sharpest frame in each window, and build the max
    # projection the content box comes from. Only frame *indices* are kept --
    # holding N decoded frames would be hundreds of MB at 4K with a large count.
    #
    # The projection is fed every `box_stride`-th frame, not every frame: it is
    # a per-pixel max over ~CONTENT_BOX_SAMPLES frames spread across the video,
    # exactly as scan.py's _find_content_box does, and running it on all ~N*fps
    # decoded frames instead measured as roughly half this function's entire
    # cost for no change in the answer.
    planned = sum(stop - start for start, stop in windows)
    box_stride = max(1, planned // CONTENT_BOX_SAMPLES)

    projection = None
    decoded = 0
    best = []
    for start, stop in windows:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        best_index, best_score = None, -1.0
        for index in range(start, stop):
            ok, frame = cap.read()
            if not ok:
                break
            if decoded % box_stride == 0:
                peak = frame.max(axis=2)
                projection = peak if projection is None else np.maximum(projection, peak)
            decoded += 1
            # Scored on the full frame, not the content box: the box is not
            # known until the projection is complete, and it does not matter --
            # matte bars are constant, so they scale every frame's variance by
            # the same factor, leaving the ranking within a window (the only
            # comparison made here) unchanged. scan.py skips them purely to
            # avoid the wasted work.
            score = _sharpness_bgr(frame)
            if score > best_score:
                best_index, best_score = index, score
        if best_index is not None:
            best.append((best_index, best_score))

    if not best:
        cap.release()
        raise OSError(f"decoded no frames from {video.name}")

    # Distinct frames only: on a video too short for N windows several slices
    # resolve to the same frame, and writing it N times would claim N previews
    # of one picture.
    seen, picks = set(), []
    for index, score in best:
        if index not in seen:
            seen.add(index)
            picks.append((index, score))

    box = content_box_from_projection(projection)
    bx, by, bw, bh = box

    # Cleared, not resumed: this path is cheap enough to always redo, and
    # leftovers from a run with a different count would otherwise linger.
    preview_dir.mkdir(parents=True, exist_ok=True)
    for stale in preview_dir.glob("*.png"):
        stale.unlink()

    # Pass 2: decode the winners one at a time and write them out.
    stem = video.stem[:32]
    entries = []
    for index, score in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = frame[by:by + bh, bx:bx + bw]
        path = preview_dir / f"{stem}_f{index:06d}.png"
        write_png_rgb(path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        entries.append({
            "file": path.name, "path": str(path),
            "frame": index, "t": round(index / fps, 3),
            "source_sharpness": round(score, 3),
            "width": bw, "height": bh,
        })
    cap.release()

    if not entries:
        raise OSError(f"could not decode any of the chosen frames from {video.name}")

    opts = dict(sheet_opts or {})
    quality = opts.pop("quality", 92)
    opts.pop("title", None)
    title = f"{video.name}   preview, {len(entries)} frames"
    sheet_path = preview_dir / "contact_sheet.jpg"
    write_sheet(build_contact_sheet(entries, title=title, **opts), sheet_path,
                quality=quality)

    manifest = build_manifest(entries, contact_sheet=sheet_path.name)
    manifest["preview"] = {
        "count_requested": count,
        "frames": len(entries),
        "window_s": PREVIEW_WINDOW_S,
        "fps": round(fps, 6),
        "declared_frames": frame_count,
    }
    manifest["content_box"] = {"x": bx, "y": by, "w": bw, "h": bh}
    write_manifest(manifest, preview_dir / "manifest.json")

    short = f", {count - len(entries)} fewer than asked (video too short)" \
        if len(entries) < count else ""
    log(f"preview: {len(entries)} frames{short} -> {sheet_path}")
    return manifest
