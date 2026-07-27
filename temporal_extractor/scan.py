"""
Stage 1: decode the video once, score every frame, find scene boundaries,
write a metadata JSON. CPU only, no model.

Everything downstream reads this file and never touches the video again until
stage 3 needs actual pixels, so the scan pays for one decode pass and answers:

  - how sharp is each frame (variance of Laplacian)
  - where are the scene cuts
  - which frames are near-duplicates of each other (dHash, compared in stage 2)
  - what is the real picture area (content box, excluding pillar/letterboxing)

Scene detection follows PySceneDetect's ContentDetector in spirit -- mean
absolute difference of the HSV channels between consecutive frames, cut when the
score crosses a threshold -- but is computed inline on the frames we are already
decoding, rather than pulling in scenedetect and a second decode pass. The
default threshold of 27.0 is on the same scale as ContentDetector's default.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .frames import content_box_from_projection

# 2: added per-frame luma, so stage 2 can reject near-black frames.
SCAN_VERSION = 2

# Scene score at or above this is treated as a cut. Same scale as PySceneDetect's
# ContentDetector default.
DEFAULT_SCENE_THRESHOLD = 27.0

# Ignore cuts that would produce a scene shorter than this many frames. Real cuts
# closer together than ~half a second are usually flicker, and a scene shorter
# than a restore window is useless to us anyway.
DEFAULT_MIN_SCENE_LEN = 15

# The scene metric is computed on a downscaled copy: it is a whole-frame average,
# so detail beyond a couple hundred pixels wide contributes nothing but time.
SCENE_WORK_WIDTH = 256

# Frames sampled up front to establish the content box.
CONTENT_BOX_SAMPLES = 60


def dhash(gray: np.ndarray, size: int = 8) -> str:
    """
    64-bit difference hash, as hex.

    Compares each pixel with its right neighbour on a size+1 by size grid, so it
    encodes gradient direction rather than absolute brightness -- stable against
    exposure changes, sensitive to actual content change. Stage 2 dedupes by
    Hamming distance over these.
    """
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _scene_score(prev_hsv: np.ndarray, hsv: np.ndarray) -> float:
    """
    Mean absolute difference across H, S and V, averaged.

    Hue is circular, so a raw difference wraps at 180 (OpenCV's 8-bit hue range);
    folding it keeps a red-to-red transition from reading as a hard cut.
    """
    diff = cv2.absdiff(prev_hsv, hsv)
    hue = diff[:, :, 0].astype(np.float32)
    np.minimum(hue, 180.0 - hue, out=hue)
    return float((hue.mean() + diff[:, :, 1].mean() + diff[:, :, 2].mean()) / 3.0)


def _find_content_box(cap, frame_count: int, samples: int):
    """
    Establish the picture area from frames spread across the whole video.

    Sampling rather than accumulating over the full pass because the box has to
    be known BEFORE scoring, so sharpness is measured on picture only. A single
    frame is not enough: fades, dark shots and letterboxed inserts would each
    under-report the box, so we take the union over a spread of frames.
    """
    if frame_count <= 0:
        return None
    step = max(1, frame_count // max(1, samples))
    acc = None
    taken = 0
    for index in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        peak = frame.max(axis=2)
        acc = peak if acc is None else np.maximum(acc, peak)
        taken += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if acc is None:
        return None
    return content_box_from_projection(acc), taken


def scan_video(path, *, scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
               min_scene_len: int = DEFAULT_MIN_SCENE_LEN,
               detect_content_box: bool = True,
               progress=None) -> dict:
    """
    Decode `path` once and return the scan metadata as a dict.

    progress: optional callable(frames_done, frames_total) for UI.
    """
    path = Path(path)
    started = time.time()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # CAP_PROP_FRAME_COUNT is a container hint and is regularly wrong; it is fine
    # for sizing the content-box sweep, but the authoritative count is what we
    # actually decode below.
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    box = (0, 0, width, height)
    box_samples = 0
    if detect_content_box:
        found = _find_content_box(cap, declared_frames, CONTENT_BOX_SAMPLES)
        if found is not None:
            box, box_samples = found
    bx, by, bw, bh = box

    scale = SCENE_WORK_WIDTH / bw if bw > SCENE_WORK_WIDTH else 1.0
    work_size = (max(1, int(bw * scale)), max(1, int(bh * scale)))

    records = []
    cuts = []
    prev_hsv = None
    index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        picture = frame[by:by + bh, bx:bx + bw]
        gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)

        # Sharpness on picture only: the bars are constant, so including them
        # would just scale every score by the same factor and waste the work.
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        small = cv2.resize(picture, work_size, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        delta = 0.0 if prev_hsv is None else _scene_score(prev_hsv, hsv)
        prev_hsv = hsv

        if delta >= scene_threshold and index > 0:
            cuts.append(index)

        records.append({
            "i": index,
            "t": round(index / fps, 4),
            "sharpness": round(sharpness, 4),
            # Mean luminance of the picture area. A fade or an unlit shot can be
            # perfectly in focus and still be useless as training data, and
            # sharpness alone will not tell you that.
            "luma": round(float(gray.mean()), 2),
            "delta": round(delta, 4),
            "dhash": dhash(gray),
        })
        index += 1
        if progress and index % 200 == 0:
            progress(index, declared_frames)

    cap.release()
    total = len(records)
    if total == 0:
        raise OSError(f"decoded no frames from {path}")

    scenes = _build_scenes(cuts, total, fps, records, min_scene_len)
    for scene in scenes:
        for rec in records[scene["start"]:scene["end"] + 1]:
            rec["scene"] = scene["id"]

    return {
        "scan_version": SCAN_VERSION,
        "video": {
            "path": str(path.resolve()),
            "filename": path.name,
            "width": width,
            "height": height,
            "fps": round(fps, 6),
            "frame_count": total,
            "declared_frame_count": declared_frames,
            "duration_s": round(total / fps, 3),
        },
        # Stage 3 crops to this before restoring: it saves the compute the bars
        # would consume, and keeps black borders out of stills that are destined
        # for LoRA training.
        "content_box": {"x": bx, "y": by, "w": bw, "h": bh},
        "scan": {
            "scene_threshold": scene_threshold,
            "min_scene_len": min_scene_len,
            "content_box_samples": box_samples,
            "elapsed_s": round(time.time() - started, 3),
            # Laplacian variance is not scale-invariant and rises with invented
            # noise, so these numbers rank frames WITHIN this video only. Never
            # threshold on an absolute value or compare across videos.
            "sharpness_metric": "variance_of_laplacian",
        },
        "scenes": scenes,
        "frames": records,
    }


def _build_scenes(cuts, total, fps, records, min_scene_len):
    """
    Turn cut indices into closed scene ranges, dropping cuts that would make a
    scene too short to be worth anything.
    """
    boundaries = [0]
    for cut in cuts:
        if cut - boundaries[-1] >= min_scene_len:
            boundaries.append(cut)
    # A trailing scene shorter than the minimum gets folded into its predecessor
    # rather than left as a stub.
    if len(boundaries) > 1 and total - boundaries[-1] < min_scene_len:
        boundaries.pop()

    scenes = []
    for n, start in enumerate(boundaries):
        end = (boundaries[n + 1] - 1) if n + 1 < len(boundaries) else (total - 1)
        window = records[start:end + 1]
        best = max(window, key=lambda r: r["sharpness"])
        scenes.append({
            "id": n,
            "start": start,
            "end": end,
            "start_t": round(start / fps, 4),
            "end_t": round(end / fps, 4),
            "frame_count": end - start + 1,
            "best_frame": best["i"],
            "best_sharpness": best["sharpness"],
            "mean_sharpness": round(sum(r["sharpness"] for r in window) / len(window), 4),
        })
    return scenes


def write_scan(meta: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def read_scan(path) -> dict:
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    if meta.get("scan_version") != SCAN_VERSION:
        raise ValueError(
            f"scan file is version {meta.get('scan_version')}, this build expects {SCAN_VERSION}; "
            "re-run the scan stage"
        )
    return meta
