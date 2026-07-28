"""
Stage 1: decode the video once, score every frame, find scene boundaries,
write a metadata JSON. CPU only, no model.

Everything downstream reads this file and never touches the video again until
stage 3 needs actual pixels, so the scan pays for one decode pass and answers:

  - how sharp is each frame (variance of Laplacian)
  - how bright is it (to reject fades and unlit shots later)
  - how much of it is blown-out highlight (to reject overexposed frames later)
  - where are the scene cuts
  - which frames are near-duplicates of each other (dHash, compared in stage 2)
  - what is the real picture area (content box, excluding pillar/letterboxing)

Scene detection follows PySceneDetect's ContentDetector in spirit -- mean
absolute difference of the HSV channels between consecutive frames, cut when the
score crosses a threshold -- but is computed inline on the frames we are already
decoding, rather than pulling in scenedetect and a second decode pass. The
default threshold of 27.0 is on the same scale as ContentDetector's default.

PERFORMANCE
-----------
Decoding is not the bottleneck; our own per-frame arithmetic is. Measured on
1920x1080, decode alone runs at 368 fps while the whole stage once ran at 53.5.
Three costs dominated, and each was replaced with a cheaper form only after
checking it does not change the decisions the scan feeds downstream:

  - Laplacian in CV_64F cost 9.13 ms/frame. CV_32F costs 4.70 and is exactly
    equivalent for our purposes: Spearman 1.00000 against the old values, same
    sharpest frame, same top ten.
  - dHash cost 2.95 ms/frame, and it was not the bit packing -- it was resizing
    1920x1080 down to 9x8. Deriving it from the small working image instead
    costs 0.04 ms. 92% of hashes come out identical, mean drift 0.09 bits, and
    100% of pairwise accept/reject decisions at the dedupe threshold are
    unchanged.
  - Building the working image with INTER_AREA cost 2.50 ms/frame; INTER_LINEAR
    costs 0.12. Scene-delta correlation 0.9998, worst deviation 0.62 against a
    cut threshold of 27 that has an empty margin of 17 below it.

Sharpness is deliberately NOT computed on a downscaled frame, which would be the
obvious next saving. At half resolution the ranking degrades to Spearman 0.924
and only one of the top ten frames survives -- it would quietly select different
stills. Full resolution stays.

The remaining work is spread across processes. OpenCV threads Laplacian barely
at all (1.25x from 1 to 24 threads), so the cores have to be used by splitting
the video into frame ranges instead. Chunked output is verified bit-identical to
serial: frame indices, sharpness, scene deltas across every chunk boundary, and
every dHash.
"""

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager
from pathlib import Path

import cv2
import numpy as np

from .frames import content_box_from_projection
from .progress import ProgressBar

# 4: added per-frame highlight_frac.
# 3: cheaper equivalent metrics (CV_32F sharpness, dHash and scene delta from
#    the small working image), so values differ slightly from v2 scans.
# 2: added per-frame luma.
SCAN_VERSION = 4

# Scene score at or above this is treated as a cut. Same scale as PySceneDetect's
# ContentDetector default.
DEFAULT_SCENE_THRESHOLD = 27.0

# Ignore cuts that would produce a scene shorter than this many frames. Real cuts
# closer together than ~half a second are usually flicker, and a scene shorter
# than a restore window is useless to us anyway.
DEFAULT_MIN_SCENE_LEN = 15

# A pixel at or above this (8-bit) is counted as blown-highlight for
# highlight_frac. Picked from measurement, not guessed: a frame with a
# genuinely overexposed backdrop showed ~40% of its pixels at or above this
# level, while other frames in the same source at similar or higher MEAN luma
# -- but with an intact, non-clipped background -- showed close to 0%. Mean
# luma alone cannot make this distinction (the overexposed frame's mean was
# not the highest in its scene); the fraction of near-ceiling pixels can.
HIGHLIGHT_LEVEL = 230

# The scene metric is a whole-frame average, so detail beyond a couple hundred
# pixels wide contributes nothing but time.
SCENE_WORK_WIDTH = 256

# Frames sampled up front to establish the content box.
CONTENT_BOX_SAMPLES = 60

# Conservative by design: a 4-core machine is a fair assumption, and oversubscribing
# hurts. Raise it with --workers on a bigger CPU; measured returns flatten past
# about 12 and 24 is slower than 12 through decode contention.
DEFAULT_WORKERS = 4

# Below this, process startup and seeking cost more than they save.
MIN_FRAMES_FOR_PARALLEL = 300


def dhash_bits(gray_small: np.ndarray, size: int = 8) -> str:
    """
    64-bit difference hash, as hex.

    Compares each pixel with its right neighbour on a size+1 by size grid, so it
    encodes gradient direction rather than absolute brightness -- stable against
    exposure changes, sensitive to actual content change. Stage 2 dedupes by
    Hamming distance over these.

    Takes an already-downscaled grayscale image: resizing from full resolution
    was 74x more expensive and made no difference to any dedupe decision.
    """
    small = cv2.resize(gray_small, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


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


def _measure(frame, box, work_size, prev_hsv):
    """
    All per-frame metrics, in one place.

    Shared verbatim by the serial and parallel paths so the two cannot drift
    apart -- the parallel result is only trustworthy because this is the single
    definition of what a frame's numbers are.
    """
    x, y, w, h = box
    picture = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)

    # Sharpness on picture only, at full resolution. The bars are constant, so
    # including them would scale every score alike and waste the work.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    luma = float(gray.mean())
    # Fraction of the frame that is blown out, as opposed to merely bright --
    # see HIGHLIGHT_LEVEL. Reused verbatim (not recomputed) on the full-res
    # gray already built for sharpness.
    highlight_frac = float((gray >= HIGHLIGHT_LEVEL).mean())

    small = cv2.resize(picture, work_size, interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    delta = 0.0 if prev_hsv is None else _scene_score(prev_hsv, hsv)
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    return {
        "sharpness": round(sharpness, 4),
        "luma": round(luma, 2),
        "highlight_frac": round(highlight_frac, 4),
        "delta": round(delta, 4),
        "dhash": dhash_bits(gray_small),
    }, hsv


def _scan_range(job):
    """
    Worker: scan frames [start, stop).

    Decodes one frame of lead-in before `start` so the scene delta at a chunk's
    first frame is computed against its true predecessor, exactly as the serial
    pass would. Without it, every chunk boundary would report a delta of 0 and
    a real cut landing there would be missed.

    `force_no_lead` skips that lead-in: set for the first chunk of a scan range
    that has no real predecessor to compare against -- either frame 0 of the
    whole video, or the first frame of a `--segment`, where the frame just
    before `start` belongs to different, excluded footage and comparing against
    it would be meaningless.
    """
    path, start, stop, box, work_size, queue, report_every, force_no_lead = job
    cv2.setNumThreads(1)  # workers must not fight each other for cores

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    lead = start if force_no_lead else max(0, start - 1)
    if lead:
        cap.set(cv2.CAP_PROP_POS_FRAMES, lead)

    records = []
    prev_hsv = None
    pending = 0
    for index in range(lead, stop):
        ok, frame = cap.read()
        if not ok:
            break
        fields, prev_hsv = _measure(frame, box, work_size, prev_hsv)
        if index >= start:
            records.append((index, fields))
            pending += 1
            if queue is not None and pending >= report_every:
                queue.put(pending)
                pending = 0
    cap.release()
    if queue is not None and pending:
        queue.put(pending)
    return records


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


def resolve_workers(requested, frame_count: int) -> int:
    """How many processes to actually use, given the machine and the workload."""
    cpus = os.cpu_count() or 1
    want = DEFAULT_WORKERS if requested in (None, 0) else int(requested)
    want = max(1, min(want, cpus))
    if frame_count and frame_count < MIN_FRAMES_FOR_PARALLEL:
        return 1
    if frame_count:
        # Never so many chunks that each is trivially short.
        want = max(1, min(want, frame_count // 100 or 1))
    return want


def resolve_segments(segments_s, fps: float) -> list[tuple[int, int]]:
    """
    Convert `--segment FROM TO` pairs (seconds) into sorted, merged, disjoint
    (start, stop) frame ranges with stop EXCLUSIVE -- the same convention
    `_scan_range`/`_scan_serial`/`_scan_parallel` already use.

    Both endpoints are treated as inclusive of the frame at that timestamp, so
    `stop` is the rounded end frame plus one. Overlapping or adjacent segments
    are merged so a frame is never scanned twice and never ends up split
    across two "independent" scenes that actually share footage.

    Not clipped to the container's declared frame count: that count is a hint
    and is regularly wrong (see `scan_video`'s docstring). A segment that
    reaches past the true end of the video simply decodes fewer frames than
    requested -- `scan_video` treats an empty result as the real error.
    """
    frame_pairs = []
    for start_s, end_s in segments_s:
        if end_s <= start_s:
            raise ValueError(
                f"segment {start_s:.3f}s-{end_s:.3f}s: end must be after start")
        start_f = max(0, round(start_s * fps))
        stop_f = round(end_s * fps) + 1
        if stop_f <= start_f:
            stop_f = start_f + 1
        frame_pairs.append((start_f, stop_f))

    frame_pairs.sort()
    merged = []
    for start_f, stop_f in frame_pairs:
        if merged and start_f <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop_f))
        else:
            merged.append((start_f, stop_f))
    return merged


def scan_video(path, *, scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
               min_scene_len: int = DEFAULT_MIN_SCENE_LEN,
               detect_content_box: bool = True,
               workers: int | None = None,
               segments=None,
               show_progress: bool = True) -> dict:
    """
    Decode `path` once and return the scan metadata as a dict.

    `segments`, if given, is a list of (start_s, end_s) second pairs (as
    produced by `--segment FROM TO`): only that footage is decoded and scored,
    which is the point for a feature-length source where only a few scenes are
    usable -- scanning 90 minutes to keep 3 is wasted decode. Each segment
    forms its own independent set of scenes: a scene never spans a segment
    boundary, because the frames on either side of that boundary are not
    adjacent in the source and blending across it (stage 3's restore window)
    would be meaningless. `None` (the default) scans the whole video, exactly
    as before this option existed.
    """
    path = Path(path)
    started = time.time()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # A container hint, regularly wrong. Fine for sizing the content-box sweep
    # and the progress bar, but the authoritative count is what we decode.
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    box = (0, 0, width, height)
    box_samples = 0
    if detect_content_box:
        bar = ProgressBar(total=None, label="  content box",
                          enabled=show_progress, interval=0.3)
        found = _find_content_box(cap, declared_frames, CONTENT_BOX_SAMPLES)
        if found is not None:
            box, box_samples = found
        bar.close(f"  content box   {box[2]}x{box[3]} at ({box[0]},{box[1]}) "
                  f"from {box_samples} samples")
    bx, by, bw, bh = box

    scale = SCENE_WORK_WIDTH / bw if bw > SCENE_WORK_WIDTH else 1.0
    work_size = (max(1, int(bw * scale)), max(1, int(bh * scale)))

    frame_segments = resolve_segments(segments, fps) if segments else None

    if frame_segments is None:
        n_workers = resolve_workers(workers, declared_frames)
        bar = ProgressBar(total=declared_frames, label="  scanning", enabled=show_progress)

        if n_workers > 1:
            cap.release()
            pairs = _scan_parallel(str(path), 0, declared_frames, box, work_size, n_workers, bar)
        else:
            pairs = _scan_serial(cap, box, work_size, bar)
            cap.release()

        records = [{"i": index, "t": round(index / fps, 4), **fields} for index, fields in pairs]
        total = len(records)
        bar.set_done(total)
        bar.close(f"  scanning      {total:,} frames in "
                  f"{time.time() - started:.1f}s using {n_workers} "
                  f"worker{'s' if n_workers != 1 else ''}")

        if total == 0:
            raise OSError(f"decoded no frames from {path}")

        cuts = [r["i"] for r in records if r["delta"] >= scene_threshold and r["i"] > 0]
        scenes = _build_scenes(cuts, total, records, min_scene_len)
        for scene in scenes:
            for rec in records[scene["start"]:scene["end"] + 1]:
                rec["scene"] = scene["id"]
        segment_meta = None
    else:
        cap.release()
        scan_total = sum(stop - start for start, stop in frame_segments)
        bar = ProgressBar(total=scan_total, label="  scanning", enabled=show_progress)

        records, scenes = [], []
        base_done = id_base = position_offset = 0
        workers_used = 1
        for seg_start, seg_stop in frame_segments:
            seg_records, seg_scenes, seg_workers = _scan_one_segment(
                str(path), seg_start, seg_stop, box, work_size, workers,
                scene_threshold, min_scene_len, fps, bar, base_done)
            if not seg_records:
                raise OSError(
                    f"segment {seg_start / fps:.3f}s-{(seg_stop - 1) / fps:.3f}s "
                    f"(frames {seg_start}-{seg_stop - 1}) produced no frames; is "
                    "the timestamp beyond the video's actual length?")
            for scene in seg_scenes:
                scene["id"] += id_base
            for rec in seg_records:
                rec["scene"] += id_base
            for scene in seg_scenes:
                scene["start"] += position_offset
                scene["end"] += position_offset

            records.extend(seg_records)
            scenes.extend(seg_scenes)
            base_done += (seg_stop - seg_start)
            id_base += len(seg_scenes)
            position_offset += len(seg_records)
            workers_used = max(workers_used, seg_workers)

        total = len(records)
        bar.set_done(scan_total)
        bar.close(f"  scanning      {total:,} frames across {len(frame_segments)} "
                  f"segment{'s' if len(frame_segments) != 1 else ''} in "
                  f"{time.time() - started:.1f}s using up to {workers_used} "
                  f"worker{'s' if workers_used != 1 else ''}")
        n_workers = workers_used
        segment_meta = [{
            "start": start, "end": stop - 1,
            "start_t": round(start / fps, 4), "end_t": round((stop - 1) / fps, 4),
        } for start, stop in frame_segments]

    return {
        "scan_version": SCAN_VERSION,
        "video": {
            "path": str(path.resolve()),
            "filename": path.name,
            "width": width,
            "height": height,
            "fps": round(fps, 6),
            # In segmented mode these describe what was actually decoded (the
            # requested footage), not the whole source -- consistent with the
            # whole-video case, where they already mean "what we decoded" and
            # not "what the container claims".
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
            "workers": n_workers,
            "elapsed_s": round(time.time() - started, 3),
            # Laplacian variance is not scale-invariant and rises with invented
            # noise, so these numbers rank frames WITHIN this video only. Never
            # threshold on an absolute value or compare across videos.
            "sharpness_metric": "variance_of_laplacian",
            "segments": segment_meta,
        },
        "scenes": scenes,
        "frames": records,
    }


def _scan_serial(cap, box, work_size, bar, start=0, stop=None, base_done=0):
    """
    Serial scan of frames [start, stop) from an already-positioned `cap`
    (stop=None reads to true EOF, ignoring any declared/requested bound --
    used for the whole-video path, where the container's frame count is a hint
    that is regularly wrong).

    `base_done`/`start` let this be called once per `--segment` while keeping
    one continuous progress bar across all of them.
    """
    pairs = []
    prev_hsv = None
    index = start
    while stop is None or index < stop:
        ok, frame = cap.read()
        if not ok:
            break
        fields, prev_hsv = _measure(frame, box, work_size, prev_hsv)
        pairs.append((index, fields))
        index += 1
        bar.set_done(base_done + (index - start))
    return pairs


def _scan_parallel(path, start, stop, box, work_size, n_workers, bar, base_done=0):
    """
    Split frames [start, stop) into one range per worker and merge the results.

    Progress is aggregated: each worker posts its completed-frame count to a
    shared queue, which the parent drains while waiting. Reporting per chunk
    completion instead would leave the bar frozen for most of the run and then
    jump.
    """
    length = stop - start
    edges = [start + length * k // n_workers for k in range(n_workers + 1)]
    report_every = max(16, length // (n_workers * 200) or 16)

    with Manager() as manager:
        queue = manager.Queue()
        jobs = [(path, edges[k], edges[k + 1], box, work_size, queue, report_every, k == 0)
                for k in range(n_workers)]
        done = 0

        def drain():
            nonlocal done
            moved = False
            while True:
                try:
                    done += queue.get_nowait()
                except Exception:
                    break
                moved = True
            if moved:
                bar.set_done(base_done + done)

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_scan_range, job) for job in jobs]
            pending = set(futures)
            # Waiting on the result iterator directly would block for the whole
            # run and freeze the bar. Poll with a short timeout instead so
            # progress keeps flowing while the workers are busy.
            while pending:
                _, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                drain()
            results = [f.result() for f in futures]
        drain()

    merged = [item for chunk in results for item in chunk]
    merged.sort(key=lambda pair: pair[0])
    return merged


def _build_scenes(cuts, total, records, min_scene_len):
    """
    Turn cut indices into closed scene ranges, dropping cuts that would make a
    scene too short to be worth anything.

    `cuts`/`start`/`end` are POSITIONS in `records` (0-based, contiguous), not
    necessarily the frames' own `"i"` values -- the two coincide for a
    whole-video scan but not inside a `--segment`, where `records` holds only
    the scanned subset and `"i"`/`"t"` carry the frames' real index/time in the
    source video. `select.py` slices `meta["frames"]` by position, so `start`/
    `end` have to stay positions; `start_i`/`end_i` carry the real frame
    numbers alongside them, for anything that wants to seek the actual video
    (the CLI's scene table, external tools) rather than index this scan.
    `start_t`/`end_t` are read from the records themselves for the same
    reason, rather than recomputed as `start / fps`.
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
            "start_i": records[start]["i"],
            "end_i": records[end]["i"],
            "start_t": records[start]["t"],
            "end_t": records[end]["t"],
            "frame_count": end - start + 1,
            "best_frame": best["i"],
            "best_sharpness": best["sharpness"],
            "mean_sharpness": round(sum(r["sharpness"] for r in window) / len(window), 4),
        })
    return scenes


def _scan_one_segment(path, seg_start, seg_stop, box, work_size, workers,
                      scene_threshold, min_scene_len, fps, bar, base_done):
    """
    Scan frames [seg_start, seg_stop) and build that segment's own scenes,
    completely independent of any other segment.

    Cuts are computed from POSITION within this segment's own records, not
    from the frames' real `"i"`, and position 0 -- the segment's first frame
    -- is never treated as a cut candidate. That is the same rule the
    whole-video path applies to real frame 0, applied here to "first frame we
    actually decoded", which is exactly right: whatever frame preceded this
    segment in the source belongs to footage we deliberately excluded, so
    there is nothing meaningful to compare against, and forcing every
    segment's first position out of the cut search guarantees no scene can
    ever span a segment boundary.
    """
    length = seg_stop - seg_start
    n_workers = resolve_workers(workers, length)
    if n_workers > 1:
        pairs = _scan_parallel(path, seg_start, seg_stop, box, work_size, n_workers, bar, base_done)
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise OSError(f"cannot open video: {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
        pairs = _scan_serial(cap, box, work_size, bar, start=seg_start, stop=seg_stop,
                             base_done=base_done)
        cap.release()

    if not pairs:
        return [], [], n_workers

    seg_records = [{"i": index, "t": round(index / fps, 4), **fields} for index, fields in pairs]
    cuts = [pos for pos, r in enumerate(seg_records) if r["delta"] >= scene_threshold and pos > 0]
    seg_scenes = _build_scenes(cuts, len(seg_records), seg_records, min_scene_len)
    for scene in seg_scenes:
        for rec in seg_records[scene["start"]:scene["end"] + 1]:
            rec["scene"] = scene["id"]
    return seg_records, seg_scenes, n_workers


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
