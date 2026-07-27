"""
Stage 2: choose which frames to restore, from the scan metadata alone.

Never touches the video -- stage 1 already recorded everything needed, so this
stage is pure bookkeeping and runs instantly.

Sharpness alone is not a good enough criterion. A slow pan is a long run of
frames that are all in focus and all nearly the same picture, so ranking by
sharpness and taking the top N returns twenty views of one shot. Three
constraints are applied together:

  sharpness   rank candidates within a scene
  dedupe      reject a candidate too close in dHash to one already picked
  spread      allocate picks across scenes in proportion to their length

And one constraint inherited from stage 3: a restore window [i-k .. i+k] must
lie inside a single scene, because blending frames across a cut is meaningless.
"""

import json
from pathlib import Path

SELECT_VERSION = 1

DEFAULT_COUNT = 15
DEFAULT_WINDOW = 5

# Minimum dHash Hamming distance between any two picks. Measured on the sample
# footage: adjacent frames score 0, frames 0.2s apart score 3, 1s apart score
# 13, and 9s apart within one long take score 41. 16 keeps picks meaningfully
# different without demanding they come from different shots.
DEFAULT_HASH_DISTANCE = 16

# No two picks closer than this many seconds. Segmentation already spreads picks,
# but two adjacent segments can both choose a frame near their shared boundary,
# which put two picks 1s apart in one shot on the sample footage. dHash cannot
# catch that on its own: in a close-up, a second of small movement flips enough
# bits to clear the distance floor while the picture is, for training purposes,
# the same. This is the backstop for that case.
DEFAULT_MIN_GAP_S = 1.0


def hamming(a: str, b: str) -> int:
    """Bit distance between two hex-encoded dHashes."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _allocate(scene_capacity: dict, count: int, max_per_scene: int | None) -> dict:
    """
    Spread `count` picks over scenes in proportion to how many eligible frames
    each holds, by largest remainder, with every non-empty scene guaranteed at
    least one pick so a short scene is not shut out by a long one.
    """
    scenes = [s for s, cap in scene_capacity.items() if cap > 0]
    if not scenes or count <= 0:
        return {}

    quota = {s: 1 for s in scenes[:count]}
    remaining = count - len(quota)

    if remaining > 0:
        total = sum(scene_capacity[s] for s in quota)
        exact = {s: remaining * scene_capacity[s] / total for s in quota}
        for s in quota:
            quota[s] += int(exact[s])
        leftover = count - sum(quota.values())
        # Largest fractional remainder wins the odd picks.
        for s in sorted(quota, key=lambda s: exact[s] - int(exact[s]), reverse=True)[:leftover]:
            quota[s] += 1

    # Never promise a scene more picks than it has frames to give.
    for s in quota:
        quota[s] = min(quota[s], scene_capacity[s])
        if max_per_scene is not None:
            quota[s] = min(quota[s], max_per_scene)
    return quota


def select_frames(meta: dict, *, count: int = DEFAULT_COUNT, window: int = DEFAULT_WINDOW,
                  hash_distance: int = DEFAULT_HASH_DISTANCE,
                  min_gap_s: float = DEFAULT_MIN_GAP_S,
                  max_per_scene: int | None = None) -> dict:
    """
    Pick frames to restore. Returns a select-metadata dict.

    meta:           the dict written by stage 1
    count:          how many stills to aim for
    window:         restore window size; picks are kept far enough from a cut
                    that the whole window stays inside one scene
    hash_distance:  minimum dHash Hamming distance between any two picks
    min_gap_s:      minimum time between any two picks, in seconds
    max_per_scene:  optional cap so one long take cannot dominate
    """
    if window < 5 or window % 4 != 1:
        raise ValueError(f"window must be 4n+1 and at least 5, got {window}")

    min_gap = int(round(min_gap_s * meta["video"]["fps"]))

    half = window // 2
    frames = meta["frames"]
    scenes = {s["id"]: s for s in meta["scenes"]}

    # Eligible = a full window fits inside this frame's own scene. Kept in frame
    # order, because the selection below slices these ranges by time.
    eligible = {}
    for scene in meta["scenes"]:
        lo, hi = scene["start"] + half, scene["end"] - half
        eligible[scene["id"]] = [frames[i] for i in range(lo, hi + 1)] if hi >= lo else []

    skipped = [s["id"] for s in meta["scenes"] if not eligible[s["id"]]]
    quota = _allocate({sid: len(v) for sid, v in eligible.items()}, count, max_per_scene)

    chosen = []

    def accepts(frame):
        """Reject anything too close to something already picked, anywhere."""
        return all(
            abs(frame["i"] - c["i"]) >= min_gap
            and hamming(frame["dhash"], c["dhash"]) >= hash_distance
            for c in chosen
        )

    # Pass 1: split each scene into as many equal time segments as it has picks,
    # and take the sharpest acceptable frame from each.
    #
    # Ranking a whole scene by sharpness and taking the top q does NOT work: in a
    # 35s take of a face, the sharpest six frames landed inside one 5s span and
    # still passed the dHash floor, because small movements in a close-up flip
    # plenty of hash bits. Segmenting forces temporal spread, and picking the
    # local best within each segment keeps the sharpness criterion intact.
    for sid in sorted(quota):
        pool = eligible[sid]
        q = quota[sid]
        for k in range(q):
            lo = k * len(pool) // q
            hi = (k + 1) * len(pool) // q
            segment = sorted(pool[lo:hi], key=lambda f: f["sharpness"], reverse=True)
            for frame in segment:
                if accepts(frame):
                    chosen.append(frame)
                    break

    # Pass 2: dedupe may have starved a segment, leaving us short. Fill the
    # shortfall with the candidate FURTHEST in time from everything already
    # picked, breaking ties by sharpness.
    #
    # Filling by sharpness alone silently undoes pass 1's work: when the last
    # segment of a long take had nothing acceptable, the sharpest frame left in
    # the whole video sat right beside an existing pick, and the fill dropped it
    # there -- two picks 1.2s apart. Maximin keeps the set spread out.
    filled = 0
    if len(chosen) < count:
        picked = {f["i"] for f in chosen}
        pool = [f for group in eligible.values() for f in group if f["i"] not in picked]
        while len(chosen) < count:
            best = None
            best_key = None
            for frame in pool:
                if frame["i"] in picked or not accepts(frame):
                    continue
                key = (min(abs(frame["i"] - c["i"]) for c in chosen) if chosen else 0,
                       frame["sharpness"])
                if best_key is None or key > best_key:
                    best, best_key = frame, key
            if best is None:
                break
            chosen.append(best)
            picked.add(best["i"])
            filled += 1

    chosen.sort(key=lambda f: f["i"])
    box = meta["content_box"]
    picks = [{
        "frame": f["i"],
        "t": f["t"],
        "scene": f.get("scene"),
        "sharpness": f["sharpness"],
        "dhash": f["dhash"],
        # Closed interval, inclusive of both ends -- exactly what stage 3 decodes.
        "window": [f["i"] - half, f["i"] + half],
    } for f in chosen]

    return {
        "select_version": SELECT_VERSION,
        "scan_version": meta.get("scan_version"),
        "video": meta["video"],
        "content_box": box,
        "select": {
            "requested": count,
            "selected": len(picks),
            "window": window,
            "hash_distance": hash_distance,
            "min_gap_s": min_gap_s,
            "min_gap_frames": min_gap,
            "max_per_scene": max_per_scene,
            "scene_quota": {str(k): v for k, v in sorted(quota.items())},
            "scenes_too_short": skipped,
            "eligible_frames": sum(len(v) for v in eligible.values()),
            # How many picks came from the maximin fill rather than a segment.
            # A high number means dedupe is starving segments -- the footage has
            # less variety than `count` assumes.
            "filled_globally": filled,
        },
        "scenes": [scenes[sid] for sid in sorted(scenes)],
        "picks": picks,
    }


def write_selection(sel: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sel, indent=2), encoding="utf-8")
    return path


def read_selection(path) -> dict:
    sel = json.loads(Path(path).read_text(encoding="utf-8"))
    if sel.get("select_version") != SELECT_VERSION:
        raise ValueError(
            f"selection file is version {sel.get('select_version')}, "
            f"this build expects {SELECT_VERSION}; re-run the select stage"
        )
    return sel
