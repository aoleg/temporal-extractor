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

# 2: per-scene allocation with no global ceiling; weak-frame rejection;
#    scene-adaptive minimum gap.
SELECT_VERSION = 2

DEFAULT_WINDOW = 5

# How much scene time earns one still, before clamping. Scene length has to
# matter: a 35s take of a character turning through several angles is worth more
# stills than a 7s insert, and those angles are exactly what a LoRA needs.
DEFAULT_SECONDS_PER_STILL = 4.0

# Ceiling per scene, so one very long take cannot swamp the set. There is
# deliberately NO global ceiling: with one, a film with more scenes than the
# ceiling would silently lose whole scenes, which is worse than returning more
# stills than expected.
DEFAULT_PER_SCENE_MAX = 8

# Minimum dHash distance between picks. Kept low on purpose. Two frames a second
# apart in a close-up can differ by a shirt collar coming into view or a slight
# head turn -- visually "the same shot", but genuinely different training
# examples. This filter is here to catch frames that are actually redundant
# (static shots, held frames), not to enforce variety.
DEFAULT_HASH_DISTANCE = 8

# Spread backstop, not a duplicate filter. The real spreading is done by
# segmentation; this only stops two adjacent segments from both picking right at
# their shared boundary. Expressed as a fraction of the scene's own natural
# spacing (eligible length / picks), so it scales with the scene instead of
# imposing one number on a 7s insert and a 35s take alike.
DEFAULT_GAP_FRACTION = 0.2
DEFAULT_MIN_GAP_S = 0.4

# Weak-frame rejection. The output is training data; a near-black or badly soft
# frame is not worth a restore pass, let alone a slot in the set.
#
# Sharpness is judged relative to the best frame in its own scene, never against
# an absolute number: variance of Laplacian is not comparable across scenes, let
# alone across videos. A dim scene shot on a long lens has its own scale.
DEFAULT_MIN_SHARPNESS_FRAC = 0.35
DEFAULT_MIN_LUMA = 24.0


def hamming(a: str, b: str) -> int:
    """Bit distance between two hex-encoded dHashes."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _allocate(eligible: dict, scenes: dict, fps: float, seconds_per_still: float,
              per_scene_max: int) -> dict:
    """
    Decide how many stills each scene earns, from its own length alone.

    Every scene with a usable frame gets at least one -- a short scene is still a
    distinct look, and dropping it loses that look entirely. Beyond the first,
    scenes earn a still per `seconds_per_still` of screen time, capped by
    `per_scene_max`. No global total: the video's structure sets the count.

    Screen time here is the scene's real duration, NOT the number of frames that
    survived weak-frame rejection. Those are different questions: a 12s scene
    with heavy motion blur is still a 12s scene and still deserves its share of
    the output, it just has fewer frames to choose from. Sizing the quota off the
    survivors made a whole scene collapse to one still because half its frames
    were soft.
    """
    quota = {}
    for sid, frames in eligible.items():
        if not frames:
            continue
        scene = scenes[sid]
        seconds = (scene["end"] - scene["start"] + 1) / fps
        earned = int(round(seconds / seconds_per_still)) if seconds_per_still > 0 else 1
        quota[sid] = max(1, min(earned, per_scene_max, len(frames)))
    return quota


def select_frames(meta: dict, *, window: int = DEFAULT_WINDOW,
                  seconds_per_still: float = DEFAULT_SECONDS_PER_STILL,
                  per_scene_max: int = DEFAULT_PER_SCENE_MAX,
                  hash_distance: int = DEFAULT_HASH_DISTANCE,
                  min_gap_s: float = DEFAULT_MIN_GAP_S,
                  gap_fraction: float = DEFAULT_GAP_FRACTION,
                  min_sharpness_frac: float = DEFAULT_MIN_SHARPNESS_FRAC,
                  min_luma: float = DEFAULT_MIN_LUMA) -> dict:
    """
    Pick frames to restore. Returns a select-metadata dict.

    meta:                the dict written by stage 1
    window:              restore window size; picks are kept far enough from a
                         cut that the whole window stays inside one scene
    seconds_per_still:   scene time that earns one still
    per_scene_max:       ceiling per scene (there is no global ceiling)
    hash_distance:       minimum dHash Hamming distance between picks
    min_gap_s:           absolute floor on the gap between picks
    gap_fraction:        gap floor as a fraction of a scene's natural spacing
    min_sharpness_frac:  reject frames below this fraction of their scene's best
    min_luma:            reject frames dimmer than this mean luminance
    """
    if window < 5 or window % 4 != 1:
        raise ValueError(f"window must be 4n+1 and at least 5, got {window}")
    if meta.get("scan_version", 1) < 2 and min_luma > 0:
        raise ValueError(
            "this scan predates per-frame luma, so near-black frames cannot be "
            "rejected; re-run the scan stage, or pass min_luma=0"
        )

    fps = meta["video"]["fps"]
    min_gap = int(round(min_gap_s * fps))

    half = window // 2
    frames = meta["frames"]
    scenes = {s["id"]: s for s in meta["scenes"]}

    # Eligible = a full window fits inside this frame's own scene, and the frame
    # is worth restoring at all. Kept in frame order: the selection below slices
    # these ranges by time.
    eligible = {}
    rejected = {"too_dark": 0, "too_soft": 0}
    for scene in meta["scenes"]:
        lo, hi = scene["start"] + half, scene["end"] - half
        pool = [frames[i] for i in range(lo, hi + 1)] if hi >= lo else []
        # Judged against this scene's own best, never an absolute threshold.
        floor = scene["best_sharpness"] * min_sharpness_frac
        keep = []
        for f in pool:
            if min_luma > 0 and f.get("luma", 255) < min_luma:
                rejected["too_dark"] += 1
            elif f["sharpness"] < floor:
                rejected["too_soft"] += 1
            else:
                keep.append(f)
        eligible[scene["id"]] = keep

    skipped = [s["id"] for s in meta["scenes"] if not eligible[s["id"]]]
    quota = _allocate(eligible, scenes, fps, seconds_per_still, per_scene_max)

    # Scene-adaptive gap: a scene taking q picks across its own span has a
    # natural spacing of span/q, and picks should not crowd far inside that.
    # Measured on the span, not on the surviving frame count, for the same
    # reason the quota is -- survivors can be bunched into part of the scene.
    scene_span = {}
    scene_gap = {}
    for sid, q in quota.items():
        scene = scenes[sid]
        lo, hi = scene["start"] + half, scene["end"] - half
        scene_span[sid] = (lo, hi)
        scene_gap[sid] = max(min_gap, int(gap_fraction * (hi - lo + 1) / q))

    chosen = []

    def accepts(frame, sid=None):
        """Reject anything too close to something already picked, anywhere."""
        gap = scene_gap.get(sid, min_gap) if sid is not None else min_gap
        return all(
            abs(frame["i"] - c["i"]) >= (gap if c.get("scene") == sid else min_gap)
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
    #
    # Segments are spans of TIME, not equal slices of the surviving-frame list.
    # Slicing the list put two picks half a second apart: weak-frame rejection
    # had left survivors bunched into one part of the scene, so equal slices of
    # the list were not equal slices of the scene.
    for sid in sorted(quota):
        q = quota[sid]
        lo, hi = scene_span[sid]
        span = hi - lo + 1
        by_index = {f["i"]: f for f in eligible[sid]}
        for k in range(q):
            seg_lo = lo + k * span // q
            seg_hi = lo + (k + 1) * span // q
            segment = sorted((by_index[i] for i in range(seg_lo, seg_hi) if i in by_index),
                             key=lambda f: f["sharpness"], reverse=True)
            for frame in segment:
                if accepts(frame, sid):
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
    # The fill stays WITHIN the starved scene. Each scene's count is a promise
    # derived from its own length, so borrowing a pick from elsewhere to hit a
    # number would misrepresent the video's structure.
    filled = 0
    picked = {f["i"] for f in chosen}
    for sid in sorted(quota):
        have = sum(1 for c in chosen if c.get("scene") == sid)
        pool = [f for f in eligible[sid] if f["i"] not in picked]
        while have < quota[sid]:
            best = None
            best_key = None
            for frame in pool:
                if frame["i"] in picked or not accepts(frame, sid):
                    continue
                key = (min(abs(frame["i"] - c["i"]) for c in chosen) if chosen else 0,
                       frame["sharpness"])
                if best_key is None or key > best_key:
                    best, best_key = frame, key
            if best is None:
                break
            chosen.append(best)
            picked.add(best["i"])
            have += 1
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
            "selected": len(picks),
            "window": window,
            "seconds_per_still": seconds_per_still,
            "per_scene_max": per_scene_max,
            "hash_distance": hash_distance,
            "min_gap_s": min_gap_s,
            "min_gap_frames": min_gap,
            "gap_fraction": gap_fraction,
            "scene_gap_frames": {str(k): v for k, v in sorted(scene_gap.items())},
            "min_sharpness_frac": min_sharpness_frac,
            "min_luma": min_luma,
            "rejected_weak": rejected,
            "scene_quota": {str(k): v for k, v in sorted(quota.items())},
            "scenes_unusable": skipped,
            "eligible_frames": sum(len(v) for v in eligible.values()),
            # Picks that came from the within-scene maximin fill rather than
            # from their own segment. A high number means the scene has less
            # variety than its length suggests.
            "filled": filled,
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
