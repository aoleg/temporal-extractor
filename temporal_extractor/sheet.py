"""
Stage 4: contact sheet for manual review, and the manifest.

The contact sheet exists to be looked at -- it is the point in the pipeline
where a human decides which stills to keep, so every cell is labelled with what
it needs to be traced back: the source frame, its timestamp, its scene, and the
seed that produced it.

The manifest is the durable record: for each still, which source frames were fed
to the restorer to produce it. That is the thing you cannot reconstruct later by
looking at the PNG.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

MANIFEST_VERSION = 1

# Canonical still filename, e.g. clip_s003_f000880_seed42.png. Stage 4 reads
# frame/scene/seed back out of it, so the whole pipeline agrees on one shape.
STILL_PATTERN = re.compile(
    r"_s(?P<scene>\d+)_f(?P<frame>\d+)(?:_seed(?P<seed>\d+))?", re.IGNORECASE
)

BACKGROUND = (24, 24, 24)
TEXT = (235, 235, 235)
TEXT_DIM = (150, 150, 150)


def still_name(video_stem: str, scene, frame: int, seed=None) -> str:
    """The one place the still filename convention is defined."""
    scene_part = f"_s{scene:03d}" if scene is not None else "_sxxx"
    seed_part = f"_seed{seed}" if seed is not None else ""
    return f"{video_stem}{scene_part}_f{frame:06d}{seed_part}.png"


# What --upscale-stills appends. A suffix rather than a separate folder so the
# pair sits together in one listing, and so STILL_PATTERN still finds the
# frame/scene it was built from -- the upscaled copy keeps its provenance.
UPSCALED_SUFFIX = "_upscaled"


def upscaled_name(path) -> Path:
    """The companion --upscale-stills writes beside a kept still."""
    path = Path(path)
    return path.with_name(f"{path.stem}{UPSCALED_SUFFIX}.png")


def is_upscaled(path) -> bool:
    return Path(path).stem.endswith(UPSCALED_SUFFIX)


def parse_still_name(path) -> dict:
    """Recover frame/scene/seed from a still filename; empty dict if it does not match."""
    m = STILL_PATTERN.search(Path(path).stem)
    if not m:
        return {}
    out = {"frame": int(m.group("frame")), "scene": int(m.group("scene"))}
    if m.group("seed") is not None:
        out["seed"] = int(m.group("seed"))
    return out


def collect_stills(stills_dir, selection: dict | None = None) -> list[dict]:
    """
    Gather stills from a directory and attach what the selection knows about them.

    Sorted by source frame so the sheet reads in temporal order -- reviewing a
    grid that jumps around the film is needlessly hard.
    """
    picks = {p["frame"]: p for p in (selection or {}).get("picks", [])}
    entries = []
    for path in sorted(Path(stills_dir).glob("*.png")):
        entry = {"file": path.name, "path": str(path)}
        entry.update(parse_still_name(path))
        pick = picks.get(entry.get("frame"))
        if pick:
            entry["t"] = pick["t"]
            entry["scene"] = pick["scene"]
            entry["source_sharpness"] = pick["sharpness"]
            entry["window"] = pick["window"]
            # The whole reason this tool exists: the still came from these
            # frames, not from the one frame it is centred on.
            entry["source_frames"] = list(range(pick["window"][0], pick["window"][1] + 1))
        entries.append(entry)

    entries.sort(key=lambda e: (e.get("frame", 1 << 30), e.get("seed", -1)))
    return entries


def _fit(image: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    """Scale to fit inside the cell, preserving aspect, padding the remainder."""
    h, w = image.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                     interpolation=cv2.INTER_AREA)
    canvas = np.full((cell_h, cell_w, 3), BACKGROUND, np.uint8)
    y = (cell_h - new.shape[0]) // 2
    x = (cell_w - new.shape[1]) // 2
    canvas[y:y + new.shape[0], x:x + new.shape[1]] = new
    return canvas


def build_contact_sheet(entries: list[dict], *, columns: int = 5, thumb_width: int = 420,
                        title: str | None = None) -> np.ndarray:
    """Lay the stills out in a labelled grid. Returns a BGR image."""
    if not entries:
        raise ValueError("no stills to lay out")

    images = []
    for entry in entries:
        img = cv2.imread(entry["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"cannot read still: {entry['path']}")
        images.append(img)

    # Size cells from the median aspect so one odd-shaped still cannot stretch
    # the whole grid.
    aspect = float(np.median([img.shape[0] / img.shape[1] for img in images]))
    cell_w = thumb_width
    cell_h = int(round(cell_w * aspect))

    scale = cell_w / 420.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = int(26 * scale)
    pad = int(10 * scale)
    label_h = line_h * 2 + pad
    title_h = int(52 * scale) if title else 0

    columns = max(1, min(columns, len(entries)))
    rows = (len(entries) + columns - 1) // columns

    sheet = np.full((title_h + rows * (cell_h + label_h), columns * cell_w, 3),
                    BACKGROUND, np.uint8)

    if title:
        cv2.putText(sheet, title, (pad, int(34 * scale)), font, 0.72 * scale, TEXT,
                    max(1, int(2 * scale)), cv2.LINE_AA)

    for n, (entry, img) in enumerate(zip(entries, images)):
        r, c = divmod(n, columns)
        y0 = title_h + r * (cell_h + label_h)
        x0 = c * cell_w
        sheet[y0:y0 + cell_h, x0:x0 + cell_w] = _fit(img, cell_w, cell_h)

        frame = entry.get("frame")
        primary = f"#{n:02d}"
        if frame is not None:
            primary += f"  f{frame}"
        if entry.get("t") is not None:
            primary += f"  {entry['t']:.2f}s"
        if entry.get("scene") is not None:
            primary += f"  scene {entry['scene']}"

        bits = []
        if entry.get("variant"):
            bits.append(entry["variant"])
        if entry.get("seed") is not None:
            bits.append(f"seed {entry['seed']}")
        if entry.get("source_frames"):
            w = entry["window"]
            bits.append(f"from {len(entry['source_frames'])} frames [{w[0]}..{w[1]}]")
        if img is not None:
            bits.append(f"{img.shape[1]}x{img.shape[0]}")
        secondary = "   ".join(bits)

        ty = y0 + cell_h + line_h - int(6 * scale)
        cv2.putText(sheet, primary, (x0 + pad, ty), font, 0.56 * scale, TEXT,
                    max(1, int(1 * scale)), cv2.LINE_AA)
        cv2.putText(sheet, secondary, (x0 + pad, ty + line_h - int(4 * scale)), font,
                    0.46 * scale, TEXT_DIM, max(1, int(1 * scale)), cv2.LINE_AA)

    return sheet


def build_manifest(entries: list[dict], *, selection: dict | None = None,
                   restore_params: dict | None = None,
                   contact_sheet: str | None = None) -> dict:
    """The record of what was produced and, crucially, what each still came from."""
    stills = []
    for entry in entries:
        record = {k: entry[k] for k in
                  ("file", "frame", "scene", "t", "seed", "window", "source_frames",
                   "source_sharpness", "width", "height", "restored")
                  if k in entry}
        stills.append(record)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "still_count": len(stills),
        "stills": stills,
    }
    if contact_sheet:
        manifest["contact_sheet"] = contact_sheet
    if restore_params:
        manifest["restore_params"] = restore_params
    if selection:
        manifest["video"] = selection.get("video")
        manifest["content_box"] = selection.get("content_box")
        manifest["select"] = selection.get("select")
        manifest["scenes"] = selection.get("scenes")
    return manifest


def write_sheet(sheet: np.ndarray, path, quality: int = 92) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return path


def write_manifest(manifest: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
