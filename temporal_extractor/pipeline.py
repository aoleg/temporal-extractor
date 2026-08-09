"""
The whole thing, end to end: scan -> select -> restore -> sheet.

Layout of an output directory:

    <out>/
      stills/            the deliverable
      contact_sheet.jpg  for review
      manifest.json      what came from where
      work/              scan.json, select.json

One directory holds everything, so a run is one thing to move or delete and
nothing is orphaned if it dies partway.

Resume is derived from those artifacts rather than from a separate progress
file. A still that exists on disk is a still that is done; a scan that exists is
a scan that need not be redone. There is no bookkeeping to fall out of sync with
reality, and a run interrupted anywhere picks up where it stopped. Everything is
written to a temporary name and renamed into place, so a process killed
mid-write leaves no half-file that would later be mistaken for finished work.

`no_restore` swaps stage 3 for a straight capture of each pick's centre frame:
same filenames, same sheet, same manifest, no SeedVR2 and no GPU. It exists so
the picks can be reviewed before paying for the restore. Because the two modes
write to the same paths, the manifest records per still whether it was actually
restored (`"restored"`), and that record -- not the current run's mode -- is what
later runs trust. Without it a mixed directory would silently claim restored
stills that are raw grabs, or the reverse.
"""

import json
import os
from pathlib import Path

import cv2

from .frames import read_png_rgb, write_png_rgb
from .restore import SeedVR2Restorer
from .scan import read_scan, scan_video, write_scan
from .select import read_selection, select_frames, write_selection
from .sheet import (
    build_contact_sheet,
    build_manifest,
    collect_stills,
    still_name,
    write_manifest,
    write_sheet,
)


def _atomic_png(path: Path, rgb) -> None:
    tmp = path.with_suffix(".part.png")
    write_png_rgb(tmp, rgb)
    os.replace(tmp, path)


def _atomic_json(writer, data, path: Path) -> None:
    tmp = path.with_suffix(".part.json")
    writer(data, tmp)
    os.replace(tmp, path)


def _prior_restored(manifest_path: Path) -> dict:
    """
    Read `{filename: was_it_restored}` back out of an existing manifest.

    A manifest written before `no_restore` existed has no `restored` field and
    could only describe restored stills, so a missing field reads as True.
    Anything unreadable reads as "nothing known", which leaves the caller to
    treat what is on disk as restored -- the pre-existing assumption.
    """
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {s["file"]: s.get("restored", True)
            for s in data.get("stills", []) if "file" in s}


def decode_window(video: str, lo: int, hi: int, box: dict) -> list:
    """Decode frames [lo..hi] inclusive as RGB, cropped to the content box."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise OSError(f"cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    frames = []
    for index in range(lo, hi + 1):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise OSError(f"could not decode frame {index} of {video}")
        frame = frame[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]]
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def choose_variants(variants: list[dict]) -> list[dict]:
    """
    Decide which of a pick's seed variants to keep.

    Currently keeps all of them: with the default of one seed there is nothing to
    choose between, and when sweeping, seeing every variant on the sheet is the
    point. This is the seam for a future --keep sharpest mode, which would score
    the variants and return one; the caller writes whatever comes back, and the
    sheet and manifest follow automatically.
    """
    return variants


def run_pipeline(video, out_dir=None, *, select_opts=None, restore_opts=None,
                 worker_opts=None, sheet_opts=None, scan_opts=None,
                 seeds=1, force=False, no_restore=False, log=print) -> dict:
    """Run every stage. Returns the manifest."""
    video = Path(video).resolve()
    out_dir = Path(out_dir) if out_dir else video.parent / video.stem
    work = out_dir / "work"
    stills_dir = out_dir / "stills"
    manifest_path = out_dir / "manifest.json"
    stills_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    was_restored = {} if force else _prior_restored(manifest_path)

    # --- stage 1 -------------------------------------------------------------
    scan_path = work / "scan.json"
    meta = None
    if scan_path.exists() and not force:
        try:
            meta = read_scan(scan_path)
            log(f"scan: reusing {scan_path}")
        except ValueError as exc:
            # A stale schema is not a reason to stop; it is a reason to redo it.
            log(f"scan: {exc}")
    if meta is None:
        log(f"scan: decoding {video.name} ...")
        meta = scan_video(video, **(scan_opts or {}))
        _atomic_json(write_scan, meta, scan_path)
        log(f"scan: {meta['video']['frame_count']:,} frames, {len(meta['scenes'])} scenes, "
            f"{meta['scan']['elapsed_s']:.1f}s")

    # --- stage 2 -------------------------------------------------------------
    # Recomputed unless it is already there: cheap, but reusing keeps the picks
    # stable across a resume so existing stills stay meaningful.
    select_path = work / "select.json"
    selection = None
    if select_path.exists() and not force:
        try:
            selection = read_selection(select_path)
            log(f"select: reusing {select_path}")
        except ValueError as exc:
            log(f"select: {exc}")
    if selection is None:
        selection = select_frames(meta, **(select_opts or {}))
        _atomic_json(write_selection, selection, select_path)
    log(f"select: {selection['select']['selected']} picks, "
        f"quota {selection['select']['scene_quota']}")

    # --- stage 3 -------------------------------------------------------------
    box = selection["content_box"]
    stem = video.stem[:32]
    if no_restore:
        # A seed only means something to the restorer; N variants of a decoded
        # frame would be N identical files under different names.
        if seeds > 1:
            log(f"extract: --seeds {seeds} ignored, capture is deterministic")
        seeds = 1

    stage = "extract" if no_restore else "restore"
    todo, stale = [], []
    for pick in selection["picks"]:
        for seed_index in range(seeds):
            seed = (restore_opts or {}).get("seed", 42) + seed_index
            name = still_name(stem, pick["scene"], pick["frame"],
                              seed=seed if seeds > 1 else None)
            path = stills_dir / name
            if path.exists() and not force:
                # An existing still counts as done only if it was made the way
                # this run is making them. A raw grab is not a restored still.
                if not no_restore and not was_restored.get(name, True):
                    stale.append(name)
                continue
            todo.append((pick, seed, path))

    if stale:
        raise SystemExit(
            f"{len(stale)} still(s) in {stills_dir} were captured with --no_restore "
            f"and are not restored (e.g. {stale[0]}).\n"
            "Restoring over them would discard the previews you are reviewing, so "
            "this run stops instead.\n"
            "Delete them, or re-run with --force to redo the whole job.")

    total = len(selection["picks"]) * seeds
    if not todo:
        log(f"{stage}: all {total} stills already present, nothing to do")
    elif no_restore:
        log(f"extract: {len(todo)} of {total} frames to capture (no restore)")
        # Decode the pick's whole window and keep its centre, rather than
        # seeking straight to the one frame: identical work to what the restore
        # path does, so the preview is the same image a later restore centres on.
        for done, (pick, _seed, path) in enumerate(todo, 1):
            lo, hi = pick["window"]
            frames = decode_window(str(video), lo, hi, box)
            _atomic_png(path, frames[len(frames) // 2])
            log(f"extract: [{done}/{len(todo)}] f{pick['frame']} scene {pick['scene']} "
                f"-> {path.name}")
    else:
        log(f"restore: {len(todo)} of {total} stills to produce")
        opts = dict(restore_opts or {})
        opts.pop("seed", None)
        # The worker is only started when there is real work: a fully resumed
        # run should not pay ~10s of model materialisation to discover that.
        with SeedVR2Restorer(**(worker_opts or {})) as restorer:
            log(f"restore: worker ready ({restorer.info.get('device')})")
            done = 0
            for pick, seed, path in todo:
                lo, hi = pick["window"]
                frames = decode_window(str(video), lo, hi, box)
                centre = restorer.restore(frames, seed=seed, **opts)
                _atomic_png(path, centre)
                done += 1
                log(f"restore: [{done}/{len(todo)}] f{pick['frame']} scene {pick['scene']} "
                    f"seed {seed} -> {path.name}")

    # What this run actually produced, so the manifest can tell the truth about
    # a directory that has been through both modes.
    was_restored.update({path.name: not no_restore for _, _, path in todo})

    # --- stage 4 -------------------------------------------------------------
    entries = collect_stills(stills_dir, selection)
    by_pick = {}
    for entry in entries:
        by_pick.setdefault(entry.get("frame"), []).append(entry)
    entries = [e for group in by_pick.values() for e in choose_variants(group)]
    entries.sort(key=lambda e: (e.get("frame", 1 << 30), e.get("seed", -1)))

    for entry in entries:
        img = read_png_rgb(entry["path"])
        entry["height"], entry["width"] = img.shape[:2]
        # Anything on disk that no manifest accounts for predates this field and
        # can only have come from a restore.
        entry["restored"] = was_restored.get(entry["file"], True)

    raw = sum(1 for e in entries if not e["restored"])
    sheet_path = out_dir / "contact_sheet.jpg"
    opts = dict(sheet_opts or {})
    suffix = "   (not restored)" if raw == len(entries) else ""
    title = opts.pop("title", None) or f"{video.name}   {len(entries)} stills{suffix}"
    quality = opts.pop("quality", 92)
    sheet = build_contact_sheet(entries, title=title, **opts)
    write_sheet(sheet, sheet_path, quality=quality)
    manifest = build_manifest(entries, selection=selection,
                              # Generation parameters no restore ever used would
                              # be a false record of how these stills were made.
                              restore_params=None if raw == len(entries) else restore_opts,
                              contact_sheet=sheet_path.name)
    _atomic_json(write_manifest, manifest, manifest_path)

    log(f"sheet: {len(entries)} stills{f' ({raw} not restored)' if raw else ''} -> {sheet_path}")
    log(f"manifest -> {manifest_path}")
    return manifest
