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

`preview` swaps stage 3 for a straight capture of each pick's centre frame: same
filenames, same sheet, same manifest, no SeedVR2 and no GPU. It exists so the
picks can be reviewed before paying for the restore. Because the two modes write
to the same paths, the manifest records per still whether it was actually
restored (`"restored"`), and that record -- not the current run's mode -- is what
later runs trust. Without it a mixed directory would silently claim restored
stills that are raw grabs, or the reverse.

`upscale_stills` is the other half of that workflow: having reviewed the previews
and deleted the ones they did not want, the user re-runs, and only the files
still in `stills/` are restored -- each written as `<name>_upscaled.png` beside
its original. It is the one stage driven by the folder rather than by the
selection, because the deletions *are* the instruction.
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
    is_upscaled,
    parse_still_name,
    still_name,
    upscaled_name,
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

    A manifest written before `preview` existed has no `restored` field and
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


def _kept_stills(stills_dir: Path, selection: dict, was_restored: dict, seed: int,
                 force: bool, log) -> list:
    """
    Work list for --upscale-stills: whatever is still in `stills_dir`.

    This is the one stage driven by the folder rather than by the selection.
    That inversion is the whole point -- the user curates by deleting, so the
    files that survive ARE the instruction, and a pick with no file left is a
    pick that was rejected.

    The selection is still needed, for the window each surviving still was
    centred on: that is what makes this a temporal restore rather than a
    single-image upscale, and it cannot be recovered from the PNG.
    """
    picks = {p["frame"]: p for p in selection["picks"]}
    todo, orphans, done = [], [], 0
    for path in sorted(stills_dir.glob("*.png")):
        if is_upscaled(path):
            continue
        # A still the manifest already calls restored is the real thing; running
        # it through again would restore a restored image. Unknown files default
        # to "needs it" -- this mode was asked for explicitly, and silently doing
        # nothing is a worse answer than doing the work.
        if was_restored.get(path.name) is True:
            done += 1
            continue
        pick = picks.get(parse_still_name(path).get("frame"))
        if pick is None:
            orphans.append(path.name)
            continue
        out = upscaled_name(path)
        if out.exists() and not force:
            continue
        todo.append((pick, seed, out))

    if done:
        log(f"upscale: {done} still(s) already restored, left alone")
    if orphans:
        log(f"upscale: {len(orphans)} still(s) match no pick in the selection and "
            f"carry no window, so they cannot be restored: "
            f"{orphans[:3]}{' ...' if len(orphans) > 3 else ''}")
    # The easy mistake: a bare `--preview` writes to preview/, not stills/, so
    # curating those and coming here finds an empty folder and would otherwise
    # just shrug. Point at the difference rather than let it look like a no-op.
    if not todo and not done and not any(stills_dir.glob("*.png")):
        quick = stills_dir.parent / "preview"
        if any(quick.glob("*.png")):
            log(f"upscale: stills/ is empty, but {quick.name}/ has frames in it. "
                "Those come from a bare --preview, which is only a quick look and "
                "feeds nothing. Re-run with --preview and a selection option "
                "(--interval, --seconds_per_still, ...) to fill stills/, curate "
                "that, then come back here.")
    return todo


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
                 seeds=1, force=False, preview=False, upscale_stills=False,
                 log=print) -> dict:
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
    s = selection["select"]
    log(f"select: {s['selected']} picks, "
        + (f"one per {s['interval_s']}s interval" if s.get("mode") == "interval"
           else f"quota {s['scene_quota']}"))

    # --- stage 3 -------------------------------------------------------------
    box = selection["content_box"]
    stem = video.stem[:32]
    if preview:
        # A seed only means something to the restorer; N variants of a decoded
        # frame would be N identical files under different names.
        if seeds > 1:
            log(f"preview: --seeds {seeds} ignored, capture is deterministic")
        seeds = 1

    stage = "preview" if preview else "upscale" if upscale_stills else "restore"
    todo, stale = [], []
    if upscale_stills:
        todo = _kept_stills(stills_dir, selection, was_restored,
                            (restore_opts or {}).get("seed", 42), force, log)
        total = len(todo)
    else:
        for pick in selection["picks"]:
            for seed_index in range(seeds):
                seed = (restore_opts or {}).get("seed", 42) + seed_index
                name = still_name(stem, pick["scene"], pick["frame"],
                                  seed=seed if seeds > 1 else None)
                path = stills_dir / name
                if path.exists() and not force:
                    # An existing still counts as done only if it was made the
                    # way this run is making them. A raw grab is not a restored
                    # still.
                    if not preview and not was_restored.get(name, True):
                        stale.append(name)
                    continue
                todo.append((pick, seed, path))
        total = len(selection["picks"]) * seeds

    if stale:
        raise SystemExit(
            f"{len(stale)} still(s) in {stills_dir} were captured with --preview "
            f"and are not restored (e.g. {stale[0]}).\n"
            "Restoring over them would discard the previews you are reviewing, so "
            "this run stops instead.\n"
            "Delete the ones you do not want and re-run with --upscale-stills to "
            "restore only what is left, or pass --force to redo the whole job.")

    if not todo:
        log(f"{stage}: nothing to do" if upscale_stills
            else f"{stage}: all {total} stills already present, nothing to do")
    elif preview:
        log(f"preview: {len(todo)} of {total} frames to capture (no restore)")
        # Decode the pick's whole window and keep its centre, rather than
        # seeking straight to the one frame: identical work to what the restore
        # path does, so the preview is the same image a later restore centres on.
        for done, (pick, _seed, path) in enumerate(todo, 1):
            lo, hi = pick["window"]
            frames = decode_window(str(video), lo, hi, box)
            _atomic_png(path, frames[len(frames) // 2])
            log(f"preview: [{done}/{len(todo)}] f{pick['frame']} scene {pick['scene']} "
                f"-> {path.name}")
    else:
        # Identical work either way: both modes restore a window and write its
        # centre. Only the list differs -- picks for a normal run, surviving
        # files for --upscale-stills.
        log(f"{stage}: {len(todo)} of {total} stills to produce" if not upscale_stills
            else f"upscale: {len(todo)} kept still(s) to restore")
        opts = dict(restore_opts or {})
        opts.pop("seed", None)
        # The worker is only started when there is real work: a fully resumed
        # run should not pay ~10s of model materialisation to discover that.
        with SeedVR2Restorer(**(worker_opts or {})) as restorer:
            log(f"{stage}: worker ready ({restorer.info.get('device')})")
            done = 0
            for pick, seed, path in todo:
                lo, hi = pick["window"]
                frames = decode_window(str(video), lo, hi, box)
                centre = restorer.restore(frames, seed=seed, **opts)
                _atomic_png(path, centre)
                done += 1
                log(f"{stage}: [{done}/{len(todo)}] f{pick['frame']} scene {pick['scene']} "
                    f"seed {seed} -> {path.name}")

    # What this run actually produced, so the manifest can tell the truth about
    # a directory that has been through both modes.
    was_restored.update({path.name: not preview for _, _, path in todo})

    # --- stage 4 -------------------------------------------------------------
    entries = collect_stills(stills_dir, selection)
    if not entries:
        # Reachable since --upscale-stills made stills/ a folder the user edits:
        # delete everything and there is nothing to lay out. Leave the existing
        # sheet and manifest alone rather than overwriting them with emptiness --
        # they still describe what was there, and this run produced nothing to
        # replace that with.
        log(f"sheet: no stills in {stills_dir}, nothing to lay out; "
            "the existing contact sheet and manifest are left as they are")
        return build_manifest([], selection=selection)

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
    # After --upscale-stills the sheet holds before/after pairs at the same
    # frame. Say which is which, but only then -- on a normal sheet every cell
    # would carry the same word and it would be noise.
    if raw and raw != len(entries):
        for entry in entries:
            entry["variant"] = "upscaled" if entry["restored"] else "source"
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
