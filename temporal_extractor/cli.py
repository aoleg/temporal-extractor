"""
Command line for every stage.

Each stage is runnable on its own so it can be debugged and swept in isolation,
and `run` chains all four. Invoked through extract.bat / extract.py.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

from . import config as cfg
from .frames import content_box, crop, read_png_rgb, write_png_rgb
from .restore import SeedVR2Restorer
from .scan import (
    DEFAULT_MIN_SCENE_LEN,
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_WORKERS,
    read_scan,
    scan_video,
    write_scan,
)
from .timecode import format_timestamp, parse_timestamp
from .pipeline import run_pipeline
from .preview import DEFAULT_PREVIEW_COUNT, PREVIEW_WINDOW_S, quick_preview
from .select import (
    DEFAULT_GAP_FRACTION,
    DEFAULT_HASH_DISTANCE,
    DEFAULT_MAX_HIGHLIGHT_FRAC,
    DEFAULT_MIN_GAP_S,
    DEFAULT_MIN_LUMA,
    DEFAULT_MIN_SHARPNESS_FRAC,
    DEFAULT_PER_SCENE_MAX,
    DEFAULT_SECONDS_PER_STILL,
    DEFAULT_WINDOW,
    INTERVAL_STEP_S,
    MIN_INTERVAL_S,
    hamming,
    read_selection,
    select_frames,
    write_selection,
)
from .sheet import (
    build_contact_sheet,
    build_manifest,
    collect_stills,
    write_manifest,
    write_sheet,
)


# Extensions --preview will look for when `run`'s video argument is a folder.
# cv2.VideoCapture will happily attempt to open anything; this list exists only
# to decide what counts as "a video" for a directory listing, not to validate
# the file. Matched case-insensitively (.MPG and .mpg both count).
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".flv",
    ".mpg", ".mpeg", ".m4v", ".ts", ".m2ts", ".mts", ".3gp",
}


def _find_videos(folder: Path) -> list:
    """First-level video files in `folder`, by extension, case-insensitive."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


# Options that only mean anything to the full scan->select pipeline. Passing any
# of them alongside --preview says "I have an opinion about how frames are
# chosen", so the quick path steps aside and the real one runs (still without
# restoring). Sheet and output options are deliberately absent: --out, --columns
# and friends apply to a quick preview perfectly well.
PIPELINE_FLAGS = frozenset({
    "--interval", "--window", "--seconds_per_still", "--per_scene_max",
    "--hash_distance", "--min_gap", "--gap_fraction", "--min_sharpness_frac",
    "--min_luma", "--max_highlight_frac",
    "--scene_threshold", "--min_scene_len", "--no_content_box", "--workers",
    "--segment",
})


def _configured(argv) -> set:
    """Which PIPELINE_FLAGS the user actually typed.

    Read off argv rather than compared against argparse defaults, so that
    explicitly passing a flag's own default value still counts as configuring
    it -- `--min_luma 24.0` means the user is thinking about selection, and
    silently giving them the quick path instead would be a surprise.
    """
    return {token.split("=", 1)[0] for token in argv} & PIPELINE_FLAGS


def preview_count(text: str) -> int:
    """`--preview`'s optional count: a positive integer."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a frame count, got {text!r}"
            + (f" -- if that is your video, put it before --preview"
               if not text.startswith("-") else ""))
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {text}")
    return value


def _load_window(png_dir: Path, window: int):
    paths = sorted(p for p in png_dir.iterdir()
                   if p.suffix.lower() == ".png" and not p.name.startswith("restored_"))
    if not paths:
        raise SystemExit(f"no PNGs in {png_dir}")
    if window and window < len(paths):
        k = (len(paths) - window) // 2
        paths = paths[k:k + window]
    return paths, [read_png_rgb(p) for p in paths]


def cmd_scan(args) -> int:
    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"no such video: {video}")

    meta = scan_video(video, **scan_kwargs(args), show_progress=not args.quiet)

    out = Path(args.out) if args.out else video.with_suffix(".scan.json")
    write_scan(meta, out)

    v, box, scenes = meta["video"], meta["content_box"], meta["scenes"]
    print(f"{v['filename']}: {v['width']}x{v['height']} @ {v['fps']:.3f}fps, "
          f"{v['frame_count']} frames, {v['duration_s']:.1f}s")
    segs = meta["scan"]["segments"]
    if segs:
        print(f"segments: {len(segs)} requested, " + ", ".join(
            f"{format_timestamp(s['start_t'])}-{format_timestamp(s['end_t'])}" for s in segs))
    if (box["w"], box["h"]) != (v["width"], v["height"]):
        print(f"content box: {box['w']}x{box['h']} at ({box['x']},{box['y']}) "
              f"-- {100 * (1 - box['w'] * box['h'] / (v['width'] * v['height'])):.0f}% of the frame is matte")
    else:
        print("content box: full frame (no pillar/letterboxing)")
    print(f"scenes: {len(scenes)} (threshold {meta['scan']['scene_threshold']}, "
          f"min length {meta['scan']['min_scene_len']} frames)")

    for scene in scenes[:args.show_scenes]:
        print(f"  scene {scene['id']:>3}  frames {scene['start_i']:>6}-{scene['end_i']:<6} "
              f"{scene['start_t']:>8.2f}s-{scene['end_t']:<8.2f}s  "
              f"n={scene['frame_count']:<5} best={scene['best_frame']:<6} "
              f"sharp={scene['best_sharpness']:.1f} (mean {scene['mean_sharpness']:.1f})")
    if len(scenes) > args.show_scenes:
        print(f"  ... {len(scenes) - args.show_scenes} more")

    print(f"scanned in {meta['scan']['elapsed_s']:.1f}s -> {out}")
    return 0


def scan_kwargs(args) -> dict:
    """The scan-stage knobs, shared by `scan` and `run`."""
    return {
        "scene_threshold": args.scene_threshold,
        "min_scene_len": args.min_scene_len,
        "detect_content_box": not args.no_content_box,
        "workers": args.workers,
        "segments": args.segment,
    }


def interval_seconds(text: str) -> float:
    """`--interval`'s type: at least MIN_INTERVAL_S, on the INTERVAL_STEP_S grid."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}")
    if value < MIN_INTERVAL_S:
        raise argparse.ArgumentTypeError(f"must be at least {MIN_INTERVAL_S}, got {text}")
    steps = value / INTERVAL_STEP_S
    if abs(steps - round(steps)) > 1e-6:
        raise argparse.ArgumentTypeError(
            f"must be a multiple of {INTERVAL_STEP_S}, got {text}")
    return round(round(steps) * INTERVAL_STEP_S, 1)


def select_kwargs(args) -> dict:
    """The select-stage knobs, shared by `select` and `run`."""
    if args.interval:
        # Interval mode uses none of the others -- no quotas, no gaps, no
        # rejection. Forwarding them anyway would only hide that they do nothing.
        return {"window": args.window, "interval": args.interval}
    return {
        "window": args.window,
        "seconds_per_still": args.seconds_per_still,
        "per_scene_max": args.per_scene_max,
        "hash_distance": args.hash_distance,
        "min_gap_s": args.min_gap,
        "gap_fraction": args.gap_fraction,
        "min_sharpness_frac": args.min_sharpness_frac,
        "min_luma": args.min_luma,
        "max_highlight_frac": args.max_highlight_frac,
    }


def _print_picks(sel: dict) -> None:
    s = sel["select"]
    print()
    print(f"  {'frame':>7} {'time':>9} {'scene':>6} {'sharpness':>10}  window")
    for p in sel["picks"]:
        print(f"  {p['frame']:>7} {p['t']:>8.2f}s {p['scene']:>6} {p['sharpness']:>10.1f}  "
              f"[{p['window'][0]}..{p['window'][1]}]")

    # The closest pair is the honest measure of how varied the set is. Interval
    # mode enforces no floor, but the number still says how much of a dense
    # sample is near-duplicate.
    if len(sel["picks"]) > 1:
        pairs = [(hamming(a["dhash"], b["dhash"]), a["frame"], b["frame"])
                 for i, a in enumerate(sel["picks"]) for b in sel["picks"][i + 1:]]
        d, fa, fb = min(pairs)
        floor = f" (floor {s['hash_distance']})" if "hash_distance" in s else ""
        print(f"\nclosest pair: f{fa} and f{fb} at dHash distance {d}{floor}")


def print_selection(sel: dict, picks: bool = False) -> None:
    s = sel["select"]
    if s.get("mode") == "interval":
        print(f"{sel['video']['filename']}: {s['selected']} picks, the sharpest frame of "
              f"every {s['interval_s']}s "
              f"({s['intervals']} intervals across {s['eligible_frames']} scanned frames, "
              f"window {s['window']})")
        print("interval mode: no scene quotas, no dedupe, no weak-frame rejection -- "
              "every interval yields one pick")
        if s["window_unsafe"]:
            print(f"{s['window_unsafe']} pick(s) sit too close to a cut for their restore "
                  "window to stay in one scene; their interval offered nothing further out")
        if picks:
            _print_picks(sel)
        return

    print(f"{sel['video']['filename']}: {s['selected']} picks from "
          f"{s['eligible_frames']} eligible frames "
          f"(window {s['window']}, one still per {s['seconds_per_still']}s of scene, "
          f"max {s['per_scene_max']} per scene)")
    weak = s["rejected_weak"]
    if weak["too_dark"] or weak["too_soft"] or weak["blown_highlights"]:
        print(f"rejected as unusable: {weak['too_soft']} too soft "
              f"(< {s['min_sharpness_frac']:.0%} of their scene's best), "
              f"{weak['too_dark']} too dark (luma < {s['min_luma']}), "
              f"{weak['blown_highlights']} blown highlights "
              f"(> {s['max_highlight_frac']:.0%} clipped)")
    if s["scenes_unusable"]:
        print(f"scenes with no usable frame: {s['scenes_unusable']}")
    if s["filled"]:
        print(f"{s['filled']} pick(s) came from the within-scene maximin fill")
    print(f"scene quota: {s['scene_quota']}   gap floor per scene (frames): {s['scene_gap_frames']}")

    if picks:
        _print_picks(sel)


def cmd_select(args) -> int:
    meta = read_scan(args.scan)
    sel = select_frames(meta, **select_kwargs(args))

    out = Path(args.out) if args.out else Path(args.scan).with_suffix("").with_suffix(".select.json")
    write_selection(sel, out)

    print_selection(sel, picks=True)
    print(f"\n-> {out}")
    return 0


def cmd_sheet(args) -> int:
    stills_dir = Path(args.stills_dir)
    if not stills_dir.is_dir():
        raise SystemExit(f"not a directory: {stills_dir}")

    selection = read_selection(args.selection) if args.selection else None
    entries = collect_stills(stills_dir, selection)
    if not entries:
        raise SystemExit(f"no PNG stills in {stills_dir}")

    unmatched = [e["file"] for e in entries if "frame" not in e]
    if unmatched:
        print(f"warning: {len(unmatched)} still(s) do not follow the "
              f"<name>_s<scene>_f<frame>[_seed<n>].png convention, so they carry no "
              f"provenance: {unmatched[:3]}{' ...' if len(unmatched) > 3 else ''}")
    if selection:
        missing = [e["file"] for e in entries if "frame" in e and "source_frames" not in e]
        if missing:
            print(f"warning: {len(missing)} still(s) name a frame absent from the selection; "
                  "is this the right selection JSON?")

    for entry in entries:
        img = cv2.imread(entry["path"], cv2.IMREAD_COLOR)
        if img is not None:
            entry["height"], entry["width"] = img.shape[:2]

    title = args.title
    if title is None:
        name = (selection or {}).get("video", {}).get("filename", stills_dir.name)
        title = f"{name}   {len(entries)} stills"

    sheet_path = Path(args.out) if args.out else stills_dir.parent / f"{stills_dir.name}_sheet.jpg"
    manifest_path = (Path(args.manifest) if args.manifest
                     else stills_dir.parent / f"{stills_dir.name}_manifest.json")

    sheet = build_contact_sheet(entries, columns=args.columns,
                                thumb_width=args.thumb_width, title=title)
    write_sheet(sheet, sheet_path, quality=args.quality)
    manifest = build_manifest(entries, selection=selection,
                              contact_sheet=sheet_path.name)
    write_manifest(manifest, manifest_path)

    with_prov = sum(1 for e in entries if "source_frames" in e)
    print(f"{len(entries)} stills, {with_prov} with source-frame provenance")
    print(f"contact sheet {sheet.shape[1]}x{sheet.shape[0]} -> {sheet_path}")
    print(f"manifest -> {manifest_path}")
    return 0


def cmd_restore(args) -> int:
    png_dir = Path(args.png_dir)
    paths, frames = _load_window(png_dir, args.window)
    print(f"window: {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}, "
          f"centre = {paths[len(paths) // 2].name}")

    if args.crop_pillarbox:
        box = content_box(frames)
        if box[2:] != (frames[0].shape[1], frames[0].shape[0]):
            frames = crop(frames, box)
            print(f"cropped pillarbox -> {box[2]}x{box[3]} at ({box[0]},{box[1]})")

    out = Path(args.out) if args.out else png_dir.parent / f"{png_dir.name}_restored_centre.png"
    stem, suffix = out.with_suffix("").name, out.suffix or ".png"

    print(f"params: resolution={args.resolution} cfg_scale={args.cfg_scale} "
          f"input_noise={args.input_noise_scale} latent_noise={args.latent_noise_scale} "
          f"seeds={args.seed}..{args.seed + args.seeds - 1}")

    with SeedVR2Restorer(
        attention_mode=args.attention_mode,
        encode_tiled=args.vae_encode_tiled,
        decode_tiled=not args.no_decode_tiling,
        blocks_to_swap=args.blocks_to_swap,
        quiet=args.quiet,
    ) as restorer:
        print(f"worker ready: pid {restorer.info.get('pid')}, "
              f"torch {restorer.info.get('torch')}, {restorer.info.get('device')}")
        for i in range(args.seeds):
            seed = args.seed + i
            t0 = time.time()
            centre = restorer.restore(
                frames,
                resolution=args.resolution,
                seed=seed,
                cfg_scale=args.cfg_scale,
                input_noise_scale=args.input_noise_scale,
                latent_noise_scale=args.latent_noise_scale,
                color_correction=args.color_correction,
            )
            elapsed = time.time() - t0
            path = out if args.seeds == 1 else out.with_name(f"{stem}_seed{seed}{suffix}")
            write_png_rgb(path, centre)
            print(f"seed {seed}: {centre.shape[1]}x{centre.shape[0]} in {elapsed:.1f}s -> {path}")
    return 0


def cmd_doctor(args) -> int:
    """Check the configuration before a long run discovers it is broken."""
    print(f"config file: {cfg.ENV_FILE}"
          f"{'' if cfg.ENV_FILE.exists() else '   (MISSING -- run install.bat)'}")
    shown = lambda v: v if v is not None else "(not set)"
    print(f"  SEEDVR2_REPO   {shown(cfg.SEEDVR2_REPO)}")
    print(f"  SEEDVR2_PYTHON {shown(cfg.SEEDVR2_PYTHON)}")
    print(f"  MODEL_DIR      {shown(cfg.MODEL_DIR)}")
    print(f"  DIT_MODEL      {cfg.DIT_MODEL}"
          f"{'' if cfg.setting('DIT_MODEL') else '   (default)'}")
    print(f"  VAE_MODEL      {cfg.VAE_MODEL}"
          f"{'' if cfg.setting('VAE_MODEL') else '   (default)'}")
    print()

    problems = cfg.check()
    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        print(f"\nEdit {cfg.ENV_FILE} and try again.")
        return 1

    print("paths OK. Starting the restore worker to confirm it loads ...")
    try:
        with SeedVR2Restorer(quiet=True) as restorer:
            info = restorer.info
            print(f"  worker OK: pid {info.get('pid')}, torch {info.get('torch')}, "
                  f"{info.get('device')}")
    except Exception as exc:
        print(f"  worker FAILED: {exc}")
        return 1

    print("\nEverything is wired up.")
    return 0


def _run_pipeline_kwargs(args) -> dict:
    """Everything run_pipeline() needs except video/out -- identical for every
    video when --video is a folder, so this is built once and reused."""
    return dict(
        scan_opts={**scan_kwargs(args), "show_progress": not args.quiet},
        select_opts=select_kwargs(args),
        restore_opts={
            "resolution": args.resolution,
            "seed": args.seed,
            "cfg_scale": args.cfg_scale,
            "input_noise_scale": args.input_noise_scale,
            "latent_noise_scale": args.latent_noise_scale,
            "color_correction": args.color_correction,
        },
        worker_opts={
            "attention_mode": args.attention_mode,
            "encode_tiled": args.vae_encode_tiled,
            "decode_tiled": not args.no_decode_tiling,
            "blocks_to_swap": args.blocks_to_swap,
            "quiet": args.quiet,
        },
        sheet_opts={"columns": args.columns, "thumb_width": args.thumb_width,
                    "quality": args.quality},
        seeds=args.seeds,
        force=args.force,
        preview=bool(args.preview),
        upscale_stills=args.upscale_stills,
    )


def _quick_preview_wanted(args) -> bool:
    """Bare `--preview [N]`: no selection or scan option was configured."""
    return bool(args.preview) and not getattr(args, "configured_flags", set())


def _run_one(video: Path, out, args) -> None:
    """One video, by whichever path the options ask for."""
    if _quick_preview_wanted(args):
        quick_preview(video, out, count=args.preview,
                      sheet_opts={"columns": args.columns,
                                  "thumb_width": args.thumb_width,
                                  "quality": args.quality})
    else:
        run_pipeline(video, out, **_run_pipeline_kwargs(args))


def _run_folder(folder: Path, args) -> int:
    """--preview over every video in a folder, one at a time.

    Each video's own default output layout still applies (a folder beside it,
    named after it) unless --out is given, in which case each video gets its
    own subfolder under --out -- sharing one output directory across videos
    would mean each video's stills, sheet and manifest overwriting the last.

    A video that fails does not stop the rest: one corrupt or unreadable file
    in a folder of fifty should not cost the other forty-nine.
    """
    videos = _find_videos(folder)
    if not videos:
        exts = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise SystemExit(f"no videos in {folder} (looked for: {exts})")

    print(f"preview: {len(videos)} video(s) in {folder}")
    out_root = Path(args.out) if args.out else None

    done, failed = [], []
    for i, video in enumerate(videos, 1):
        out = (out_root / video.stem) if out_root else None
        print(f"\n[{i}/{len(videos)}] {video.name}")
        try:
            _run_one(video, out, args)
            done.append(video.name)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed.append(video.name)

    print(f"\npreview: {len(done)} of {len(videos)} video(s) done"
          + (f", {len(failed)} failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def cmd_run(args) -> int:
    video = Path(args.video)
    if video.is_dir():
        if not args.preview:
            raise SystemExit(
                f"{video} is a folder, not a video file.\n"
                "Folder input is only supported with --preview: running every video "
                "in a folder straight through SeedVR2 would mean committing GPU time "
                "to all of them before you have looked at any. Pass --preview to "
                "preview every video in the folder, or point run at one video file."
            )
        return _run_folder(video, args)

    if not video.exists():
        raise SystemExit(f"no such video: {video}")
    _run_one(video, args.out, args)
    return 0


def _add_segment_arg(p) -> None:
    """`--segment FROM TO`, shared verbatim by `scan` and `run`.

    Repeatable, so a feature-length source can name just the handful of scenes
    worth scanning instead of paying for a full decode pass to keep three of
    them. FROM/TO accept H:MM:SS[.sss], MM:SS[.sss] or bare SS[.sss].
    """
    p.add_argument("--segment", nargs=2, action="append", type=parse_timestamp,
                   metavar=("FROM", "TO"), default=None,
                   help="only scan this time range, e.g. --segment 1:15:36 1:20:00 "
                        "(repeat for multiple segments). Accepts H:MM:SS[.sss], "
                        "MM:SS[.sss] or SS[.sss]. Omit entirely to scan the whole video")


def _add_select_args(p) -> None:
    """Selection knobs, shared verbatim by `select` and `run`."""
    g = p.add_argument_group("selection")
    g.add_argument("--interval", type=interval_seconds, default=None, metavar="SECONDS",
                   help=f"interval mode: take the sharpest frame of every N seconds and "
                        f"nothing else. Minimum {MIN_INTERVAL_S}, in steps of {INTERVAL_STEP_S}. "
                        "Bypasses scene detection, quotas, dedupe and weak-frame rejection -- "
                        "every interval yields exactly one pick, good or bad. Every other "
                        "option in this group except --window is ignored")
    g.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                   help=f"restore window size, 4n+1 (default {DEFAULT_WINDOW}). Picks stay far "
                        "enough from a cut that the window remains inside one scene")
    g.add_argument("--seconds_per_still", type=float, default=DEFAULT_SECONDS_PER_STILL,
                   help=f"scene time earning one still (default {DEFAULT_SECONDS_PER_STILL}s). "
                        "Every scene gets at least one regardless")
    g.add_argument("--per_scene_max", type=int, default=DEFAULT_PER_SCENE_MAX,
                   help=f"ceiling per scene (default {DEFAULT_PER_SCENE_MAX}). There is no global "
                        "ceiling: total output is whatever the video's structure earns")
    g.add_argument("--hash_distance", type=int, default=DEFAULT_HASH_DISTANCE,
                   help=f"minimum dHash distance between picks (default {DEFAULT_HASH_DISTANCE}). "
                        "Low on purpose -- a changed angle or newly visible detail is a "
                        "different training example, not a duplicate")
    g.add_argument("--min_gap", type=float, default=DEFAULT_MIN_GAP_S, metavar="SECONDS",
                   help=f"absolute floor between picks (default {DEFAULT_MIN_GAP_S}s)")
    g.add_argument("--gap_fraction", type=float, default=DEFAULT_GAP_FRACTION,
                   help=f"gap floor as a fraction of a scene's natural spacing "
                        f"(default {DEFAULT_GAP_FRACTION})")
    g.add_argument("--min_sharpness_frac", type=float, default=DEFAULT_MIN_SHARPNESS_FRAC,
                   help=f"reject frames below this fraction of their own scene's best sharpness "
                        f"(default {DEFAULT_MIN_SHARPNESS_FRAC})")
    g.add_argument("--min_luma", type=float, default=DEFAULT_MIN_LUMA,
                   help=f"reject frames dimmer than this mean luminance (default {DEFAULT_MIN_LUMA}); "
                        "0 disables")
    g.add_argument("--max_highlight_frac", type=float, default=DEFAULT_MAX_HIGHLIGHT_FRAC,
                   help=f"reject frames with more than this fraction blown out to near-white "
                        f"(default {DEFAULT_MAX_HIGHLIGHT_FRAC}); 0 disables. Judged by clipped-pixel "
                        "fraction, not mean brightness -- a bright but intact frame and an "
                        "overexposed one can share the same mean luma")


def _add_restore_args(p) -> None:
    """Generation and worker knobs, shared by `restore` and `run`."""
    d = cfg.DEFAULTS
    g = p.add_argument_group("generation parameters (sweepable)")
    g.add_argument("--resolution", type=int, default=d["resolution"], help="target SHORT side")
    g.add_argument("--seed", type=int, default=d["seed"], help="base seed")
    g.add_argument("--seeds", type=int, default=1,
                   help="produce N variants per still with consecutive seeds (default 1)")
    g.add_argument("--cfg_scale", type=float, default=d["cfg_scale"],
                   help="classifier-free guidance; default 1.0 (off). Not clamped: only the "
                        "centre frame is kept, so still-image values apply. >1.0 runs both "
                        "DiT branches and roughly doubles phase 2")
    g.add_argument("--input_noise_scale", type=float, default=d["input_noise_scale"])
    g.add_argument("--latent_noise_scale", type=float, default=d["latent_noise_scale"])
    g.add_argument("--color_correction", default=d["color_correction"],
                   choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"])

    m = p.add_argument_group("performance / memory")
    m.add_argument("--attention_mode", default=cfg.WORKER_DEFAULTS["attention_mode"],
                   choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"])
    m.add_argument("--vae_encode_tiled", action="store_true",
                   help="tile VAE encoding; needed above ~1080p or with long windows")
    m.add_argument("--no_decode_tiling", action="store_true")
    m.add_argument("--blocks_to_swap", type=int, default=0, help="offload N DiT blocks to CPU (0-36)")
    m.add_argument("--quiet", action="store_true", help="reduce the worker's progress logging")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="temporal_extractor",
                                 description="Extract high-quality stills from low-quality video.")
    sub = ap.add_subparsers(dest="command", required=True)

    doc = sub.add_parser("doctor", help="check configuration and that the restorer loads")
    doc.set_defaults(func=cmd_doctor)

    s = sub.add_parser("scan", help="stage 1: score every frame, find scenes, write metadata JSON")
    s.add_argument("video", help="input video file")
    s.add_argument("--out", default=None, help="metadata JSON path (default: <video>.scan.json)")
    s.add_argument("--scene_threshold", type=float, default=DEFAULT_SCENE_THRESHOLD,
                   help=f"HSV content delta that counts as a cut (default {DEFAULT_SCENE_THRESHOLD}; "
                        "same scale as PySceneDetect ContentDetector). Lower = more scenes")
    s.add_argument("--min_scene_len", type=int, default=DEFAULT_MIN_SCENE_LEN,
                   help=f"discard cuts producing a scene shorter than N frames (default {DEFAULT_MIN_SCENE_LEN})")
    s.add_argument("--no_content_box", action="store_true",
                   help="skip pillar/letterbox detection and score the full frame")
    s.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"processes to scan with (default {DEFAULT_WORKERS}). OpenCV barely "
                        "threads this work, so the cores are used by splitting the video into "
                        "frame ranges. Raise it on a large CPU; returns flatten past about 12")
    _add_segment_arg(s)
    s.add_argument("--show_scenes", type=int, default=20, help="how many scenes to print")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_scan)

    sel = sub.add_parser("select", help="stage 2: pick which frames to restore, from a scan")
    sel.add_argument("scan", help="scan JSON from stage 1")
    sel.add_argument("--out", default=None, help="selection JSON (default: <video>.select.json)")
    _add_select_args(sel)
    sel.set_defaults(func=cmd_select)

    sh = sub.add_parser("sheet", help="stage 4: contact sheet + manifest from a folder of stills")
    sh.add_argument("stills_dir", help="folder of restored stills")
    sh.add_argument("--selection", default=None,
                    help="selection JSON from stage 2; without it the sheet has no "
                         "timestamps, scenes or source-frame provenance")
    sh.add_argument("--out", default=None, help="contact sheet JPEG (default: <dir>_sheet.jpg)")
    sh.add_argument("--manifest", default=None, help="manifest JSON (default: <dir>_manifest.json)")
    sh.add_argument("--columns", type=int, default=5, help="grid columns (default 5)")
    sh.add_argument("--thumb_width", type=int, default=420, help="cell width in px (default 420)")
    sh.add_argument("--quality", type=int, default=92, help="JPEG quality (default 92)")
    sh.add_argument("--title", default=None, help="sheet title (default: video name + count)")
    sh.set_defaults(func=cmd_sheet)

    r = sub.add_parser("restore", help="stage 3: restore one window, keep the centre frame")
    r.add_argument("png_dir", help="folder of PNGs forming one window")
    r.add_argument("--out", default=None, help="output PNG (seed suffix added when --seeds > 1)")
    r.add_argument("--window", type=int, default=0,
                   help=f"use the centred N frames (4n+1, min {cfg.MIN_WINDOW}). 0 = all")
    _add_restore_args(r)
    r.add_argument("--crop_pillarbox", action="store_true",
                   help="crop to the non-black content box before restoring")
    r.set_defaults(func=cmd_restore)

    run = sub.add_parser("run", help="all four stages: scan -> select -> restore -> sheet")
    run.add_argument("video", help="input video file, or (with --preview) a folder of them")
    run.add_argument("--out", default=None,
                     help="output directory (default: a folder beside the video, named after it)")
    run.add_argument("--force", action="store_true",
                     help="redo everything, ignoring existing scan, selection and stills")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--preview", nargs="?", type=preview_count,
                      const=DEFAULT_PREVIEW_COUNT, default=None, metavar="N",
                      help=f"quick preview: {DEFAULT_PREVIEW_COUNT} frames by default, or N, "
                           f"spread evenly over the video. Decodes only {PREVIEW_WINDOW_S:g}s at "
                           "each point and keeps the sharpest frame, so it never scans the whole "
                           "file; output is a contact sheet in <out>/preview/, and stills/ and "
                           "work/ are not touched. No SeedVR2 and no GPU. Passing any selection "
                           "or scan option (--interval, --seconds_per_still, --segment, ...) "
                           "switches to the full pipeline instead, still without restoring. "
                           "Required if `video` is a folder: every video directly inside it "
                           f"({', '.join(sorted(VIDEO_EXTENSIONS))}) is previewed in turn")
    mode.add_argument("--upscale-stills", dest="upscale_stills", action="store_true",
                      help="restore whatever is still in stills/, and nothing else. The other "
                           "half of --preview: preview, delete the ones you do not want, "
                           "then run this to restore only the survivors. Each is written as "
                           "<name>_upscaled.png beside the original, which is left in place")
    scan_group = run.add_argument_group("scan")
    scan_group.add_argument("--scene_threshold", type=float, default=DEFAULT_SCENE_THRESHOLD)
    scan_group.add_argument("--min_scene_len", type=int, default=DEFAULT_MIN_SCENE_LEN)
    scan_group.add_argument("--no_content_box", action="store_true")
    scan_group.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                            help=f"processes to scan with (default {DEFAULT_WORKERS})")
    _add_segment_arg(scan_group)
    _add_select_args(run)
    _add_restore_args(run)
    sheet_group = run.add_argument_group("sheet")
    sheet_group.add_argument("--columns", type=int, default=5)
    sheet_group.add_argument("--thumb_width", type=int, default=420)
    sheet_group.add_argument("--quality", type=int, default=92)
    run.set_defaults(func=cmd_run)
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    # `run` needs to know not just the values but whether the user chose them:
    # bare --preview takes the quick path, --preview with any selection or scan
    # option takes the full one. See _configured().
    args.configured_flags = _configured(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
