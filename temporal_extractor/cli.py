"""
Stage 3 on the command line: window in, restored centre frame(s) out.

Exists so the restore stage can be exercised and swept in isolation, which is
how the pipeline is meant to be debugged -- each stage runnable on its own.

    T:\\claude\\vidstills\\.venv\\Scripts\\python.exe -m temporal_extractor.cli restore <png_dir>
"""

import argparse
import os
import sys
import time
from pathlib import Path

from . import config as cfg
from .frames import content_box, crop, read_png_rgb, write_png_rgb
from .restore import SeedVR2Restorer
from .scan import (
    DEFAULT_MIN_SCENE_LEN,
    DEFAULT_SCENE_THRESHOLD,
    read_scan,
    scan_video,
    write_scan,
)
from .select import (
    DEFAULT_COUNT,
    DEFAULT_HASH_DISTANCE,
    DEFAULT_MIN_GAP_S,
    DEFAULT_WINDOW,
    hamming,
    select_frames,
    write_selection,
)


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

    def progress(done, total):
        pct = f" ({100 * done / total:.0f}%)" if total > 0 else ""
        print(f"\r  scanned {done} frames{pct}", end="", flush=True)

    meta = scan_video(
        video,
        scene_threshold=args.scene_threshold,
        min_scene_len=args.min_scene_len,
        detect_content_box=not args.no_content_box,
        progress=None if args.quiet else progress,
    )
    if not args.quiet:
        print()

    out = Path(args.out) if args.out else video.with_suffix(".scan.json")
    write_scan(meta, out)

    v, box, scenes = meta["video"], meta["content_box"], meta["scenes"]
    print(f"{v['filename']}: {v['width']}x{v['height']} @ {v['fps']:.3f}fps, "
          f"{v['frame_count']} frames, {v['duration_s']:.1f}s")
    if (box["w"], box["h"]) != (v["width"], v["height"]):
        print(f"content box: {box['w']}x{box['h']} at ({box['x']},{box['y']}) "
              f"-- {100 * (1 - box['w'] * box['h'] / (v['width'] * v['height'])):.0f}% of the frame is matte")
    else:
        print("content box: full frame (no pillar/letterboxing)")
    print(f"scenes: {len(scenes)} (threshold {meta['scan']['scene_threshold']}, "
          f"min length {meta['scan']['min_scene_len']} frames)")

    for scene in scenes[:args.show_scenes]:
        print(f"  scene {scene['id']:>3}  frames {scene['start']:>6}-{scene['end']:<6} "
              f"{scene['start_t']:>8.2f}s-{scene['end_t']:<8.2f}s  "
              f"n={scene['frame_count']:<5} best={scene['best_frame']:<6} "
              f"sharp={scene['best_sharpness']:.1f} (mean {scene['mean_sharpness']:.1f})")
    if len(scenes) > args.show_scenes:
        print(f"  ... {len(scenes) - args.show_scenes} more")

    print(f"scanned in {meta['scan']['elapsed_s']:.1f}s -> {out}")
    return 0


def cmd_select(args) -> int:
    meta = read_scan(args.scan)
    sel = select_frames(
        meta,
        count=args.count,
        window=args.window,
        hash_distance=args.hash_distance,
        min_gap_s=args.min_gap,
        max_per_scene=args.max_per_scene,
    )

    out = Path(args.out) if args.out else Path(args.scan).with_suffix("").with_suffix(".select.json")
    write_selection(sel, out)

    s = sel["select"]
    print(f"{sel['video']['filename']}: {s['selected']}/{s['requested']} picks "
          f"from {s['eligible_frames']} eligible frames "
          f"(window {s['window']}, min dHash distance {s['hash_distance']}, "
          f"min gap {s['min_gap_s']}s = {s['min_gap_frames']} frames)")
    if s["filled_globally"]:
        print(f"{s['filled_globally']} pick(s) came from the maximin fill "
              "-- dedupe starved that many segments")
    if s["scenes_too_short"]:
        print(f"scenes too short to host a {s['window']}-frame window: {s['scenes_too_short']}")
    print(f"scene quota: {s['scene_quota']}")
    print()
    print(f"  {'frame':>7} {'time':>9} {'scene':>6} {'sharpness':>10}  window")
    for p in sel["picks"]:
        print(f"  {p['frame']:>7} {p['t']:>8.2f}s {p['scene']:>6} {p['sharpness']:>10.1f}  "
              f"[{p['window'][0]}..{p['window'][1]}]")

    # The closest pair is the honest measure of how varied the set is; if it sits
    # at the threshold, the limit is binding and worth raising or lowering.
    if len(sel["picks"]) > 1:
        pairs = [(hamming(a["dhash"], b["dhash"]), a["frame"], b["frame"])
                 for i, a in enumerate(sel["picks"]) for b in sel["picks"][i + 1:]]
        d, fa, fb = min(pairs)
        print(f"\nclosest pair: f{fa} and f{fb} at dHash distance {d} "
              f"(floor {s['hash_distance']})")
    if s["selected"] < s["requested"]:
        print(f"\nshort by {s['requested'] - s['selected']}: nothing else cleared both filters. "
              f"Lower --hash_distance (now {s['hash_distance']}) to allow more similar picks, "
              f"or --min_gap (now {s['min_gap_s']}s) to allow closer ones. "
              "If neither helps, the footage genuinely has fewer distinct moments than requested.")
    print(f"\n-> {out}")
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="temporal_extractor",
                                 description="Extract high-quality stills from low-quality video.")
    sub = ap.add_subparsers(dest="command", required=True)

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
    s.add_argument("--show_scenes", type=int, default=20, help="how many scenes to print")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_scan)

    sel = sub.add_parser("select", help="stage 2: pick the best N frames from a scan")
    sel.add_argument("scan", help="scan JSON from stage 1")
    sel.add_argument("--out", default=None, help="selection JSON (default: <video>.select.json)")
    sel.add_argument("--count", type=int, default=DEFAULT_COUNT, help=f"stills to aim for (default {DEFAULT_COUNT})")
    sel.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                     help=f"restore window size, 4n+1 (default {DEFAULT_WINDOW}). Picks are kept "
                          "far enough from a cut that the window stays inside one scene")
    sel.add_argument("--hash_distance", type=int, default=DEFAULT_HASH_DISTANCE,
                     help=f"minimum dHash Hamming distance between picks (default {DEFAULT_HASH_DISTANCE}). "
                          "Higher = more varied but fewer picks")
    sel.add_argument("--min_gap", type=float, default=DEFAULT_MIN_GAP_S, metavar="SECONDS",
                     help=f"minimum time between any two picks (default {DEFAULT_MIN_GAP_S}s). "
                          "Backstop for two adjacent segments choosing frames either side of "
                          "their shared boundary; 0 disables")
    sel.add_argument("--max_per_scene", type=int, default=None,
                     help="cap picks from any one scene, so a long take cannot dominate")
    sel.set_defaults(func=cmd_select)

    r = sub.add_parser("restore", help="stage 3: restore a window, keep the centre frame")
    r.add_argument("png_dir", help="folder of PNGs forming one window")
    r.add_argument("--out", default=None, help="output PNG (seed suffix added when --seeds > 1)")
    r.add_argument("--window", type=int, default=0,
                   help=f"use the centred N frames (4n+1, min {cfg.MIN_WINDOW}). 0 = all")

    g = r.add_argument_group("generation parameters (sweepable)")
    d = cfg.DEFAULTS
    g.add_argument("--resolution", type=int, default=d["resolution"], help="target SHORT side")
    g.add_argument("--seed", type=int, default=d["seed"], help="base seed")
    g.add_argument("--seeds", type=int, default=1,
                   help="run the window N times with consecutive seeds, writing every variant")
    g.add_argument("--cfg_scale", type=float, default=d["cfg_scale"],
                   help="classifier-free guidance; default 1.0 (off). Not clamped: only the "
                        "centre frame is kept, so still-image values apply. >1.0 runs both "
                        "DiT branches and roughly doubles phase 2")
    g.add_argument("--input_noise_scale", type=float, default=d["input_noise_scale"])
    g.add_argument("--latent_noise_scale", type=float, default=d["latent_noise_scale"])
    g.add_argument("--color_correction", default=d["color_correction"],
                   choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"])

    p = r.add_argument_group("performance / memory")
    p.add_argument("--attention_mode", default=cfg.WORKER_DEFAULTS["attention_mode"],
                   choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"])
    p.add_argument("--vae_encode_tiled", action="store_true",
                   help="tile VAE encoding; needed above ~1080p or with long windows")
    p.add_argument("--no_decode_tiling", action="store_true")
    p.add_argument("--blocks_to_swap", type=int, default=0, help="offload N DiT blocks to CPU (0-36)")
    p.add_argument("--crop_pillarbox", action="store_true",
                   help="crop to the non-black content box before restoring")
    # Not fully silent: the reference repo emits some lines with force=True that
    # bypass its own debug flag, so this reduces the noise rather than killing it.
    p.add_argument("--quiet", action="store_true", help="reduce the worker's progress logging")
    r.set_defaults(func=cmd_restore)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
