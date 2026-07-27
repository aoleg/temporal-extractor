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


def _load_window(png_dir: Path, window: int):
    paths = sorted(p for p in png_dir.iterdir()
                   if p.suffix.lower() == ".png" and not p.name.startswith("restored_"))
    if not paths:
        raise SystemExit(f"no PNGs in {png_dir}")
    if window and window < len(paths):
        k = (len(paths) - window) // 2
        paths = paths[k:k + window]
    return paths, [read_png_rgb(p) for p in paths]


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
