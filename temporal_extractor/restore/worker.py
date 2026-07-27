"""
Persistent SeedVR2 restore worker.

Runs under .venv-seedvr2 and is the ONLY file in this project that imports
torch. It is launched by client.SeedVR2Restorer, never by hand (though it can
be, for debugging -- see __main__ below).

It loads the models once and serves restore requests until told to shut down,
which is what makes a 15-still run or a seed sweep affordable: model
materialisation dominates a single window.

Frames arrive as raw uint8 RGB bytes over a socket; the restored centre frame
goes back the same way.
"""

import argparse
import os
import socket
import sys
import time
import traceback
from pathlib import Path

# Both the project (for protocol/config) and the SeedVR2 reference repo need to
# be importable. The reference repo is not a package and resolves its own
# imports relative to its root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from temporal_extractor import config as cfg
from temporal_extractor.protocol import (
    PROTOCOL_VERSION,
    array_header,
    recv_message,
    send_message,
)

if str(cfg.SEEDVR2_REPO) not in sys.path:
    sys.path.insert(0, str(cfg.SEEDVR2_REPO))

# Must be set BEFORE torch is imported. Without it the caching allocator
# fragments during VAE decode and spills to system RAM -- measured at ~8GB
# paged, turning a 3s decode into 100s.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import numpy as np
import torch

from src.core.generation_utils import (  # noqa: E402  (must follow sys.path setup)
    compute_generation_info,
    load_text_embeddings,
    log_generation_start,
    prepare_runner,
    script_directory,
    setup_generation_context,
)
from src.core.generation_phases import (  # noqa: E402
    decode_all_batches,
    encode_all_batches,
    postprocess_all_batches,
    upscale_all_batches,
)
from src.utils.debug import Debug  # noqa: E402


class SeedVR2Engine:
    """
    Owns the loaded runner. Models stay in the reference repo's global cache
    between calls, so only the first window pays materialisation.
    """

    def __init__(self, *, model_dir, dit_model, vae_model, attention_mode,
                 encode_tiled, decode_tiled, tile, tile_overlap, blocks_to_swap,
                 debug_enabled):
        self.debug = Debug(enabled=debug_enabled)
        self.model_dir = str(model_dir)
        self.dit_model = dit_model
        self.vae_model = vae_model
        self.attention_mode = attention_mode
        self.encode_tiled = encode_tiled
        self.decode_tiled = decode_tiled
        self.tile = tile
        self.tile_overlap = tile_overlap
        self.blocks_to_swap = blocks_to_swap
        self._cfg_scale = 1.0  # read by the inference wrapper at call time

    def _build(self):
        ctx = setup_generation_context(
            dit_device="cuda:0",
            vae_device="cuda:0",
            # Caching needs an offload target; models park in RAM between calls.
            dit_offload_device="cpu",
            vae_offload_device="cpu",
            tensor_offload_device="cpu",
            debug=self.debug,
        )
        runner, cache_context = prepare_runner(
            dit_model=self.dit_model,
            vae_model=self.vae_model,
            model_dir=self.model_dir,
            debug=self.debug,
            ctx=ctx,
            dit_cache=True,
            vae_cache=True,
            dit_id="temporal_extractor_dit",
            vae_id="temporal_extractor_vae",
            block_swap_config={
                "blocks_to_swap": self.blocks_to_swap,
                "swap_io_components": False,
                "offload_device": "cpu",
            },
            encode_tiled=self.encode_tiled,
            encode_tile_size=(self.tile, self.tile),
            encode_tile_overlap=(self.tile_overlap, self.tile_overlap),
            decode_tiled=self.decode_tiled,
            decode_tile_size=(self.tile, self.tile),
            decode_tile_overlap=(self.tile_overlap, self.tile_overlap),
            attention_mode=self.attention_mode,
        )
        ctx["cache_context"] = cache_context
        ctx["text_embeds"] = load_text_embeddings(
            script_directory, ctx["dit_device"], ctx["compute_dtype"], self.debug
        )

        # upscale_all_batches() sets config.diffusion.cfg.scale = 1.0 on entry
        # and calls runner.inference() without cfg_scale, so the config value is
        # unreachable -- setting it before the call is silently discarded.
        # Injecting the argument here is the only route to the actual model.
        if not getattr(runner, "_cfg_patched", False):
            original_inference = runner.inference

            def inference_with_cfg(*args, **kwargs):
                kwargs.setdefault("cfg_scale", self._cfg_scale)
                return original_inference(*args, **kwargs)

            runner.inference = inference_with_cfg
            runner._cfg_patched = True
        return runner, ctx

    def restore(self, frames, *, resolution, seed, cfg_scale, input_noise_scale,
                latent_noise_scale, color_correction):
        """
        frames:  uint8 RGB array [T, H, W, 3]; T must be 4n+1 and >= MIN_WINDOW
        returns: uint8 RGB array [H', W', 3] -- the restored centre frame
        """
        n = int(frames.shape[0])
        if n < cfg.MIN_WINDOW:
            raise ValueError(
                f"window of {n} is below the {cfg.MIN_WINDOW}-frame minimum; "
                "SeedVR2 needs neighbouring frames to have any temporal information to use"
            )
        if n % 4 != 1:
            raise ValueError(f"window of {n} violates the 4n+1 VAE constraint (use 5, 9, 13...)")

        self._cfg_scale = cfg_scale
        runner, ctx = self._build()

        # The pipeline wants [T, H, W, C] float16 in [0,1], RGB, UNnormalised --
        # it applies its own resize/pad/normalise internally.
        images = torch.from_numpy(frames.astype(np.float32) / 255.0).to(torch.float16)

        # batch_size == n keeps the whole window in one batch, which is the
        # entire point: the model gets to use the neighbouring frames.
        images, info = compute_generation_info(
            ctx=ctx, images=images, resolution=resolution, max_resolution=0,
            batch_size=n, uniform_batch_size=False, seed=seed,
            prepend_frames=0, temporal_overlap=0, debug=self.debug,
        )
        log_generation_start(info, self.debug)

        ctx = encode_all_batches(
            runner, ctx=ctx, images=images, debug=self.debug, batch_size=n,
            uniform_batch_size=False, seed=seed, progress_callback=None,
            temporal_overlap=0, resolution=resolution, max_resolution=0,
            input_noise_scale=input_noise_scale, color_correction=color_correction,
        )
        ctx = upscale_all_batches(
            runner, ctx=ctx, debug=self.debug, progress_callback=None, seed=seed,
            latent_noise_scale=latent_noise_scale, cache_model=True,
        )
        ctx = decode_all_batches(
            runner, ctx=ctx, debug=self.debug, progress_callback=None, cache_model=True,
        )
        ctx = postprocess_all_batches(
            ctx=ctx, debug=self.debug, progress_callback=None,
            color_correction=color_correction, prepend_frames=0,
            temporal_overlap=0, batch_size=n,
        )

        out = ctx["final_video"]
        if out.is_cuda:
            out = out.cpu()
        if out.dtype != torch.float32:
            out = out.to(torch.float32)
        centre = out[n // 2].numpy()
        return (np.clip(centre, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def serve(port: int, engine_kwargs: dict) -> int:
    sock = socket.create_connection(("127.0.0.1", port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    engine = SeedVR2Engine(**engine_kwargs)
    send_message(sock, {
        "op": "ready",
        "protocol": PROTOCOL_VERSION,
        "pid": os.getpid(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    })

    while True:
        try:
            header, payload = recv_message(sock)
        except Exception:
            # Parent vanished. Nothing useful left to do.
            return 1

        op = header.get("op")
        if op == "shutdown":
            send_message(sock, {"op": "bye"})
            sock.close()
            return 0

        if op == "ping":
            send_message(sock, {"op": "pong"})
            continue

        if op != "restore":
            send_message(sock, {"op": "error", "message": f"unknown op {op!r}"})
            continue

        try:
            frames = np.frombuffer(payload, dtype=np.dtype(header["dtype"]))
            frames = frames.reshape(header["shape"])
            started = time.time()
            centre = engine.restore(frames, **header["params"])
            elapsed = time.time() - started
            reply = {"op": "ok", "elapsed": elapsed}
            reply.update(array_header(centre))
            send_message(sock, reply, np.ascontiguousarray(centre).tobytes())
        except Exception as exc:
            # A bad request must not kill a worker that took 10s to warm up.
            send_message(sock, {
                "op": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })


def main() -> int:
    ap = argparse.ArgumentParser(description="SeedVR2 persistent restore worker (internal).")
    ap.add_argument("--port", type=int, required=True, help="localhost port opened by the parent")
    ap.add_argument("--model-dir", default=str(cfg.MODEL_DIR))
    ap.add_argument("--dit-model", default=cfg.DIT_MODEL)
    ap.add_argument("--vae-model", default=cfg.VAE_MODEL)
    ap.add_argument("--attention-mode", default=cfg.WORKER_DEFAULTS["attention_mode"])
    ap.add_argument("--encode-tiled", action="store_true")
    ap.add_argument("--no-decode-tiling", action="store_true")
    ap.add_argument("--tile", type=int, default=cfg.WORKER_DEFAULTS["tile"])
    ap.add_argument("--tile-overlap", type=int, default=cfg.WORKER_DEFAULTS["tile_overlap"])
    ap.add_argument("--blocks-to-swap", type=int, default=cfg.WORKER_DEFAULTS["blocks_to_swap"])
    ap.add_argument("--quiet", action="store_true", help="reduce SeedVR2 progress logging")
    args = ap.parse_args()

    return serve(args.port, {
        "model_dir": args.model_dir,
        "dit_model": args.dit_model,
        "vae_model": args.vae_model,
        "attention_mode": args.attention_mode,
        "encode_tiled": args.encode_tiled,
        "decode_tiled": not args.no_decode_tiling,
        "tile": args.tile,
        "tile_overlap": args.tile_overlap,
        "blocks_to_swap": args.blocks_to_swap,
        "debug_enabled": not args.quiet,
    })


if __name__ == "__main__":
    sys.exit(main())
