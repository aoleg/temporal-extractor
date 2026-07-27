"""
Paths and defaults. Everything here can be overridden per-call or by env var so
the tool is not welded to one machine.
"""

import os
from pathlib import Path

# Repo root: .../temporal_extractor/temporal_extractor/config.py -> .../vidstills
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDSTILLS_ROOT = PROJECT_ROOT.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


# The restorer's interpreter. Deliberately NOT this process's interpreter: the
# tool runs under .venv with no torch, the worker under .venv-seedvr2 with a
# pinned torch/CUDA stack. Keeping them apart is the whole point of the split.
SEEDVR2_PYTHON = _env_path(
    "VIDSTILLS_SEEDVR2_PYTHON", VIDSTILLS_ROOT / ".venv-seedvr2" / "Scripts" / "python.exe"
)
SEEDVR2_REPO = _env_path("VIDSTILLS_SEEDVR2_REPO", VIDSTILLS_ROOT / "refs" / "seedvr2")
MODEL_DIR = _env_path("VIDSTILLS_MODEL_DIR", VIDSTILLS_ROOT / "models" / "seedvr2")

DIT_MODEL = os.environ.get(
    "VIDSTILLS_DIT_MODEL", "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
)
VAE_MODEL = os.environ.get("VIDSTILLS_VAE_MODEL", "ema_vae_fp16.safetensors")

# Minimum window the restorer will accept. SeedVR2 is a multi-frame model; below
# 5 frames there is no temporal information to exploit and the tool has no
# reason to exist. Windows must additionally satisfy 4n+1 (VAE temporal
# downsample factor is 4), so the legal sizes are 5, 9, 13, ...
MIN_WINDOW = 5

# Generation defaults. cfg_scale defaults to 1.0: the checkpoint is one-step
# distilled, and measured against a flat background patch, cfg > 1.0 adds
# invented high-frequency speckle without recovering real detail. It stays
# exposed and unclamped for sweeps -- it just is not the default.
DEFAULTS = {
    "resolution": 1080,
    "seed": 42,
    "cfg_scale": 1.0,
    "input_noise_scale": 0.0,
    "latent_noise_scale": 0.0,
    "color_correction": "lab",
}

# Memory/perf knobs that are fixed at worker start (they change how the models
# are built, so they cannot vary per call without a reload).
WORKER_DEFAULTS = {
    "attention_mode": "sdpa",
    "encode_tiled": False,
    "decode_tiled": True,
    "tile": 1024,
    "tile_overlap": 128,
    "blocks_to_swap": 0,
}
