"""
Configuration: paths, model names and generation defaults.

Everything that varies by machine comes from the `.env` file in the project
root. Exactly one variable is required -- SEEDVR2_REPO -- because every other
path can be derived from it for a standard SeedVR2 layout. Each derived path can
still be overridden explicitly when your layout differs.

Precedence, highest first:
    1. real environment variables
    2. .env in the project root
    3. derived defaults

Real environment variables win so CI and one-off overrides work without editing
a tracked-adjacent file.
"""

import os
from pathlib import Path

# .../temporal_extractor/temporal_extractor/config.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def _load_env_file(path: Path) -> dict:
    """
    Minimal KEY=VALUE reader.

    Deliberately not python-dotenv: this is the tool's only configuration input
    and it is not worth a dependency, particularly on the side of the process
    that is meant to stay dependency-light.
    """
    values = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            values[key.strip()] = value
    return values


_ENV = _load_env_file(ENV_FILE)


def setting(name: str, default=None):
    """Resolve one setting through the precedence chain above."""
    return os.environ.get(name) or _ENV.get(name) or default


class ConfigError(RuntimeError):
    """Something required is missing or points nowhere."""


def _require_repo() -> Path:
    value = setting("SEEDVR2_REPO")
    if not value:
        raise ConfigError(
            f"SEEDVR2_REPO is not set.\n"
            f"Edit {ENV_FILE} and point it at your SeedVR2 checkout, e.g.\n"
            f"    SEEDVR2_REPO=D:\\models\\ComfyUI-SeedVR2_VideoUpscaler\n"
            f"Run install.bat if that file does not exist yet."
        )
    return Path(value)


SEEDVR2_REPO = _require_repo() if setting("SEEDVR2_REPO") else None

# Derived defaults assume the common layout, where the SeedVR2 checkout, its
# virtualenv and the model folder are siblings under one root:
#
#     <root>/refs/seedvr2      <- SEEDVR2_REPO
#     <root>/.venv-seedvr2     <- the interpreter that has torch
#     <root>/models/seedvr2    <- the checkpoints
#
# Set the variables explicitly in .env if yours is arranged differently.
_ROOT = SEEDVR2_REPO.parent.parent if SEEDVR2_REPO else PROJECT_ROOT

# The restorer's interpreter. Deliberately NOT this process's interpreter: the
# tool runs with no torch, the worker with a pinned torch/CUDA stack. Keeping
# them apart is the point of the subprocess split.
SEEDVR2_PYTHON = Path(setting("SEEDVR2_PYTHON",
                              _ROOT / ".venv-seedvr2" / "Scripts" / "python.exe"))
MODEL_DIR = Path(setting("MODEL_DIR", _ROOT / "models" / "seedvr2"))

DIT_MODEL = setting("DIT_MODEL", "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors")
VAE_MODEL = setting("VAE_MODEL", "ema_vae_fp16.safetensors")


def check() -> list[str]:
    """Return a list of human-readable configuration problems; empty means fine."""
    problems = []
    if SEEDVR2_REPO is None:
        problems.append("SEEDVR2_REPO is not set (see .env)")
    elif not (SEEDVR2_REPO / "src").is_dir():
        problems.append(f"SEEDVR2_REPO does not look like a SeedVR2 checkout: {SEEDVR2_REPO} "
                        "(expected a 'src' directory inside)")
    if not SEEDVR2_PYTHON.exists():
        problems.append(f"SEEDVR2_PYTHON not found: {SEEDVR2_PYTHON} "
                        "(the interpreter that has torch installed)")
    if not MODEL_DIR.is_dir():
        problems.append(f"MODEL_DIR not found: {MODEL_DIR}")
    else:
        for name in (DIT_MODEL, VAE_MODEL):
            if not (MODEL_DIR / name).exists():
                problems.append(f"model file missing: {MODEL_DIR / name}")
    return problems


# Minimum window the restorer will accept. SeedVR2 is a multi-frame model; below
# 5 frames there is no temporal information to exploit and the tool has no reason
# to exist. Windows must also satisfy 4n+1 (the VAE downsamples time by 4), so
# the legal sizes are 5, 9, 13, ...
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

# Memory/perf knobs fixed at worker start: they change how the models are built,
# so they cannot vary per call without a reload.
WORKER_DEFAULTS = {
    "attention_mode": "sdpa",
    "encode_tiled": False,
    "decode_tiled": True,
    "tile": 1024,
    "tile_overlap": 128,
    "blocks_to_swap": 0,
}
