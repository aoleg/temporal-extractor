"""
Configuration: paths, model names and generation defaults.

Everything machine-specific comes from the `.env` file in the project root.

Three paths are REQUIRED and none of them is guessed. The SeedVR2 checkout, the
interpreter that has torch, and the folder holding the checkpoints are three
independent locations: nothing about installing SeedVR2 implies they sit near
each other, so deriving any of them from another would only produce confident
wrong answers.

Checkpoint filenames are optional and default to the names the SeedVR2 release
commonly ships. They are only a convenience -- quantisation variants (fp16, fp8,
int8, nvfp4, whatever a later release adds) all carry different names, and files
get renamed in practice, so set them explicitly whenever yours differ.

Precedence, highest first:
    1. real environment variables
    2. .env in the project root
    3. the checkpoint-name defaults below (paths have no defaults)

Real environment variables win so CI and one-off overrides work without editing
a file.
"""

import os
from pathlib import Path

# .../temporal_extractor/temporal_extractor/config.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Required path settings, with the explanation shown when one is missing.
REQUIRED_PATHS = {
    "SEEDVR2_REPO": "your SeedVR2 checkout (the folder containing src/ and configs_7b/)",
    "SEEDVR2_PYTHON": "the python.exe of the virtualenv that has torch installed",
    "MODEL_DIR": "the folder holding the DiT checkpoint and the VAE",
}

# Names from the common SeedVR2 release. Convenience only -- see module docstring.
DEFAULT_DIT_MODEL = "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
DEFAULT_VAE_MODEL = "ema_vae_fp16.safetensors"


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


def _path(name: str):
    """A required path setting: Path if set, None if not. Never guessed."""
    value = setting(name)
    return Path(value) if value else None


SEEDVR2_REPO = _path("SEEDVR2_REPO")
# NOT this process's interpreter: the tool runs with no torch, the worker with a
# pinned torch/CUDA stack. Keeping them apart is the point of the split.
SEEDVR2_PYTHON = _path("SEEDVR2_PYTHON")
MODEL_DIR = _path("MODEL_DIR")

DIT_MODEL = setting("DIT_MODEL", DEFAULT_DIT_MODEL)
VAE_MODEL = setting("VAE_MODEL", DEFAULT_VAE_MODEL)


def _describe_available(model_dir: Path) -> str:
    """List what is actually in the model folder, to make a name mismatch obvious."""
    try:
        found = sorted(p.name for p in model_dir.iterdir()
                       if p.suffix.lower() in {".safetensors", ".gguf", ".pth"})
    except OSError:
        return ""
    if not found:
        return "        (no .safetensors/.gguf/.pth files found there)"
    return "        available: " + "\n                   ".join(found)


def check() -> list[str]:
    """Return human-readable configuration problems; an empty list means fine."""
    problems = []

    for name, description in REQUIRED_PATHS.items():
        if globals()[name] is None:
            problems.append(f"{name} is not set -- {description}")

    if SEEDVR2_REPO is not None and not (SEEDVR2_REPO / "src").is_dir():
        problems.append(f"SEEDVR2_REPO does not look like a SeedVR2 checkout: {SEEDVR2_REPO} "
                        "(expected a 'src' directory inside)")
    if SEEDVR2_PYTHON is not None and not SEEDVR2_PYTHON.exists():
        problems.append(f"SEEDVR2_PYTHON not found: {SEEDVR2_PYTHON} "
                        "(the interpreter that has torch installed)")

    if MODEL_DIR is not None:
        if not MODEL_DIR.is_dir():
            problems.append(f"MODEL_DIR not found: {MODEL_DIR}")
        else:
            for variable, name in (("DIT_MODEL", DIT_MODEL), ("VAE_MODEL", VAE_MODEL)):
                if not (MODEL_DIR / name).exists():
                    problems.append(
                        f"{variable} '{name}' is not in {MODEL_DIR}.\n"
                        f"        Set {variable} in .env to the filename you actually have.\n"
                        + _describe_available(MODEL_DIR))
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
