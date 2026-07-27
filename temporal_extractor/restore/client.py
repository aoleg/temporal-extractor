"""
Tool-side handle on the SeedVR2 worker.

This is the narrow interface the rest of the tool sees:

    with SeedVR2Restorer() as restorer:
        centre = restorer.restore(frames)        # list[np.ndarray] -> np.ndarray

numpy and stdlib only. Nothing in this module (or anything it imports) may pull
in torch -- the restorer's dependency pins must not dictate what the tool can
use, which is why the model lives behind a subprocess in a separate venv.
"""

import socket
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from .. import config as cfg
from ..protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    array_header,
    recv_message,
    send_message,
)

_WORKER = Path(__file__).resolve().parent / "worker.py"


class RestoreError(RuntimeError):
    """The worker rejected a request or died trying to serve it."""


class SeedVR2Restorer:
    """
    Starts one persistent worker and keeps it warm.

    Model materialisation dominates the cost of a single window (~10s of a ~13s
    1080p call), so the worker is started once and reused for every window in a
    run. Starting one per window would roughly double a 15-still job.
    """

    def __init__(self, *, python=None, model_dir=None, dit_model=None, vae_model=None,
                 attention_mode=None, encode_tiled=None, decode_tiled=None,
                 tile=None, tile_overlap=None, blocks_to_swap=None,
                 startup_timeout=180.0, quiet=False, log=None):
        # Left unresolved until start(): these are required settings with no
        # defaults, so they may legitimately be None here, and start() reports
        # every configuration problem at once rather than dying on the first.
        self.python = Path(python) if python else cfg.SEEDVR2_PYTHON
        self.model_dir = Path(model_dir) if model_dir else cfg.MODEL_DIR
        self.dit_model = dit_model or cfg.DIT_MODEL
        self.vae_model = vae_model or cfg.VAE_MODEL
        w = cfg.WORKER_DEFAULTS
        self.attention_mode = attention_mode or w["attention_mode"]
        self.encode_tiled = w["encode_tiled"] if encode_tiled is None else encode_tiled
        self.decode_tiled = w["decode_tiled"] if decode_tiled is None else decode_tiled
        self.tile = tile or w["tile"]
        self.tile_overlap = tile_overlap or w["tile_overlap"]
        self.blocks_to_swap = w["blocks_to_swap"] if blocks_to_swap is None else blocks_to_swap
        self.startup_timeout = startup_timeout
        self.quiet = quiet
        self._log = log if log is not None else (lambda line: print(line, file=sys.stderr))

        self._proc = None
        self._sock = None
        self._listener = None
        self._pump = None
        self.info = {}

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        if self._proc is not None:
            return self

        problems = cfg.check()
        if problems:
            raise RestoreError(
                "the restorer is not configured:\n  - "
                + "\n  - ".join(problems)
                + f"\nEdit {cfg.ENV_FILE}, then run: extract.bat doctor"
            )

        # Bind an ephemeral port and let the worker dial back, so stdout stays
        # free for the reference repo's very chatty logging.
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self._listener.settimeout(self.startup_timeout)
        port = self._listener.getsockname()[1]

        cmd = [
            str(self.python), str(_WORKER),
            "--port", str(port),
            "--model-dir", str(self.model_dir),
            "--dit-model", self.dit_model,
            "--vae-model", self.vae_model,
            "--attention-mode", self.attention_mode,
            "--tile", str(self.tile),
            "--tile-overlap", str(self.tile_overlap),
            "--blocks-to-swap", str(self.blocks_to_swap),
        ]
        if self.encode_tiled:
            cmd.append("--encode-tiled")
        if not self.decode_tiled:
            cmd.append("--no-decode-tiling")
        if self.quiet:
            cmd.append("--quiet")

        env = {
            # The reference repo prints emoji at import; under a cp1252 console
            # that is a UnicodeEncodeError before any work happens.
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
        import os
        merged = dict(os.environ)
        merged.update(env)

        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=merged,
        )
        self._pump = threading.Thread(target=self._drain_output, daemon=True)
        self._pump.start()

        try:
            self._sock, _ = self._listener.accept()
        except socket.timeout as exc:
            self._kill()
            raise RestoreError(
                f"worker did not connect within {self.startup_timeout}s -- see worker log above"
            ) from exc
        finally:
            self._listener.close()
            self._listener = None
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        header, _ = recv_message(self._sock)
        if header.get("op") != "ready":
            self._kill()
            raise RestoreError(f"unexpected handshake from worker: {header}")
        if header.get("protocol") != PROTOCOL_VERSION:
            self._kill()
            raise RestoreError(
                f"protocol mismatch: worker speaks {header.get('protocol')}, "
                f"tool speaks {PROTOCOL_VERSION}"
            )
        self.info = header
        return self

    def _drain_output(self):
        """Relay worker stdout so its logs and tracebacks are not swallowed."""
        stream = self._proc.stdout
        if stream is None:
            return
        for line in stream:
            self._log(line.rstrip())

    def _kill(self):
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(timeout=10)
        self._proc = None

    def close(self):
        if self._sock is not None:
            try:
                send_message(self._sock, {"op": "shutdown"})
                recv_message(self._sock)
            except Exception:
                pass  # worker already gone; the kill below is the backstop
            finally:
                self._sock.close()
                self._sock = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.close()
        return False

    # -- the interface -----------------------------------------------------

    def restore(self, frames, **params) -> np.ndarray:
        """
        Restore a window of frames and return the centre one.

        frames: list of HxWx3 uint8 RGB arrays (or one [T,H,W,3] array).
                len must be 4n+1 and at least MIN_WINDOW.
        params: any of resolution, seed, cfg_scale, input_noise_scale,
                latent_noise_scale, color_correction. Unset keys take
                config.DEFAULTS -- notably cfg_scale defaults to 1.0.

        returns: HxWx3 uint8 RGB centre frame.
        """
        if self._proc is None:
            self.start()

        unknown = set(params) - set(cfg.DEFAULTS)
        if unknown:
            raise TypeError(f"unknown restore parameter(s): {sorted(unknown)}")
        call = dict(cfg.DEFAULTS)
        call.update(params)

        stack = frames if isinstance(frames, np.ndarray) else np.stack(frames)
        if stack.dtype != np.uint8:
            raise TypeError(f"frames must be uint8 RGB, got {stack.dtype}")
        if stack.ndim != 4 or stack.shape[-1] != 3:
            raise TypeError(f"frames must be [T, H, W, 3], got {stack.shape}")
        stack = np.ascontiguousarray(stack)

        request = {"op": "restore", "params": call}
        request.update(array_header(stack))
        try:
            send_message(self._sock, request, stack.tobytes())
            header, payload = recv_message(self._sock)
        except (OSError, ProtocolError) as exc:
            code = self._proc.poll() if self._proc else None
            raise RestoreError(
                f"lost the worker while restoring ({exc}); exit code {code}. "
                "A hard CUDA OOM shows up here -- try a lower resolution, "
                "encode_tiled=True, or a shorter window."
            ) from exc

        if header.get("op") == "error":
            raise RestoreError(header.get("traceback") or header.get("message", "unknown error"))
        if header.get("op") != "ok":
            raise RestoreError(f"unexpected reply: {header}")

        self.last_elapsed = header.get("elapsed")
        out = np.frombuffer(payload, dtype=np.dtype(header["dtype"]))
        return out.reshape(header["shape"]).copy()  # frombuffer is read-only
