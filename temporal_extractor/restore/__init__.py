"""
Stage 3: restore a window of frames, keep the centre.

Only the client is exported. worker.py deliberately is NOT imported here -- it
runs under a different interpreter and imports torch, which must never happen
in the tool's process.
"""

from .client import RestoreError, SeedVR2Restorer

__all__ = ["SeedVR2Restorer", "RestoreError"]
