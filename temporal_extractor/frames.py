"""
Frame helpers shared by the pipeline stages. numpy/OpenCV only -- tool side.
"""

from pathlib import Path

import cv2
import numpy as np


def read_png_rgb(path) -> np.ndarray:
    """Read a PNG as HxWx3 uint8 RGB (OpenCV hands back BGR)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_png_rgb(path, rgb: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def content_box(frames, thresh: int = 16):
    """
    Find the non-black content box shared by every frame in a window.

    Returns (x, y, w, h).

    Sources here are routinely pillarboxed -- e.g. 638x480 of picture inside an
    848x480 frame. The bars matter for three separate reasons: they waste ~25%
    of restore compute, they dilute any sharpness score computed over the whole
    frame, and they would be baked into stills destined for LoRA training. At
    1440p, cropping them was the difference between running and a hard OOM.
    """
    stack = frames if isinstance(frames, np.ndarray) else np.stack(frames)
    acc = stack.max(axis=0).max(axis=2)          # brightest value each pixel reaches
    cols = np.where(acc.max(axis=0) > thresh)[0]
    rows = np.where(acc.max(axis=1) > thresh)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, stack.shape[2], stack.shape[1]
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    return x0, y0, x1 - x0, y1 - y0


def crop(frames, box):
    x, y, w, h = box
    return [f[y:y + h, x:x + w] for f in frames]


def sharpness(rgb: np.ndarray) -> float:
    """
    Variance of the Laplacian.

    Only meaningful for ranking frames WITHIN one video at one resolution: it is
    not scale-invariant (the same film scored 15.0 at 480p and 5.4 at 1080p), and
    invented high-frequency noise inflates it, so it will happily rank a noisy
    image above a clean one. Never threshold on an absolute value.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
