"""
Terminal progress reporting.

Long stages must never look stuck. A full-length feature takes minutes to scan
even after the speedups, so the bar reports percent, frames, throughput and a
time remaining rather than a bare counter.

Two things it deliberately handles:

- The declared frame count of a container is a hint and is regularly wrong. If
  we pass it, the bar stops pretending it knows the total and degrades to a
  spinner rather than showing 104%.
- Not every destination is a terminal. When output is redirected, \\r rewriting
  turns a log into noise, so it switches to occasional newline updates instead.

ASCII only: the reference SeedVR2 code already taught us that emitting non-ASCII
to a cp1252 console raises UnicodeEncodeError before any work happens.
"""

import sys
import time

SPINNER = "|/-\\"


def _hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class ProgressBar:
    """
    Usage:

        with ProgressBar(total, "scanning") as bar:
            ...
            bar.advance()          # or bar.set_done(n)

    Rendering is rate-limited by time, not by a frame count, so the cost is
    constant whether a stage runs at 50 or 5000 frames a second.
    """

    def __init__(self, total=None, label="", stream=None, interval=0.15,
                 width=28, enabled=True):
        self.total = total if (total and total > 0) else None
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.width = width
        self.enabled = enabled
        self.isatty = bool(getattr(self.stream, "isatty", lambda: False)())
        # Redirected output gets sparse newline updates; a rewriting bar there
        # would just fill the log with control characters.
        self.interval = interval if self.isatty else max(interval, 5.0)
        self.start = time.monotonic()
        self.done = 0
        self._last_render = 0.0
        self._spin = 0
        self._width_written = 0

    # -- state -------------------------------------------------------------

    def set_done(self, n: int) -> None:
        self.done = n
        if self.total is not None and n > self.total:
            # The container lied about the length. Stop quoting a percentage.
            self.total = None
        self._maybe_render()

    def advance(self, n: int = 1) -> None:
        self.set_done(self.done + n)

    # -- rendering ---------------------------------------------------------

    def _line(self) -> str:
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        head = f"{self.label}  " if self.label else ""

        if self.total:
            frac = min(1.0, self.done / self.total)
            filled = int(self.width * frac)
            bar = "#" * filled + "-" * (self.width - filled)
            # An estimate from the first fraction of a second is nonsense --
            # "eta 1:36:33" on a 20s job -- so withhold it until the rate means
            # something.
            settled = elapsed >= 1.0 and self.done >= 32 and rate > 0
            eta = _hms((self.total - self.done) / rate) if settled else "--:--"
            return (f"{head}[{bar}] {frac*100:3.0f}%  "
                    f"{self.done:,}/{self.total:,}  {rate:,.0f} fps  eta {eta}")

        self._spin = (self._spin + 1) % len(SPINNER)
        return (f"{head}{SPINNER[self._spin]}  {self.done:,} frames  "
                f"{rate:,.0f} fps  {_hms(elapsed)} elapsed")

    def _write(self, text: str, end: str = "") -> None:
        try:
            self.stream.write(text + end)
            self.stream.flush()
        except (ValueError, OSError):
            self.enabled = False  # stream went away; stop trying

    def _maybe_render(self, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_render < self.interval:
            return
        self._last_render = now
        line = self._line()
        if self.isatty:
            # Pad to erase whatever the previous, possibly longer, line left.
            pad = max(0, self._width_written - len(line))
            self._width_written = len(line)
            self._write("\r" + line + " " * pad)
        else:
            self._write(line, "\n")

    def close(self, summary: str | None = None) -> None:
        if not self.enabled:
            return
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        text = summary if summary is not None else (
            f"{self.label}  {self.done:,} frames in {_hms(elapsed)}  ({rate:,.0f} fps)")
        if self.isatty:
            pad = max(0, self._width_written - len(text))
            self._write("\r" + text + " " * pad, "\n")
        else:
            self._write(text, "\n")
        self.enabled = False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False
