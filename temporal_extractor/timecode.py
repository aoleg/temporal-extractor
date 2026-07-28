"""
Parsing and formatting for the timestamps accepted by `--segment FROM TO`.

Accepted formats, most to least specific:

    "1:15:36"    H:MM:SS
    "1:15:36.5"  H:MM:SS.sss
    "15:36"      MM:SS
    "36"         SS, same as "00:00:36"

Deliberately lenient about range: components are summed rather than validated
against a 0-59 ceiling, so "1:75:00" is accepted as 1h75m (= 2h15m) rather than
rejected. A user who typed that meant something by it.
"""


def parse_timestamp(text: str) -> float:
    """Parse a colon-separated timestamp into seconds. Raises ValueError."""
    raw = text.strip()
    if not raw:
        raise ValueError("empty timestamp")
    parts = raw.split(":")
    if len(parts) > 3:
        raise ValueError(
            f"invalid timestamp {text!r}: expected H:MM:SS[.sss], MM:SS[.sss] or SS[.sss]")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"invalid timestamp {text!r}: not a number") from None
    if any(n < 0 for n in nums):
        raise ValueError(f"invalid timestamp {text!r}: negative component")
    while len(nums) < 3:
        nums.insert(0, 0.0)
    hours, minutes, seconds = nums
    return hours * 3600.0 + minutes * 60.0 + seconds


def format_timestamp(seconds: float) -> str:
    """H:MM:SS.sss for display, hours omitted when zero. Not a parse inverse."""
    seconds = max(0.0, seconds)
    whole = int(seconds)
    ms = round((seconds - whole) * 1000)
    if ms == 1000:
        ms = 0
        whole += 1
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    base = f"{m:02d}:{s:02d}.{ms:03d}"
    return f"{h}:{base}" if h else base
