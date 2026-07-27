"""
Wire protocol between the tool (.venv) and the SeedVR2 worker (.venv-seedvr2).

Stdlib only -- this module is imported by both interpreters, so it must not
depend on anything either side might not have.

Why a socket rather than stdin/stdout: the SeedVR2 reference code prints a
banner and progress logs to stdout unconditionally, so stdout is not a usable
channel for framed data. The parent listens on an ephemeral localhost port,
hands the port to the worker on the command line, and leaves the worker's
stdout/stderr free for logging.

Framing is: 4-byte header length, 4-byte payload length (both big-endian
unsigned), then the UTF-8 JSON header, then the raw payload bytes. Payload is
used for ndarray data so frames never go through JSON.
"""

import json
import socket
import struct

HEADER_STRUCT = struct.Struct(">II")

# Bumped when the message shapes change incompatibly; the worker reports its
# version at handshake and the client refuses a mismatch rather than hanging
# on a subtly wrong reply.
PROTOCOL_VERSION = 1


class ProtocolError(RuntimeError):
    """Framing broke, or the peer said something we cannot parse."""


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes, or raise. socket.recv is free to return fewer."""
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ProtocolError(f"peer closed with {remaining} of {n} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    raw_header = json.dumps(header).encode("utf-8")
    sock.sendall(HEADER_STRUCT.pack(len(raw_header), len(payload)))
    sock.sendall(raw_header)
    if payload:
        sock.sendall(payload)


def recv_message(sock: socket.socket) -> tuple[dict, bytes]:
    header_len, payload_len = HEADER_STRUCT.unpack(_recv_exactly(sock, HEADER_STRUCT.size))
    try:
        header = json.loads(_recv_exactly(sock, header_len).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"malformed header: {exc}") from exc
    payload = _recv_exactly(sock, payload_len) if payload_len else b""
    return header, payload


def array_header(array) -> dict:
    """Describe an ndarray well enough to rebuild it from raw bytes."""
    return {"shape": list(array.shape), "dtype": array.dtype.str}
