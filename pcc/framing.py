"""Compact unsigned-length framing shared by V2 carriers."""

from __future__ import annotations

from .errors import FrameError

MAX_FRAME_BODY = 65535
MAX_LENGTH_BYTES = 3


def encode_length(value: int) -> bytes:
    if not 0 <= value <= MAX_FRAME_BODY:
        raise FrameError("V2 frame body length is invalid")
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(output)


def frame_target_from_bits(bits: list[int]) -> tuple[int, int] | None:
    """Return (prefix bits, total bits) once the variable length is complete."""
    value = 0
    shift = 0
    for index in range(MAX_LENGTH_BYTES):
        offset = index * 8
        if len(bits) < offset + 8:
            return None
        byte = 0
        for bit in bits[offset : offset + 8]:
            byte = (byte << 1) | bit
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if value > MAX_FRAME_BODY:
                raise FrameError("V2 frame body length is invalid")
            if len(encode_length(value)) != index + 1:
                raise FrameError("V2 frame length prefix is not canonical")
            prefix_bits = (index + 1) * 8
            return prefix_bits, prefix_bits + value * 8
        shift += 7
    raise FrameError("V2 frame length prefix is too long")
