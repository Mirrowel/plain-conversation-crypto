"""Reversible conversion between encrypted bytes and pack-selected messages."""

from __future__ import annotations

import json
import secrets
import struct
from typing import Any

from .crypto import MAX_PLAINTEXT, derive_keys, decrypt, encrypt
from .errors import CapacityExceeded, FrameError, InvalidArgument, PackMismatch, TruncatedTranscript
from .limits import MAX_CARRIER_TEXT_BYTES, MAX_TRANSCRIPT_MESSAGES
from .pack import DialoguePack

FRAME_LENGTH_SIZE = 4
MAX_SEALED = MAX_PLAINTEXT + 1024
DEFAULT_COVER_BUCKET = 8
INNER_MAGIC = b"PCC1"
INNER_HEADER_SIZE = len(INNER_MAGIC) + 2


def _bits_from_bytes(data: bytes) -> list[int]:
    return [(byte >> bit) & 1 for byte in data for bit in range(7, -1, -1)]


def _bytes_from_bits(bits: list[int]) -> bytes:
    if len(bits) % 8:
        raise FrameError("bit stream is not byte aligned")
    output = bytearray()
    for offset in range(0, len(bits), 8):
        value = 0
        for bit in bits[offset : offset + 8]:
            value = (value << 1) | bit
        output.append(value)
    return bytes(output)


def _read_int(bits: list[int], offset: int, count: int) -> int:
    value = 0
    for bit in bits[offset : offset + count]:
        value = (value << 1) | bit
    return value


def encode_message(
    pack: DialoguePack,
    passphrase: str | bytes,
    plaintext: bytes,
    sequence: int = 0,
    cover_bucket: int = DEFAULT_COVER_BUCKET,
    random_cover: bool = True,
) -> dict[str, Any]:
    if cover_bucket < 0:
        raise InvalidArgument("cover bucket must not be negative")
    if cover_bucket and cover_bucket & (cover_bucket - 1):
        raise InvalidArgument("cover bucket must be a power of two")
    if cover_bucket > 32768:
        raise InvalidArgument("cover bucket is too large")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0 or sequence >= 2**64:
        raise InvalidArgument("sequence must be an unsigned 64-bit integer")
    if len(plaintext) + INNER_HEADER_SIZE > MAX_PLAINTEXT:
        raise CapacityExceeded("plaintext is too large")
    keys = derive_keys(passphrase, pack.salt)
    # The tail count is encrypted so an active intermediary cannot strip the
    # cover padding without making the transcript fail after decryption.
    sealed_length = len(plaintext) + INNER_HEADER_SIZE + 16
    frame_bits = (FRAME_LENGTH_SIZE + sealed_length) * 8
    required_messages = _required_messages(pack, keys, sequence, frame_bits)
    tail_count = 0
    if cover_bucket:
        target_messages = ((required_messages + cover_bucket - 1) // cover_bucket) * cover_bucket
        target_messages += secrets.randbelow(cover_bucket) if random_cover else 0
        tail_count = target_messages - required_messages
    if tail_count > 65535:
        raise CapacityExceeded("cover padding exceeds the protocol limit")
    if required_messages + tail_count > MAX_TRANSCRIPT_MESSAGES:
        raise CapacityExceeded("carrier would exceed the transcript message limit")
    inner = INNER_MAGIC + struct.pack(">H", tail_count) + plaintext
    sealed = encrypt(keys, pack.pack_id, sequence, inner)
    if len(sealed) > MAX_SEALED:
        raise CapacityExceeded("encrypted message is too large")
    frame = struct.pack(">I", len(sealed)) + sealed
    bits = _bits_from_bytes(frame)
    messages: list[dict[str, str]] = []
    offset = 0
    segment = 0
    while offset < len(bits):
        arc_index = pack.arc_index(keys, sequence, segment)
        arc = pack.arcs[arc_index]
        for turn_index, turn in enumerate(arc.turns):
            label = _read_int(bits, offset, min(turn.bits, len(bits) - offset))
            available = min(turn.bits, len(bits) - offset)
            offset += available
            if available < turn.bits:
                label <<= turn.bits - available
            messages.append({"role": turn.role, "text": pack.render(keys, arc_index, turn_index, label)})
            if offset >= len(bits):
                break
        segment += 1
    while len(messages) < required_messages + tail_count:
        position = len(messages)
        segment, turn_index = divmod(position, pack.turn_count)
        arc_index = pack.arc_index(keys, sequence, segment)
        turn = pack.arcs[arc_index].turns[turn_index]
        label = secrets.randbelow(2**turn.bits) if random_cover else 0
        messages.append({"role": turn.role, "text": pack.render(keys, arc_index, turn_index, label)})
    return {
        "format": "pcc/v1",
        "pack_id": pack.pack_id,
        "sequence": sequence,
        "messages": messages,
    }


def decode_message(pack: DialoguePack, passphrase: str | bytes, transcript: dict[str, Any]) -> bytes:
    if transcript.get("format") != "pcc/v1":
        raise FrameError("unsupported transcript format")
    if transcript.get("pack_id") != pack.pack_id:
        raise PackMismatch("transcript was created for another dialogue pack")
    sequence = transcript.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0 or sequence >= 2**64:
        raise FrameError("invalid transcript sequence")
    messages = transcript.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TruncatedTranscript("transcript contains no messages")
    if len(messages) > MAX_TRANSCRIPT_MESSAGES:
        raise FrameError("transcript contains too many messages")
    keys = derive_keys(passphrase, pack.salt)
    bits: list[int] = []
    complete_target: int | None = None
    frame_complete = False
    frame_message_count: int | None = None
    for position, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("text"), str):
            raise FrameError("transcript message is malformed")
        try:
            carrier_text_bytes = len(message["text"].encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise FrameError("carrier message contains invalid Unicode") from exc
        if carrier_text_bytes > MAX_CARRIER_TEXT_BYTES:
            raise FrameError("carrier message is too large")
        segment, turn_index = divmod(position, pack.turn_count)
        arc_index = pack.arc_index(keys, sequence, segment)
        turn = pack.arcs[arc_index].turns[turn_index]
        role = message.get("role")
        if role != turn.role:
            raise PackMismatch(f"unexpected role at transcript position {position}")
        label = pack.decode_label(keys, arc_index, turn_index, message["text"])
        if frame_complete:
            continue
        bits.extend((label >> bit) & 1 for bit in range(turn.bits - 1, -1, -1))
        if complete_target is None and len(bits) >= FRAME_LENGTH_SIZE * 8:
            sealed_length = _read_int(bits, 0, FRAME_LENGTH_SIZE * 8)
            if sealed_length < 16 or sealed_length > MAX_SEALED:
                raise FrameError("invalid encrypted frame length")
            complete_target = (FRAME_LENGTH_SIZE + sealed_length) * 8
        if complete_target is not None and len(bits) >= complete_target:
            if any(bits[complete_target:]):
                raise FrameError("encrypted frame has non-zero trailing bits")
            frame_complete = True
            frame_message_count = position + 1
    if complete_target is None or len(bits) < complete_target:
        raise TruncatedTranscript("transcript ended before the encrypted frame")
    sealed_length = _read_int(bits, 0, FRAME_LENGTH_SIZE * 8)
    sealed = _bytes_from_bits(bits[FRAME_LENGTH_SIZE * 8 : FRAME_LENGTH_SIZE * 8 + sealed_length * 8])
    inner = decrypt(keys, pack.pack_id, sequence, sealed)
    if len(inner) < INNER_HEADER_SIZE or inner[: len(INNER_MAGIC)] != INNER_MAGIC:
        raise FrameError("invalid encrypted message envelope")
    tail_count = struct.unpack(">H", inner[len(INNER_MAGIC) : INNER_HEADER_SIZE])[0]
    if frame_message_count is None or len(messages) != frame_message_count + tail_count:
        raise FrameError("cover padding count does not match the authenticated envelope")
    return inner[INNER_HEADER_SIZE:]


def _required_messages(pack: DialoguePack, keys, sequence: int, frame_bits: int) -> int:
    offset = 0
    position = 0
    while offset < frame_bits:
        if position >= MAX_TRANSCRIPT_MESSAGES:
            raise CapacityExceeded("carrier would exceed the transcript message limit")
        segment, turn_index = divmod(position, pack.turn_count)
        arc_index = pack.arc_index(keys, sequence, segment)
        offset += pack.arcs[arc_index].turns[turn_index].bits
        position += 1
    return position


def transcript_json(transcript: dict[str, Any]) -> str:
    return json.dumps(transcript, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
