"""V2 one-logical-message protocol with interchangeable language carriers."""

from __future__ import annotations

import hashlib
import hmac
import struct
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESSIV

from .compression import compression_name, compress_message, decompress_message
from .crypto import DerivedKeys, derive_keys, stream_crypt, v2_context
from .errors import AuthenticationError, CapacityExceeded, FrameError, InvalidArgument, PackMismatch, PCCError, TruncatedTranscript
from .framing import MAX_FRAME_BODY, encode_length, frame_target_from_bits
from .limits import MAX_CARRIER_TEXT_BYTES, MAX_TRANSCRIPT_BYTES, MAX_TRANSCRIPT_MESSAGES
from .model import HuffmanMarkovModel, TopicMarkovModel
from .neural import OnnxLanguageCarrier
from .pack import DialoguePack

PROFILES = ("secure", "compact", "dense")
DENSE_MAGIC = b"D"
COMPACT_TAG_SIZE = 8
MAX_SEALED = MAX_FRAME_BODY


def _bits_from_bytes(data: bytes) -> list[int]:
    return [(byte >> bit) & 1 for byte in data for bit in range(7, -1, -1)]


def _bytes_from_bits(bits: list[int]) -> bytes:
    if len(bits) % 8:
        raise FrameError("V2 frame is not byte aligned")
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


def _effective_profile(profile: str, interleave: bool) -> str:
    return profile + ("-interleaved" if interleave else "-ordered")


def _seal(keys: DerivedKeys, carrier_id: str, profile: str, sequence: int, interleave: bool, inner: bytes) -> bytes:
    effective = _effective_profile(profile, interleave)
    context = v2_context(carrier_id, effective, sequence)
    if profile == "secure":
        return AESSIV(keys.encryption).encrypt(inner, [context])
    ciphertext = stream_crypt(keys, carrier_id, effective, sequence, inner)
    if profile == "compact":
        tag = hmac.new(keys.authentication, context + ciphertext, hashlib.sha256).digest()[:COMPACT_TAG_SIZE]
        return ciphertext + tag
    return ciphertext


def _open(keys: DerivedKeys, carrier_id: str, profile: str, sequence: int, interleave: bool, sealed: bytes) -> bytes:
    effective = _effective_profile(profile, interleave)
    context = v2_context(carrier_id, effective, sequence)
    if profile == "secure":
        try:
            return AESSIV(keys.encryption).decrypt(sealed, [context])
        except InvalidTag as exc:
            raise AuthenticationError("V2 authenticated message failed") from exc
    ciphertext = sealed
    if profile == "compact":
        if len(sealed) < COMPACT_TAG_SIZE:
            raise AuthenticationError("V2 compact message failed authentication")
        ciphertext, tag = sealed[:-COMPACT_TAG_SIZE], sealed[-COMPACT_TAG_SIZE:]
        expected = hmac.new(keys.authentication, context + ciphertext, hashlib.sha256).digest()[:COMPACT_TAG_SIZE]
        if not hmac.compare_digest(tag, expected):
            raise AuthenticationError("V2 compact message failed authentication")
    return stream_crypt(keys, carrier_id, effective, sequence, ciphertext)


class _HMACRandom:
    def __init__(self, key: bytes, label: bytes):
        self.key = key
        self.label = label
        self.counter = 0
        self.buffer = bytearray()

    def _fill(self) -> None:
        self.buffer.extend(hmac.new(self.key, self.label + struct.pack(">Q", self.counter), hashlib.sha256).digest())
        self.counter += 1

    def randbelow(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        width = max(1, (bound.bit_length() + 7) // 8)
        ceiling = 1 << (width * 8)
        limit = ceiling - ceiling % bound
        while True:
            while len(self.buffer) < width:
                self._fill()
            value = int.from_bytes(self.buffer[:width], "big")
            del self.buffer[:width]
            if value < limit:
                return value % bound


def _permutation(keys: DerivedKeys, carrier_id: str, profile: str, sequence: int, length: int) -> list[int]:
    label = (
        b"pcc/v2/interleave\x00" + bytes.fromhex(carrier_id) + profile.encode("ascii")
        + struct.pack(">QH", sequence, length)
    )
    rng = _HMACRandom(keys.shuffle, label)
    indices = list(range(length))
    for index in range(length - 1, 0, -1):
        other = rng.randbelow(index + 1)
        indices[index], indices[other] = indices[other], indices[index]
    return indices


def _interleave(keys: DerivedKeys, carrier_id: str, profile: str, sequence: int, sealed: bytes) -> bytes:
    permutation = _permutation(keys, carrier_id, profile, sequence, len(sealed))
    return bytes(sealed[index] for index in permutation)


def _deinterleave(keys: DerivedKeys, carrier_id: str, profile: str, sequence: int, shuffled: bytes) -> bytes:
    permutation = _permutation(keys, carrier_id, profile, sequence, len(shuffled))
    original = bytearray(len(shuffled))
    for output_index, input_index in enumerate(permutation):
        original[input_index] = shuffled[output_index]
    return bytes(original)


def _pack_encode(pack: DialoguePack, keys: DerivedKeys, sequence: int, bits: list[int]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    offset = 0
    while offset < len(bits):
        position = len(messages)
        if position >= MAX_TRANSCRIPT_MESSAGES:
            raise CapacityExceeded("V2 carrier exceeds the message limit")
        segment, turn_index = divmod(position, pack.turn_count)
        arc_index = pack.arc_index(keys, sequence, segment)
        turn = pack.arcs[arc_index].turns[turn_index]
        available = min(turn.bits, len(bits) - offset)
        label = _read_int(bits, offset, available) << (turn.bits - available)
        offset += available
        messages.append({"role": turn.role, "text": pack.render(keys, arc_index, turn_index, label)})
    return messages


def _pack_decode(pack: DialoguePack, keys: DerivedKeys, sequence: int, messages: list[dict[str, Any]]) -> list[int]:
    bits: list[int] = []
    for position, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("text"), str):
            raise FrameError("V2 carrier message is malformed")
        text = message["text"]
        if len(text.encode("utf-8")) > MAX_CARRIER_TEXT_BYTES:
            raise FrameError("V2 carrier message is too large")
        segment, turn_index = divmod(position, pack.turn_count)
        arc_index = pack.arc_index(keys, sequence, segment)
        turn = pack.arcs[arc_index].turns[turn_index]
        role = message.get("role")
        if role is not None and role != turn.role:
            raise PackMismatch("V2 carrier role does not match the dialogue pack")
        label = pack.decode_label(keys, arc_index, turn_index, text)
        bits.extend((label >> bit) & 1 for bit in range(turn.bits - 1, -1, -1))
        frame_target = frame_target_from_bits(bits)
        if frame_target is not None and len(bits) >= frame_target[1]:
            if any(bits[frame_target[1] :]):
                raise FrameError("V2 pack carrier has non-zero trailing bits")
            if position != len(messages) - 1:
                raise PackMismatch("V2 pack carrier has trailing messages after the frame")
            return bits
    return bits


def _carrier_identity(carrier: DialoguePack | HuffmanMarkovModel | TopicMarkovModel | OnnxLanguageCarrier) -> tuple[str, str, bytes]:
    if isinstance(carrier, DialoguePack):
        return "pack", carrier.pack_id, carrier.salt
    if isinstance(carrier, (HuffmanMarkovModel, TopicMarkovModel, OnnxLanguageCarrier)):
        return "model", carrier.model_id, carrier.salt
    raise InvalidArgument("unsupported V2 carrier")


def encode_v2(
    carrier: DialoguePack | HuffmanMarkovModel | TopicMarkovModel | OnnxLanguageCarrier,
    passphrase: str | bytes,
    plaintext: bytes,
    *,
    profile: str = "secure",
    sequence: int | None = None,
    interleave: bool = True,
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise InvalidArgument("unknown V2 profile")
    if sequence is None or not isinstance(sequence, int) or isinstance(sequence, bool) or not 0 <= sequence < 2**64:
        raise InvalidArgument("sequence is mandatory and must be an unsigned 64-bit integer")
    kind, carrier_id, salt = _carrier_identity(carrier)
    keys = derive_keys(passphrase, salt)
    compressed = compress_message(plaintext)
    inner = DENSE_MAGIC + compressed if profile == "dense" else compressed
    sealed = _seal(keys, carrier_id, profile, sequence, interleave, inner)
    if len(sealed) > MAX_SEALED:
        raise CapacityExceeded("V2 sealed message exceeds the 64 KiB frame limit")
    encoded_sealed = _interleave(keys, carrier_id, profile, sequence, sealed) if interleave else sealed
    frame = encode_length(len(encoded_sealed)) + encoded_sealed
    bits = _bits_from_bytes(frame)
    attempt = 0
    if isinstance(carrier, DialoguePack):
        messages = _pack_encode(carrier, keys, sequence, bits)
    else:
        attempt = 0
        if isinstance(carrier, OnnxLanguageCarrier):
            attempts = max(1, carrier.quality_attempts)
            texts = []
            last_error: PCCError | None = None
            for candidate_attempt in range(attempts):
                try:
                    candidate = carrier.encode_bits(bits, keys.mapping, sequence, candidate_attempt)
                except PCCError as exc:
                    last_error = exc
                    continue
                if carrier.accepts_cover(" ".join(candidate)):
                    attempt = candidate_attempt
                    texts = candidate
                    break
            if not texts:
                if last_error is not None:
                    raise last_error
                raise CapacityExceeded("neural carrier produced no acceptable cover")
        else:
            texts = carrier.encode_bits(bits, keys.mapping, sequence)
        grouped = [" ".join(texts[offset : offset + 3]) for offset in range(0, len(texts), 3)]
        messages = [
            {"role": "Alice" if index % 2 == 0 else "Bob", "text": text}
            for index, text in enumerate(grouped)
        ]
    return {
        "format": "pcc/v2",
        "carrier": kind,
        "carrier_id": carrier_id,
        "profile": profile,
        "sequence": sequence,
        "interleave": interleave,
        "attempt": attempt,
        "compression": compression_name(compressed),
        "messages": messages,
    }


def decode_v2(
    carrier: DialoguePack | HuffmanMarkovModel | TopicMarkovModel | OnnxLanguageCarrier,
    passphrase: str | bytes,
    transcript: dict[str, Any],
) -> bytes:
    if not isinstance(transcript, dict) or transcript.get("format") != "pcc/v2":
        raise FrameError("unsupported V2 transcript format")
    kind, carrier_id, salt = _carrier_identity(carrier)
    if transcript.get("carrier") != kind or transcript.get("carrier_id") != carrier_id:
        raise PackMismatch("V2 transcript uses another carrier")
    profile = transcript.get("profile")
    sequence = transcript.get("sequence")
    interleave = transcript.get("interleave")
    if profile not in PROFILES or not isinstance(interleave, bool):
        raise FrameError("V2 transcript profile is invalid")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or not 0 <= sequence < 2**64:
        raise FrameError("V2 transcript sequence is invalid")
    messages = transcript.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TruncatedTranscript("V2 transcript contains no cover messages")
    if len(messages) > MAX_TRANSCRIPT_MESSAGES:
        raise FrameError("V2 transcript contains too many cover messages")
    transcript_bytes = 0
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("text"), str):
            raise FrameError("V2 carrier message is malformed")
        try:
            message_bytes = len(message["text"].encode("utf-8", "strict"))
        except UnicodeError as exc:
            raise FrameError("V2 carrier message contains invalid Unicode") from exc
        if message_bytes > MAX_CARRIER_TEXT_BYTES:
            raise FrameError("V2 carrier message is too large")
        transcript_bytes += message_bytes
        if transcript_bytes > MAX_TRANSCRIPT_BYTES:
            raise FrameError("V2 transcript text is too large")
    keys = derive_keys(passphrase, salt)
    if isinstance(carrier, DialoguePack):
        bits = _pack_decode(carrier, keys, sequence, messages)
        attempts = (0,)
    else:
        texts = []
        for message in messages:
            texts.append(message["text"])
        declared_attempt = transcript.get("attempt", 0)
        if isinstance(carrier, OnnxLanguageCarrier) and declared_attempt is None:
            attempts = tuple(range(max(1, carrier.quality_attempts)))
        else:
            if not isinstance(declared_attempt, int) or declared_attempt < 0:
                raise FrameError("V2 carrier attempt is invalid")
            attempts = (declared_attempt,)

    def open_bits(candidate_bits: list[int]) -> bytes:
        frame_target = frame_target_from_bits(candidate_bits)
        if frame_target is None:
            raise TruncatedTranscript("V2 transcript ended before the frame length")
        prefix_bits, target = frame_target
        sealed_length = (target - prefix_bits) // 8
        minimum = 16 if profile == "secure" else COMPACT_TAG_SIZE if profile == "compact" else 1
        if sealed_length < minimum or sealed_length > MAX_SEALED:
            raise FrameError("V2 encrypted frame length is invalid")
        if len(candidate_bits) < target:
            raise TruncatedTranscript("V2 transcript ended before the encrypted frame")
        encoded_sealed = _bytes_from_bits(candidate_bits[prefix_bits:target])
        sealed = _deinterleave(keys, carrier_id, profile, sequence, encoded_sealed) if interleave else encoded_sealed
        inner = _open(keys, carrier_id, profile, sequence, interleave, sealed)
        if profile == "dense":
            if len(inner) < 2 or inner[:1] != DENSE_MAGIC:
                raise AuthenticationError("V2 dense message key or framing is invalid")
            inner = inner[1:]
        if not inner:
            raise FrameError("V2 compressed payload is empty")
        return decompress_message(inner)

    if isinstance(carrier, DialoguePack):
        return open_bits(bits)
    last_error: PCCError | None = None
    for attempt in attempts:
        try:
            candidate_bits = carrier.decode_bits(texts, keys.mapping, sequence, attempt) if isinstance(carrier, OnnxLanguageCarrier) else carrier.decode_bits(texts, keys.mapping, sequence)
            return open_bits(candidate_bits)
        except PCCError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FrameError("V2 carrier produced no decodable attempt")
