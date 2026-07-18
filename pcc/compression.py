"""Versioned lossless compression portfolio for short conversational data."""

from __future__ import annotations

import bz2
import hashlib
import lzma
import zlib
from dataclasses import dataclass

import brotli
import zstandard as zstd

from .crypto import MAX_PLAINTEXT
from .errors import FrameError

# 0x00..0x7f are one-byte exact-message codes. Codec payloads begin at 0x80.
MODE_RAW = 0x80
MODE_ZLIB = 0x81
MODE_ZLIB_DICT = 0x82
MODE_DEFLATE = 0x83
MODE_DEFLATE_DICT = 0x84
MODE_BZ2 = 0x85
MODE_LZMA = 0x86
MODE_ZSTD_3 = 0x87
MODE_ZSTD_19 = 0x88
MODE_ZSTD_DICT = 0x89
MODE_BROTLI_5 = 0x8A
MODE_BROTLI_11 = 0x8B
MODE_TOKENS = 0x90
MODE_TOKENS_DEFLATE = 0x91
MODE_TOKENS_ZSTD = 0x92

MAX_COMPRESSED_BODY = 65535

COMMON_MESSAGES = (
    b"hello", b"hi", b"hey", b"okay", b"ok", b"yes", b"no", b"thanks",
    b"thank you", b"good morning", b"good night", b"on my way", b"be right there",
    b"call me", b"text me", b"let me know", b"sounds good", b"that works",
    b"see you soon", b"see you later", b"where are you?", b"are you free?",
    b"what time?", b"i'm here", b"i'm home", b"i'll be there soon",
    b"meet me outside", b"can you talk?", b"not yet", b"maybe later",
    b"i agree", b"i don't know", b"how are you?", b"what are you doing?",
    b"i miss you", b"love you", b"talk to you later", b"have a good day",
    b"safe travels", b"i'm running late", b"i'll call you", b"send me the address",
    b"when will you be home?", b"do you need anything?", b"i'm almost there",
    b"take your time", b"no problem", b"that's fine", b"perfect", b"maybe",
    b"definitely", b"probably", b"i think so", b"i don't think so",
)
if len(COMMON_MESSAGES) > 128:
    raise RuntimeError("common-message table exceeds its one-byte namespace")
COMMON_TO_INDEX = {message: index for index, message in enumerate(COMMON_MESSAGES)}

PHRASES = (
    b" after work", b" before work", b" this morning", b" this afternoon",
    b" this evening", b" tomorrow morning", b" tomorrow evening", b" later today",
    b" let me know", b" sounds good", b" on my way", b" be right there",
    b" see you later", b" thank you", b" no problem", b" at the cafe",
    b" at home", b" outside", b" when you arrive", b" when you get home",
    b"I will ", b"I'll ", b"I'm ", b"I have ", b"I think ", b"I don't ",
    b"you can ", b"we can ", b"do you ", b"can you ", b"would you ",
    b"the ", b"and ", b"that ", b"with ", b"for ", b"from ", b"about ",
    b"message", b"meeting", b"tomorrow", b"tonight", b"today", b"please",
)
if len(PHRASES) > 256:
    raise RuntimeError("phrase table exceeds its one-byte index")
PHRASES_BY_LENGTH = sorted(enumerate(PHRASES), key=lambda item: (-len(item[1]), item[0]))

CHAT_DICTIONARY = (
    b"hello hi hey okay yes no thanks thank you please sorry good morning good night "
    b"on my way let me know sounds good see you later where are you what time meet me "
    b"after work before lunch this evening tomorrow today call text message home outside "
    b"address arrive later soon probably definitely maybe perfect meeting cafe restaurant "
    b"I you we they the and that this with for from have will can should would about"
)
DICTIONARY_SHA256 = hashlib.sha256(CHAT_DICTIONARY).hexdigest()


@dataclass(frozen=True)
class CompressionCandidate:
    name: str
    payload: bytes


MODE_NAMES = {
    MODE_RAW: "raw",
    MODE_ZLIB: "zlib",
    MODE_ZLIB_DICT: "zlib-dict",
    MODE_DEFLATE: "deflate",
    MODE_DEFLATE_DICT: "deflate-dict",
    MODE_BZ2: "bz2",
    MODE_LZMA: "lzma",
    MODE_ZSTD_3: "zstd-3",
    MODE_ZSTD_19: "zstd-19",
    MODE_ZSTD_DICT: "zstd-dict",
    MODE_BROTLI_5: "brotli-5",
    MODE_BROTLI_11: "brotli-11",
    MODE_TOKENS: "tokens",
    MODE_TOKENS_DEFLATE: "tokens-deflate",
    MODE_TOKENS_ZSTD: "tokens-zstd",
}


def compression_name(payload: bytes) -> str:
    if not payload:
        raise FrameError("compressed payload is empty")
    if payload[0] < 0x80:
        return "common"
    try:
        return MODE_NAMES[payload[0]]
    except KeyError as exc:
        raise FrameError("unknown compression mode") from exc


def _zlib_compress(data: bytes, *, level: int, wbits: int, dictionary: bytes | None = None) -> bytes:
    options = {"level": level, "wbits": wbits}
    if dictionary is not None:
        options["zdict"] = dictionary
    compressor = zlib.compressobj(**options)
    return compressor.compress(data) + compressor.flush()


def _zlib_decompress(data: bytes, *, wbits: int, dictionary: bytes | None = None) -> bytes:
    try:
        options = {"wbits": wbits}
        if dictionary is not None:
            options["zdict"] = dictionary
        decompressor = zlib.decompressobj(**options)
        result = decompressor.decompress(data, MAX_PLAINTEXT + 1)
        if decompressor.unconsumed_tail or len(result) > MAX_PLAINTEXT:
            raise FrameError("decompressed message is too large")
        result += decompressor.flush()
    except zlib.error as exc:
        raise FrameError("compressed message is invalid") from exc
    if not decompressor.eof or decompressor.unused_data or len(result) > MAX_PLAINTEXT:
        raise FrameError("compressed message is invalid")
    return result


def _tokenize(data: bytes) -> bytes:
    output = bytearray()
    literals = bytearray()

    def flush_literals() -> None:
        while literals:
            chunk = bytes(literals[:255])
            del literals[:255]
            output.extend((0, len(chunk)))
            output.extend(chunk)

    offset = 0
    while offset < len(data):
        match = next(((index, phrase) for index, phrase in PHRASES_BY_LENGTH if data.startswith(phrase, offset)), None)
        if match is None:
            literals.append(data[offset])
            offset += 1
            continue
        flush_literals()
        index, phrase = match
        output.extend((1, index))
        offset += len(phrase)
    flush_literals()
    return bytes(output)


def _detokenize(data: bytes) -> bytes:
    output = bytearray()
    offset = 0
    while offset < len(data):
        opcode = data[offset]
        offset += 1
        if offset >= len(data):
            raise FrameError("token stream is truncated")
        value = data[offset]
        offset += 1
        if opcode == 0:
            if value == 0 or offset + value > len(data):
                raise FrameError("token literal is invalid")
            output.extend(data[offset : offset + value])
            offset += value
        elif opcode == 1:
            if value >= len(PHRASES):
                raise FrameError("token phrase index is invalid")
            output.extend(PHRASES[value])
        else:
            raise FrameError("unknown token opcode")
        if len(output) > MAX_PLAINTEXT:
            raise FrameError("detokenized message is too large")
    return bytes(output)


def _zstd_dictionary():
    return zstd.ZstdCompressionDict(CHAT_DICTIONARY, dict_type=zstd.DICT_TYPE_RAWCONTENT)


def compression_candidates(plaintext: bytes) -> list[CompressionCandidate]:
    """Evaluate all V2 codecs. Candidate order is the deterministic tie-break."""

    candidates = [CompressionCandidate("raw", bytes((MODE_RAW,)) + plaintext)]
    common = COMMON_TO_INDEX.get(plaintext)
    if common is not None:
        candidates.append(CompressionCandidate("common", bytes((common,))))
    candidates.extend((
        CompressionCandidate("zlib", bytes((MODE_ZLIB,)) + zlib.compress(plaintext, level=9)),
        CompressionCandidate("zlib-dict", bytes((MODE_ZLIB_DICT,)) + _zlib_compress(plaintext, level=9, wbits=zlib.MAX_WBITS, dictionary=CHAT_DICTIONARY)),
        CompressionCandidate("deflate", bytes((MODE_DEFLATE,)) + _zlib_compress(plaintext, level=9, wbits=-zlib.MAX_WBITS)),
        CompressionCandidate("deflate-dict", bytes((MODE_DEFLATE_DICT,)) + _zlib_compress(plaintext, level=9, wbits=-zlib.MAX_WBITS, dictionary=CHAT_DICTIONARY)),
        CompressionCandidate("bz2", bytes((MODE_BZ2,)) + bz2.compress(plaintext, compresslevel=9)),
        CompressionCandidate("lzma", bytes((MODE_LZMA,)) + lzma.compress(plaintext, preset=9 | lzma.PRESET_EXTREME)),
        CompressionCandidate("zstd-3", bytes((MODE_ZSTD_3,)) + zstd.ZstdCompressor(level=3).compress(plaintext)),
        CompressionCandidate("zstd-19", bytes((MODE_ZSTD_19,)) + zstd.ZstdCompressor(level=19).compress(plaintext)),
        CompressionCandidate("zstd-dict", bytes((MODE_ZSTD_DICT,)) + zstd.ZstdCompressor(level=12, dict_data=_zstd_dictionary()).compress(plaintext)),
        CompressionCandidate("brotli-5", bytes((MODE_BROTLI_5,)) + brotli.compress(plaintext, quality=5)),
        CompressionCandidate("brotli-11", bytes((MODE_BROTLI_11,)) + brotli.compress(plaintext, quality=11)),
    ))
    tokenized = _tokenize(plaintext)
    candidates.extend((
        CompressionCandidate("tokens", bytes((MODE_TOKENS,)) + tokenized),
        CompressionCandidate("tokens-deflate", bytes((MODE_TOKENS_DEFLATE,)) + _zlib_compress(tokenized, level=9, wbits=-zlib.MAX_WBITS)),
        CompressionCandidate("tokens-zstd", bytes((MODE_TOKENS_ZSTD,)) + zstd.ZstdCompressor(level=12).compress(tokenized)),
    ))
    return candidates


def compress_message(plaintext: bytes) -> bytes:
    if len(plaintext) > MAX_PLAINTEXT:
        raise FrameError("plaintext is too large to compress")
    return min(compression_candidates(plaintext), key=lambda candidate: len(candidate.payload)).payload


def _bounded_bz2(data: bytes) -> bytes:
    try:
        decoder = bz2.BZ2Decompressor()
        result = decoder.decompress(data, max_length=MAX_PLAINTEXT + 1)
    except OSError as exc:
        raise FrameError("BZ2 message is invalid") from exc
    if not decoder.eof or decoder.unused_data or len(result) > MAX_PLAINTEXT:
        raise FrameError("BZ2 message is invalid or too large")
    return result


def _bounded_lzma(data: bytes) -> bytes:
    try:
        decoder = lzma.LZMADecompressor()
        result = decoder.decompress(data, max_length=MAX_PLAINTEXT + 1)
    except lzma.LZMAError as exc:
        raise FrameError("LZMA message is invalid") from exc
    if not decoder.eof or decoder.unused_data or len(result) > MAX_PLAINTEXT:
        raise FrameError("LZMA message is invalid or too large")
    return result


def _bounded_brotli(data: bytes) -> bytes:
    try:
        decoder = brotli.Decompressor()
        output = bytearray()
        for offset in range(0, len(data), 1024):
            output.extend(decoder.process(data[offset : offset + 1024]))
            if len(output) > MAX_PLAINTEXT:
                raise FrameError("Brotli message is too large")
        if not decoder.is_finished():
            raise FrameError("Brotli message is truncated")
    except brotli.error as exc:
        raise FrameError("Brotli message is invalid") from exc
    return bytes(output)


def _bounded_zstd(data: bytes, dictionary=None) -> bytes:
    try:
        return zstd.ZstdDecompressor(dict_data=dictionary).decompress(
            data,
            max_output_size=MAX_PLAINTEXT,
            allow_extra_data=False,
        )
    except zstd.ZstdError as exc:
        raise FrameError("Zstandard message is invalid or too large") from exc


def decompress_message(payload: bytes) -> bytes:
    if not payload:
        raise FrameError("compressed payload is empty")
    mode, body = payload[0], payload[1:]
    if mode < 0x80:
        if len(payload) != 1 or mode >= len(COMMON_MESSAGES):
            raise FrameError("invalid common-message code")
        return COMMON_MESSAGES[mode]
    if len(body) > MAX_COMPRESSED_BODY:
        raise FrameError("compressed payload is too large")
    if mode == MODE_RAW:
        if len(body) > MAX_PLAINTEXT:
            raise FrameError("raw message is too large")
        return body
    if mode == MODE_ZLIB:
        return _zlib_decompress(body, wbits=zlib.MAX_WBITS)
    if mode == MODE_ZLIB_DICT:
        return _zlib_decompress(body, wbits=zlib.MAX_WBITS, dictionary=CHAT_DICTIONARY)
    if mode == MODE_DEFLATE:
        return _zlib_decompress(body, wbits=-zlib.MAX_WBITS)
    if mode == MODE_DEFLATE_DICT:
        return _zlib_decompress(body, wbits=-zlib.MAX_WBITS, dictionary=CHAT_DICTIONARY)
    if mode == MODE_BZ2:
        return _bounded_bz2(body)
    if mode == MODE_LZMA:
        return _bounded_lzma(body)
    if mode == MODE_ZSTD_3 or mode == MODE_ZSTD_19:
        return _bounded_zstd(body)
    if mode == MODE_ZSTD_DICT:
        return _bounded_zstd(body, _zstd_dictionary())
    if mode == MODE_BROTLI_5 or mode == MODE_BROTLI_11:
        return _bounded_brotli(body)
    if mode == MODE_TOKENS:
        return _detokenize(body)
    if mode == MODE_TOKENS_DEFLATE:
        return _detokenize(_zlib_decompress(body, wbits=-zlib.MAX_WBITS))
    if mode == MODE_TOKENS_ZSTD:
        return _detokenize(_bounded_zstd(body))
    raise FrameError("unknown compression mode")
