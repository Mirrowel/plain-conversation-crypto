"""Reproducible V2 carrier/profile comparison."""

from __future__ import annotations

import time
import random
from typing import Any

from .compression import compression_candidates
from .model import HuffmanMarkovModel, TopicMarkovModel
from .pack import DialoguePack
from .v2 import PROFILES, decode_v2, encode_v2


SAMPLES = {
    "hello": b"hello",
    "short-chat": b"Meet me at the cafe after work.",
    "paragraph": (
        b"I finished the errands earlier than expected, so I should have time to stop by after work. "
        b"Let me know whether the usual place still works for you."
    ),
    "binary-64": bytes(range(64)),
    "unicode": "Café déjà vu. 你好。🙂".encode("utf-8"),
    "repetitive": b"please let me know when you get home. " * 64,
    "json": b'{"message":"meet me after work","urgent":false,"count":3}',
    "base64": b"AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=",
    "binary-random": random.Random(20260222).randbytes(256),
}


def compare(
    pack: DialoguePack,
    model: HuffmanMarkovModel | TopicMarkovModel,
    passphrase: str | bytes,
    *,
    interleave: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for carrier_name, carrier in (("pack", pack), ("model", model)):
        for profile in PROFILES:
            for sequence, (sample_name, plaintext) in enumerate(SAMPLES.items()):
                started = time.perf_counter()
                transcript = encode_v2(
                    carrier,
                    passphrase,
                    plaintext,
                    profile=profile,
                    sequence=sequence,
                    interleave=interleave,
                )
                encoded_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                recovered = decode_v2(carrier, passphrase, transcript)
                decoded_ms = (time.perf_counter() - started) * 1000
                if recovered != plaintext:
                    raise AssertionError("benchmark round trip failed")
                visible_chars = sum(len(message["text"]) for message in transcript["messages"])
                rows.append({
                    "carrier": carrier_name,
                    "profile": profile,
                    "sample": sample_name,
                    "plaintext_bytes": len(plaintext),
                    "messages": len(transcript["messages"]),
                    "visible_chars": visible_chars,
                    "cover_chars_per_plaintext_byte": round(visible_chars / max(1, len(plaintext)), 3),
                    "encode_ms": round(encoded_ms, 3),
                    "decode_ms": round(decoded_ms, 3),
                    "compression": transcript["compression"],
                })
    return {
        "format": "pcc/v2-benchmark",
        "interleave": interleave,
        "pack_id": pack.pack_id,
        "model_id": model.model_id,
        "model_order": model.order,
        "model_contexts": model.context_count,
        "rows": rows,
    }


def compression_compare() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for sample_name, plaintext in SAMPLES.items():
        candidates = compression_candidates(plaintext)
        best = min(candidates, key=lambda candidate: len(candidate.payload))
        rows.append({
            "sample": sample_name,
            "plaintext_bytes": len(plaintext),
            "best": best.name,
            "best_bytes": len(best.payload),
            "best_ratio": round(len(best.payload) / max(1, len(plaintext)), 4),
            "all": {candidate.name: len(candidate.payload) for candidate in candidates},
        })
    wins: dict[str, int] = {}
    for row in rows:
        wins[row["best"]] = wins.get(row["best"], 0) + 1
    return {
        "format": "pcc/v2-compression-benchmark",
        "dictionary": "sha256:see pcc.compression.DICTIONARY_SHA256",
        "rows": rows,
        "wins": wins,
    }
