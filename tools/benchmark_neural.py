from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pcc.neural import OnnxLanguageCarrier
from pcc.v2 import decode_v2, encode_v2


SAMPLES = {
    "hello": b"hello",
    "short-chat": b"Meet me at the cafe after work.",
    "paragraph": b"I finished the errands earlier than expected, so I should have time to stop by after work. Let me know whether the usual place still works for you.",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", default="benchmark key only")
    args = parser.parse_args()
    carrier = OnnxLanguageCarrier.load(args.manifest)
    rows = []
    for profile in ("secure", "compact", "dense"):
        for sequence, (name, plaintext) in enumerate(SAMPLES.items()):
            started = time.perf_counter()
            transcript = encode_v2(carrier, args.key, plaintext, profile=profile, sequence=sequence)
            encode_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            recovered = decode_v2(carrier, args.key, transcript)
            decode_ms = (time.perf_counter() - started) * 1000
            if recovered != plaintext:
                raise AssertionError("neural round trip failed")
            text = " ".join(message["text"] for message in transcript["messages"])
            rows.append({
                "profile": profile,
                "sample": name,
                "plaintext_bytes": len(plaintext),
                "messages": len(transcript["messages"]),
                "visible_chars": len(text),
                "chars_per_byte": round(len(text) / max(1, len(plaintext)), 3),
                "encode_ms": round(encode_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "cover_text": text,
            })
    Path(args.output).write_text(json.dumps({"carrier_id": carrier.model_id, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
