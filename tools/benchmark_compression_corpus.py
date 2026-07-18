from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from pcc.compression import compression_candidates, decompress_message


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def grouped(lines: list[str], count: int) -> list[bytes]:
    usable = len(lines) - len(lines) % count
    return [" ".join(lines[offset : offset + count]).encode("utf-8") for offset in range(0, usable, count)]


def measure(messages: list[bytes]) -> dict:
    wins: Counter[str] = Counter()
    ratios: list[float] = []
    original_sizes: list[int] = []
    best_sizes: list[int] = []
    codec_ratios: dict[str, list[float]] = defaultdict(list)
    length_wins: dict[str, Counter[str]] = defaultdict(Counter)
    for message in messages:
        candidates = compression_candidates(message)
        best = min(candidates, key=lambda candidate: len(candidate.payload))
        if decompress_message(best.payload) != message:
            raise AssertionError("compression benchmark round trip failed")
        original = len(message)
        ratio = len(best.payload) / original
        wins[best.name] += 1
        ratios.append(ratio)
        original_sizes.append(original)
        best_sizes.append(len(best.payload))
        bucket = "<=64" if original <= 64 else "65-128" if original <= 128 else "129-256" if original <= 256 else ">256"
        length_wins[bucket][best.name] += 1
        for candidate in candidates:
            codec_ratios[candidate.name].append(len(candidate.payload) / original)
    return {
        "messages": len(messages),
        "plaintext_bytes_median": statistics.median(original_sizes),
        "plaintext_bytes_mean": round(statistics.mean(original_sizes), 3),
        "compressed_bytes_median": statistics.median(best_sizes),
        "best_ratio_median": round(statistics.median(ratios), 4),
        "best_ratio_mean": round(statistics.mean(ratios), 4),
        "best_ratio_p90": round(percentile(ratios, 0.90), 4),
        "winner_counts": dict(wins.most_common()),
        "winner_percent": {name: round(count * 100 / len(messages), 2) for name, count in wins.most_common()},
        "median_ratio_by_codec": {
            name: round(statistics.median(values), 4)
            for name, values in sorted(codec_ratios.items())
        },
        "winner_counts_by_length": {bucket: dict(counter.most_common()) for bucket, counter in length_wins.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="packs/model_corpus.txt")
    parser.add_argument("--output")
    args = parser.parse_args()
    lines = [line.strip() for line in Path(args.corpus).read_text(encoding="utf-8").splitlines() if line.strip()]
    report = {
        "format": "pcc/compression-corpus-v1",
        "source_lines": len(lines),
        "groups": {
            f"{count}-sentence": measure(grouped(lines, count))
            for count in (1, 2, 3)
        },
    }
    output = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8", newline="\n")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
