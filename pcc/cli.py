from __future__ import annotations

import argparse
import getpass
import json
import random
import re
import secrets
import sys

from .benchmark import compare, compression_compare
from .codec import INNER_HEADER_SIZE, MAX_PLAINTEXT, decode_message, encode_message, transcript_json
from .crypto import derive_keys
from .errors import PCCError
from .limits import MAX_TRANSCRIPT_BYTES
from .model import HuffmanMarkovModel, TopicMarkovModel
from .neural import OnnxLanguageCarrier
from .pack import DialoguePack
from .v2 import PROFILES, decode_v2, encode_v2

MAX_PLAINTEXT_INPUT = MAX_PLAINTEXT - INNER_HEADER_SIZE


def _pack(path: str) -> DialoguePack:
    return DialoguePack.load(path)


def _read_transcript(path: str | None) -> dict:
    if path:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_TRANSCRIPT_BYTES + 1)
    else:
        stream = getattr(sys.stdin, "buffer", sys.stdin)
        raw = stream.read(MAX_TRANSCRIPT_BYTES + 1)
    if len(raw) > MAX_TRANSCRIPT_BYTES:
        raise ValueError("transcript is too large")
    return json.loads(raw)


def _carrier(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        source = json.load(handle)
    if source.get("kind") == "markov":
        return HuffmanMarkovModel.from_dict(source)
    if source.get("kind") == "topic-markov":
        return TopicMarkovModel.from_dict(source)
    if source.get("kind") == "neural-onnx":
        return OnnxLanguageCarrier.load(path)
    return DialoguePack.from_dict(source)


def _plain_transcript(path: str | None, carrier, profile: str, sequence: int, interleave: bool) -> dict:
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read(MAX_TRANSCRIPT_BYTES + 1)
    else:
        text = sys.stdin.read(MAX_TRANSCRIPT_BYTES + 1)
    if len(text.encode("utf-8")) > MAX_TRANSCRIPT_BYTES:
        raise ValueError("transcript is too large")
    messages = [part.strip() for part in re.split(r"\r?\n\s*\r?\n", text) if part.strip()]
    if not messages:
        raise ValueError("transcript contains no cover messages")
    return {
        "format": "pcc/v2",
        "carrier": "pack" if isinstance(carrier, DialoguePack) else "model",
        "carrier_id": carrier.pack_id if isinstance(carrier, DialoguePack) else carrier.model_id,
        "profile": profile,
        "sequence": sequence,
        "interleave": interleave,
        "messages": [{"text": message} for message in messages],
    }


def _key(value: str | None) -> str:
    if value is not None:
        return value
    return getpass.getpass("Key: ")


def _read_limited(stream, limit: int) -> bytes:
    data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("plaintext is too large")
    return data


def main() -> int:
    try:
        return _main()
    except (PCCError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _main() -> int:
    parser = argparse.ArgumentParser(description="Encode encrypted messages into ordinary dialogue-pack text")
    sub = parser.add_subparsers(dest="command", required=True)

    encode = sub.add_parser("encode", help="encrypt and encode a message")
    encode.add_argument("--pack", required=True)
    encode.add_argument("--key", help="unsafe convenience option; omitted values are prompted securely")
    encode.add_argument("--message", help="unsafe convenience option; otherwise read plaintext bytes from stdin")
    encode.add_argument("--message-file", help="read plaintext bytes from this file")
    encode.add_argument("--sequence", type=int, default=0)
    encode.add_argument("--no-length-padding", action="store_true", help="omit randomized cover-length padding")
    encode.add_argument("--output")

    decode = sub.add_parser("decode", help="decode and decrypt a transcript")
    decode.add_argument("--pack", required=True)
    decode.add_argument("--key", help="unsafe convenience option; omitted values are prompted securely")
    decode.add_argument("--input")

    inspect = sub.add_parser("inspect", help="show pack capacity")
    inspect.add_argument("--pack", required=True)

    train = sub.add_parser("train-model", help="compile a compact statistical carrier model")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", help="UTF-8 text file with one cover sentence/message per line")
    source.add_argument("--pack", help="generate a deterministic training corpus from a dialogue pack")
    train.add_argument("--output", required=True)
    train.add_argument("--name", default="custom-cover-model")
    train.add_argument("--order", type=int, default=2)
    train.add_argument("--samples-per-turn", type=int, default=128)

    train_topics = sub.add_parser("train-topic-model", help="compile a topic-conditioned compact model")
    train_topics.add_argument("--corpus", required=True)
    train_topics.add_argument("--output", required=True)
    train_topics.add_argument("--name", default="topic-cover-model")
    train_topics.add_argument("--order", type=int, default=4)

    inspect_model = sub.add_parser("inspect-model", help="show compact model statistics")
    inspect_model.add_argument("--model", required=True)

    encode_v2_parser = sub.add_parser("encode-v2", help="compress, encrypt, and encode one V2 logical message")
    encode_v2_parser.add_argument("--carrier", required=True, help="dialogue pack or statistical model JSON")
    encode_v2_parser.add_argument("--profile", choices=PROFILES, default="secure")
    encode_v2_parser.add_argument("--allow-unsafe-dense", action="store_true", help="required for the unauthenticated dense profile")
    encode_v2_parser.add_argument("--key", help="unsafe convenience option; omitted values are prompted securely")
    encode_v2_parser.add_argument("--message", help="unsafe convenience option; otherwise read bytes from stdin")
    encode_v2_parser.add_argument("--message-file")
    encode_v2_parser.add_argument("--sequence", type=int)
    encode_v2_parser.add_argument(
        "--no-interleave", action="store_false", dest="interleave", default=True,
        help="disable the default keyed permutation of sealed bytes",
    )
    encode_v2_parser.add_argument("--plain-output", action="store_true", help="emit only blank-line-separated cover messages")
    encode_v2_parser.add_argument("--output")

    decode_v2_parser = sub.add_parser("decode-v2", help="decode one V2 logical message")
    decode_v2_parser.add_argument("--carrier", required=True)
    decode_v2_parser.add_argument("--profile", choices=PROFILES, default="secure")
    decode_v2_parser.add_argument("--allow-unsafe-dense", action="store_true", help="required for the unauthenticated dense profile")
    decode_v2_parser.add_argument("--key", help="unsafe convenience option; omitted values are prompted securely")
    decode_v2_parser.add_argument("--sequence", type=int)
    decode_v2_parser.add_argument(
        "--no-interleave", action="store_false", dest="interleave", default=True,
        help="decode plain input that was encoded without byte interleaving",
    )
    decode_v2_parser.add_argument("--plain-input", action="store_true", help="read blank-line-separated cover messages")
    decode_v2_parser.add_argument("--input")

    benchmark = sub.add_parser("benchmark", help="compare V2 no-LLM and statistical-model carriers")
    benchmark.add_argument("--pack", required=True)
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--key", help="unsafe convenience option; omitted values are prompted securely")
    benchmark.add_argument(
        "--no-interleave", action="store_false", dest="interleave", default=True,
        help="benchmark ordered sealed bytes instead of the default interleaving",
    )
    benchmark.add_argument("--output")

    compression_benchmark = sub.add_parser("compression-benchmark", help="compare every lossless compression candidate")
    compression_benchmark.add_argument("--output")

    args = parser.parse_args()
    if args.command == "inspect":
        pack = _pack(args.pack)
        print(json.dumps({
            "pack_id": pack.pack_id,
            "arcs": len(pack.arcs),
            "turns_per_arc": pack.turn_count,
            "bits_per_arc": [sum(turn.bits for turn in arc.turns) for arc in pack.arcs],
            "bits_per_turn": [[turn.bits for turn in arc.turns] for arc in pack.arcs],
        }, indent=2))
        return 0
    if args.command == "inspect-model":
        model = _carrier(args.model)
        if isinstance(model, DialoguePack):
            raise ValueError("inspect-model requires a statistical model")
        print(json.dumps({
            "model_id": model.model_id,
            "name": model.name,
            "order": model.order,
            "contexts": model.context_count,
            "vocabulary": model.vocabulary_size,
            "topics": len(model.topics) if isinstance(model, TopicMarkovModel) else 1,
        }, indent=2))
        return 0
    if args.command == "train-topic-model":
        if args.order < 1 or args.order > 5:
            raise ValueError("model order must be between 1 and 5")
        with open(args.corpus, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        model = TopicMarkovModel.train_default_topics(lines, order=args.order, name=args.name)
        model.save(args.output)
        print(json.dumps({
            "model_id": model.model_id,
            "topics": len(model.topics),
            "contexts": model.context_count,
            "output": args.output,
        }, indent=2))
        return 0
    if args.command == "train-model":
        if args.order < 1 or args.order > 5:
            raise ValueError("model order must be between 1 and 5")
        if args.samples_per_turn < 1 or args.samples_per_turn > 4096:
            raise ValueError("samples-per-turn must be between 1 and 4096")
        if args.corpus:
            with open(args.corpus, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
            salt = None
        else:
            pack = _pack(args.pack)
            authoring_keys = derive_keys("offline model corpus phrase", pack.salt)
            rng = random.Random(17)
            lines = []
            for arc_index, arc in enumerate(pack.arcs):
                for turn_index, turn in enumerate(arc.turns):
                    for _ in range(args.samples_per_turn):
                        lines.append(pack.render(authoring_keys, arc_index, turn_index, rng.randrange(2**turn.bits)))
            salt = pack.salt
        model = HuffmanMarkovModel.train(lines, order=args.order, name=args.name, salt=salt)
        model.save(args.output)
        print(json.dumps({"model_id": model.model_id, "contexts": len(model.contexts), "output": args.output}, indent=2))
        return 0
    if args.command == "benchmark":
        model = _carrier(args.model)
        if isinstance(model, DialoguePack):
            raise ValueError("benchmark --model requires a statistical model")
        report = compare(_pack(args.pack), model, _key(args.key), interleave=args.interleave)
        output = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if args.output:
            with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(output)
        else:
            sys.stdout.write(output)
        return 0
    if args.command == "compression-benchmark":
        output = json.dumps(compression_compare(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if args.output:
            with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(output)
        else:
            sys.stdout.write(output)
        return 0
    if args.command == "encode-v2":
        if args.profile == "dense" and not args.allow_unsafe_dense:
            raise ValueError("dense profile is unauthenticated; pass --allow-unsafe-dense explicitly")
        if args.message is not None and args.message_file is not None:
            raise ValueError("use only one of --message and --message-file")
        if args.message_file:
            with open(args.message_file, "rb") as handle:
                plaintext = _read_limited(handle, MAX_PLAINTEXT_INPUT)
        elif args.message is not None:
            plaintext = args.message.encode("utf-8")
        else:
            stream = getattr(sys.stdin, "buffer", sys.stdin)
            plaintext = _read_limited(stream, MAX_PLAINTEXT_INPUT)
        if args.sequence is None and args.plain_output:
            raise ValueError("plain V2 output requires an explicit unique --sequence")
        sequence = args.sequence if args.sequence is not None else secrets.randbits(64)
        transcript = encode_v2(
            _carrier(args.carrier),
            _key(args.key),
            plaintext,
            profile=args.profile,
            sequence=sequence,
            interleave=args.interleave,
        )
        if args.plain_output:
            output = "\n\n".join(message["text"] for message in transcript["messages"]) + "\n"
        else:
            output = json.dumps(transcript, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if args.output:
            with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(output)
        else:
            sys.stdout.write(output)
        return 0
    if args.command == "decode-v2":
        if args.profile == "dense" and not args.allow_unsafe_dense:
            raise ValueError("dense profile is unauthenticated; pass --allow-unsafe-dense explicitly")
        carrier = _carrier(args.carrier)
        if args.plain_input and args.sequence is None:
            raise ValueError("plain V2 input requires the sender's --sequence")
        transcript = (
            _plain_transcript(args.input, carrier, args.profile, args.sequence, args.interleave)
            if args.plain_input
            else _read_transcript(args.input)
        )
        sys.stdout.buffer.write(decode_v2(carrier, _key(args.key), transcript))
        return 0
    if args.command == "encode":
        pack = _pack(args.pack)
        if args.message is not None and args.message_file is not None:
            raise ValueError("use only one of --message and --message-file")
        if args.message_file:
            with open(args.message_file, "rb") as handle:
                plaintext = _read_limited(handle, MAX_PLAINTEXT_INPUT)
        elif args.message is not None:
            plaintext = args.message.encode("utf-8")
        else:
            stream = getattr(sys.stdin, "buffer", sys.stdin)
            plaintext = _read_limited(stream, MAX_PLAINTEXT_INPUT)
        transcript = encode_message(
            pack,
            _key(args.key),
            plaintext,
            args.sequence,
            cover_bucket=0 if args.no_length_padding else 8,
        )
        output = transcript_json(transcript)
        if args.output:
            with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(output)
        else:
            sys.stdout.write(output)
        return 0
    pack = _pack(args.pack)
    plaintext = decode_message(pack, _key(args.key), _read_transcript(args.input))
    sys.stdout.buffer.write(plaintext)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
