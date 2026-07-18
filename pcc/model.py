"""Compact deterministic word-Markov carrier with keyed Huffman coding."""

from __future__ import annotations

import base64
import hashlib
import heapq
import hmac
import json
import re
import secrets
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .errors import CapacityExceeded, InvalidPack, PackMismatch
from .limits import MAX_TRANSCRIPT_MESSAGES

BOS = "<BOS>"
END = "."
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
OUTPUT_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|\.", re.UNICODE)


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _tokens(line: str) -> list[str]:
    return WORD_RE.findall(line)


def _sentences(line: str) -> list[list[str]]:
    parts = re.split(r"(?<=[.!?])\s+", line.strip())
    sentences = [_tokens(part) for part in parts]
    return [sentence for sentence in sentences if sentence]


def _render_tokens(tokens: list[str]) -> str:
    output = ""
    for token in tokens:
        if token == END:
            output = output.rstrip() + END
        else:
            if output:
                output += " "
            output += token
    return output


def _huffman_lengths(counts: dict[str, int]) -> dict[str, int]:
    if len(counts) == 1:
        return {next(iter(counts)): 0}
    heap = []
    serial = 0
    for token, count in sorted(counts.items()):
        heapq.heappush(heap, (count, serial, token))
        serial += 1
    while len(heap) > 1:
        first_count, _, first = heapq.heappop(heap)
        second_count, _, second = heapq.heappop(heap)
        heapq.heappush(heap, (first_count + second_count, serial, (first, second)))
        serial += 1
    lengths: dict[str, int] = {}

    def walk(node, depth: int) -> None:
        if isinstance(node, str):
            lengths[node] = depth
            return
        walk(node[0], depth + 1)
        walk(node[1], depth + 1)

    walk(heap[0][2], 0)
    return lengths


@dataclass(frozen=True)
class Codebook:
    token_to_code: dict[str, str]
    code_to_token: dict[str, str]
    max_length: int


class HuffmanMarkovModel:
    """A small shared statistical model, not a neural runtime.

    Random-looking encrypted bits walk a context-dependent Huffman tree. The
    same model and mapping key recover each token's codeword exactly.
    """

    def __init__(self, raw: dict, model_id: str, name: str, salt: bytes, order: int, contexts: dict[tuple[str, ...], dict[str, int]]):
        self.raw = raw
        self.model_id = model_id
        self.name = name
        self.salt = salt
        self.order = order
        self.contexts = contexts

    @classmethod
    def train(
        cls,
        lines: Iterable[str],
        *,
        order: int = 2,
        name: str = "custom-cover-model",
        salt: bytes | None = None,
    ) -> "HuffmanMarkovModel":
        if order < 1 or order > 5:
            raise InvalidPack("model order must be between 1 and 5")
        counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        accepted = 0
        for line in lines:
            line_sentences = _sentences(line)
            if not line_sentences:
                continue
            accepted += 1
            for words in line_sentences:
                history = [BOS]
                for token in words + [END]:
                    width = min(order, len(history))
                    for context_width in range(width, -1, -1):
                        context = tuple(history[-context_width:]) if context_width else ()
                        counts[context][token] += 1
                    if token == END:
                        history = [BOS]
                    else:
                        history.append(token)
                        history = history[-order:]
        if accepted < 16:
            raise InvalidPack("model training needs at least 16 non-empty lines")
        if (BOS,) not in counts or len(counts[(BOS,)]) < 2:
            raise InvalidPack("model corpus needs at least two sentence openings")
        source = {
            "schema": 1,
            "kind": "markov",
            "name": name,
            "salt": base64.b64encode(salt or secrets.token_bytes(16)).decode("ascii"),
            "order": order,
            "contexts": [
                {"context": list(context), "counts": [[token, count] for token, count in sorted(counter.items())]}
                for context, counter in sorted(counts.items())
            ],
        }
        return cls.from_dict(source)

    @classmethod
    def load(cls, path: str | Path) -> "HuffmanMarkovModel":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_dict(cls, source: dict) -> "HuffmanMarkovModel":
        if not isinstance(source, dict) or source.get("schema") != 1 or source.get("kind") != "markov":
            raise InvalidPack("unsupported statistical model")
        canonical = dict(source)
        declared_id = canonical.pop("model_id", None)
        try:
            raw = _canonical(canonical)
        except UnicodeEncodeError as exc:
            raise InvalidPack("model contains invalid Unicode") from exc
        model_id = hashlib.sha256(raw).hexdigest()
        if declared_id is not None and declared_id != model_id:
            raise InvalidPack("model_id does not match model contents")
        try:
            salt = base64.b64decode(source["salt"], validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidPack("model salt must be base64") from exc
        order = source.get("order")
        if len(salt) < 16 or not isinstance(order, int) or not 1 <= order <= 5:
            raise InvalidPack("model salt or order is invalid")
        raw_contexts = source.get("contexts")
        if not isinstance(raw_contexts, list) or not raw_contexts:
            raise InvalidPack("model contexts must be a non-empty array")
        contexts: dict[tuple[str, ...], dict[str, int]] = {}
        for item in raw_contexts:
            if not isinstance(item, dict) or not isinstance(item.get("context"), list) or not isinstance(item.get("counts"), list):
                raise InvalidPack("model context is malformed")
            context = tuple(item["context"])
            if any(not isinstance(token, str) for token in context) or len(context) > order:
                raise InvalidPack("model context is invalid")
            token_counts: dict[str, int] = {}
            for pair in item["counts"]:
                if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                    raise InvalidPack("model token count is malformed")
                token, count = pair
                if token == BOS or (token != END and not WORD_RE.fullmatch(token)):
                    raise InvalidPack("model token is not transport-safe")
                if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                    raise InvalidPack("model token count must be positive")
                if token in token_counts:
                    raise InvalidPack("model context contains duplicate tokens")
                token_counts[token] = count
            contexts[context] = token_counts
        if (BOS,) not in contexts or () not in contexts:
            raise InvalidPack("model is missing start or fallback context")
        return cls(source, model_id, source.get("name", "model"), salt, order, contexts)

    def save(self, path: str | Path) -> None:
        source = dict(self.raw)
        source["model_id"] = self.model_id
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(source, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")

    @property
    def context_count(self) -> int:
        return len(self.contexts)

    @property
    def vocabulary_size(self) -> int:
        return len({token for counts in self.contexts.values() for token in counts})

    def _counts(self, history: list[str]) -> tuple[tuple[str, ...], dict[str, int]]:
        for width in range(min(self.order, len(history)), -1, -1):
            context = tuple(history[-width:]) if width else ()
            if context in self.contexts:
                return context, self.contexts[context]
        raise PackMismatch("statistical model has no usable context")

    def _codebook(self, history: list[str], mapping_key: bytes, sequence: int, cache: dict) -> Codebook:
        context, counts = self._counts(history)
        cached = cache.get(context)
        if cached is not None:
            return cached
        lengths = _huffman_lengths(counts)
        canonical: dict[int, list[str]] = defaultdict(list)
        code = 0
        previous_length = 0
        for length, token in sorted((length, token) for token, length in lengths.items()):
            code <<= length - previous_length
            canonical[length].append(format(code, f"0{length}b") if length else "")
            code += 1
            previous_length = length
        token_to_code: dict[str, str] = {}
        context_bytes = "\x1f".join(context).encode("utf-8")
        for length, codes in canonical.items():
            tokens = [token for token, token_length in lengths.items() if token_length == length]
            codes.sort()
            if END in tokens:
                token_to_code[END] = codes.pop(0)
                tokens.remove(END)
            tokens.sort(key=lambda token: hmac.new(
                mapping_key,
                b"pcc/v2/model-code\x00" + bytes.fromhex(self.model_id) + struct.pack(">Q", sequence)
                + context_bytes + b"\x00" + token.encode("utf-8"),
                hashlib.sha256,
            ).digest())
            for token, token_code in zip(tokens, codes):
                token_to_code[token] = token_code
        code_to_token = {token_code: token for token, token_code in token_to_code.items()}
        book = Codebook(token_to_code, code_to_token, max(lengths.values()))
        cache[context] = book
        return book

    @staticmethod
    def _advance(history: list[str], token: str, order: int) -> list[str]:
        if token == END:
            return [BOS]
        history = history + [token]
        return history[-order:]

    def encode_bits(self, bits: list[int], mapping_key: bytes, sequence: int) -> list[str]:
        if not bits:
            raise CapacityExceeded("model carrier received no bits")
        history = [BOS]
        cache: dict = {}
        offset = 0
        current: list[str] = []
        messages: list[str] = []
        steps = 0
        max_steps = max(1024, len(bits) * 4)
        while offset < len(bits):
            steps += 1
            if steps > max_steps or len(messages) >= MAX_TRANSCRIPT_MESSAGES:
                raise CapacityExceeded("statistical model could not encode the frame")
            book = self._codebook(history, mapping_key, sequence, cache)
            if book.max_length == 0:
                token = book.code_to_token[""]
            else:
                prefix = ""
                while prefix not in book.code_to_token:
                    bit = bits[offset] if offset < len(bits) else 0
                    if offset < len(bits):
                        offset += 1
                    prefix += str(bit)
                    if len(prefix) > book.max_length:
                        raise CapacityExceeded("statistical model code tree is incomplete")
                token = book.code_to_token[prefix]
            if token == END:
                if current:
                    messages.append(" ".join(current) + ".")
                    current = []
            else:
                current.append(token)
            history = self._advance(history, token, self.order)

        # Finish through valid model transitions. These tokens occur after the
        # framed bit count, so the decoder ignores their codewords.
        for _ in range(64):
            if not current:
                break
            context, counts = self._counts(history)
            del context
            token = END if END in counts else max(counts, key=lambda candidate: (counts[candidate], candidate))
            if token == END:
                messages.append(" ".join(current) + ".")
                current = []
            else:
                current.append(token)
            history = self._advance(history, token, self.order)
        if current:
            messages.append(" ".join(current) + ".")
        if not messages:
            raise CapacityExceeded("statistical model produced no carrier text")
        return messages

    def decode_bits(self, messages: list[str], mapping_key: bytes, sequence: int) -> list[int]:
        history = [BOS]
        cache: dict = {}
        bits: list[int] = []
        for message in messages:
            if not isinstance(message, str) or not message.strip():
                raise PackMismatch("statistical carrier contains an empty message")
            tokens = OUTPUT_RE.findall(message)
            if not tokens or _render_tokens(tokens) != message:
                raise PackMismatch("statistical carrier message is malformed")
            for token in tokens:
                book = self._codebook(history, mapping_key, sequence, cache)
                code = book.token_to_code.get(token)
                if code is None:
                    raise PackMismatch("text does not belong to the statistical model")
                bits.extend(int(bit) for bit in code)
                history = self._advance(history, token, self.order)
        return bits


class TopicMarkovModel:
    """A compact bundle that keeps one encoded message on one mundane topic."""

    def __init__(self, raw: dict, model_id: str, name: str, salt: bytes, topics: tuple[tuple[str, HuffmanMarkovModel], ...]):
        self.raw = raw
        self.model_id = model_id
        self.name = name
        self.salt = salt
        self.topics = topics
        self.order = max(model.order for _, model in topics)

    @classmethod
    def train(
        cls,
        topic_lines: dict[str, list[str]],
        *,
        order: int = 4,
        name: str = "topic-cover-model",
        salt: bytes | None = None,
    ) -> "TopicMarkovModel":
        shared_salt = salt or secrets.token_bytes(16)
        topics = []
        for topic, lines in sorted(topic_lines.items()):
            if len(lines) < 16:
                continue
            model = HuffmanMarkovModel.train(lines, order=order, name=f"{name}:{topic}", salt=shared_salt)
            topics.append({"name": topic, "model": model.raw})
        if len(topics) < 2:
            raise InvalidPack("topic model training needs at least two topics with 16 messages each")
        source = {
            "schema": 1,
            "kind": "topic-markov",
            "name": name,
            "salt": base64.b64encode(shared_salt).decode("ascii"),
            "topics": topics,
        }
        return cls.from_dict(source)

    @classmethod
    def train_default_topics(
        cls,
        lines: Iterable[str],
        *,
        order: int = 4,
        name: str = "topic-cover-model",
    ) -> "TopicMarkovModel":
        keywords = {
            "home": ("kitchen", "dish", "laundry", "washer", "fridge", "furnace", "house", "home", "clean", "door", "sink", "garden", "vacuum", "room", "porch"),
            "food": ("dinner", "lunch", "breakfast", "coffee", "tea", "cafe", "restaurant", "cook", "food", "meal", "soup", "taco", "pizza", "bread", "grocery"),
            "work": ("work", "meeting", "report", "project", "office", "deadline", "client", "email", "team", "manager", "commute", "train", "bus"),
            "plans": ("weekend", "movie", "book", "trip", "travel", "saturday", "sunday", "tomorrow", "tonight", "tickets", "plan", "walk", "hike"),
        }
        grouped: dict[str, list[str]] = {name: [] for name in keywords}
        grouped["everyday"] = []
        for line in lines:
            normalized = line.casefold()
            scores = {topic: sum(word in normalized for word in words) for topic, words in keywords.items()}
            topic, score = max(scores.items(), key=lambda item: (item[1], item[0]))
            grouped[topic if score else "everyday"].append(line)
        return cls.train(grouped, order=order, name=name)

    @classmethod
    def load(cls, path: str | Path) -> "TopicMarkovModel":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_dict(cls, source: dict) -> "TopicMarkovModel":
        if not isinstance(source, dict) or source.get("schema") != 1 or source.get("kind") != "topic-markov":
            raise InvalidPack("unsupported topic model")
        canonical = dict(source)
        declared_id = canonical.pop("model_id", None)
        model_id = hashlib.sha256(_canonical(canonical)).hexdigest()
        if declared_id is not None and declared_id != model_id:
            raise InvalidPack("model_id does not match topic model contents")
        try:
            salt = base64.b64decode(source["salt"], validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidPack("topic model salt must be base64") from exc
        raw_topics = source.get("topics")
        if len(salt) < 16 or not isinstance(raw_topics, list) or len(raw_topics) < 2:
            raise InvalidPack("topic model contents are invalid")
        topics = []
        for item in raw_topics:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("model"), dict):
                raise InvalidPack("topic model entry is malformed")
            model = HuffmanMarkovModel.from_dict(item["model"])
            if model.salt != salt:
                raise InvalidPack("topic model salts do not match")
            topics.append((item["name"], model))
        return cls(source, model_id, source.get("name", "topic-model"), salt, tuple(topics))

    def save(self, path: str | Path) -> None:
        source = dict(self.raw)
        source["model_id"] = self.model_id
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(source, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")

    @property
    def context_count(self) -> int:
        return sum(model.context_count for _, model in self.topics)

    @property
    def vocabulary_size(self) -> int:
        return len({token for _, model in self.topics for counts in model.contexts.values() for token in counts})

    def selected_topic(self, mapping_key: bytes, sequence: int) -> tuple[str, HuffmanMarkovModel]:
        digest = hmac.new(
            mapping_key,
            b"pcc/v2/topic\x00" + bytes.fromhex(self.model_id) + struct.pack(">Q", sequence),
            hashlib.sha256,
        ).digest()
        return self.topics[int.from_bytes(digest[:8], "big") % len(self.topics)]

    def encode_bits(self, bits: list[int], mapping_key: bytes, sequence: int) -> list[str]:
        _, model = self.selected_topic(mapping_key, sequence)
        return model.encode_bits(bits, mapping_key, sequence)

    def decode_bits(self, messages: list[str], mapping_key: bytes, sequence: int) -> list[int]:
        _, model = self.selected_topic(mapping_key, sequence)
        return model.decode_bits(messages, mapping_key, sequence)
