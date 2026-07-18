"""Optional ONNX language-model carrier.

The model is a carrier distribution only. It never receives plaintext; it is
driven by already encrypted frame bits and a keyed permutation of candidate
tokens.
"""

from __future__ import annotations

import base64
import heapq
import hashlib
import hmac
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .errors import CapacityExceeded, InvalidPack, PackMismatch
from .framing import frame_target_from_bits
from .limits import MAX_TRANSCRIPT_MESSAGES


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OnnxLanguageCarrier:
    """Deterministic top-k token carrier backed by a local ONNX model."""

    def __init__(self, manifest: dict, model_id: str, salt: bytes, root: Path):
        self.raw = manifest
        self.model_id = model_id
        self.name = manifest.get("name", "onnx-language-carrier")
        self.salt = salt
        self.root = root
        self.prompt = manifest["prompt"]
        self.coding = manifest.get("coding", "fixed")
        self.top_k = manifest.get("top_k")
        self.candidate_count = manifest.get("candidate_count", self.top_k)
        self.bits_per_token = self.top_k.bit_length() - 1 if self.coding == "fixed" else None
        self.quality_attempts = int(manifest.get("quality_attempts", 1))
        self.order = 0
        self.context_count = 0
        self.tokenizer = Tokenizer.from_file(str(root / manifest["tokenizer_file"]))
        self.vocabulary_size = self.tokenizer.get_vocab_size(with_added_tokens=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(manifest.get("intra_op_threads", 1))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(root / manifest["model_file"]),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.past_names = sorted(name for name in self.input_names if name.startswith("past_"))
        self.present_names = set(name for name in self.output_names if name.startswith("present"))
        if not self.past_names or len(self.past_names) != len(self.present_names):
            raise InvalidPack("ONNX model does not expose compatible KV-cache inputs")
        self.layer_count = len(self.past_names) // 2
        self.input_specs = {item.name: item for item in self.session.get_inputs()}
        self.present_for_past = {}
        for past_name in self.past_names:
            if past_name.startswith("past_key_values."):
                present_name = "present." + past_name[len("past_key_values.") :]
            else:
                present_name = past_name.replace("past_", "present_", 1)
            if present_name not in self.present_names:
                raise InvalidPack(f"ONNX state output {present_name!r} is missing")
            self.present_for_past[past_name] = present_name
        self.allowed_ids = self._allowed_token_ids()

    @classmethod
    def load(cls, manifest_path: str | Path) -> "OnnxLanguageCarrier":
        manifest_path = Path(manifest_path)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema") != 1 or manifest.get("kind") != "neural-onnx":
            raise InvalidPack("unsupported neural carrier manifest")
        root = manifest_path.parent
        model_path = root / manifest["model_file"]
        tokenizer_path = root / manifest["tokenizer_file"]
        data_path = root / manifest["model_data_file"] if manifest.get("model_data_file") else None
        hashes_match = _sha256(model_path) == manifest["model_sha256"] and _sha256(tokenizer_path) == manifest["tokenizer_sha256"]
        if data_path is not None:
            hashes_match = hashes_match and _sha256(data_path) == manifest["model_data_sha256"]
        if not hashes_match:
            raise InvalidPack("neural carrier artifact hash does not match its manifest")
        canonical = dict(manifest)
        declared_id = canonical.pop("carrier_id", None)
        carrier_id = hashlib.sha256(_canonical(canonical)).hexdigest()
        if declared_id is not None and declared_id != carrier_id:
            raise InvalidPack("neural carrier ID does not match its manifest")
        try:
            salt = base64.b64decode(manifest["salt"], validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidPack("neural carrier salt is invalid") from exc
        coding = manifest.get("coding", "fixed")
        top_k = manifest.get("top_k")
        candidate_count = manifest.get("candidate_count", top_k)
        valid_coding = (
            coding == "fixed" and isinstance(top_k, int) and top_k in (2, 4, 8, 16, 32)
        ) or (
            coding == "huffman" and isinstance(candidate_count, int) and candidate_count in (8, 16, 32, 64)
        )
        if len(salt) < 16 or not valid_coding:
            raise InvalidPack("neural carrier parameters are invalid")
        quality_attempts = manifest.get("quality_attempts", 1)
        if not isinstance(quality_attempts, int) or not 1 <= quality_attempts <= 16:
            raise InvalidPack("neural carrier quality-attempt count is invalid")
        return cls(manifest, carrier_id, salt, root)

    def _allowed_token_ids(self) -> list[int]:
        vocabulary_size = self.tokenizer.get_vocab_size(with_added_tokens=True)
        allowed = []
        for token_id in range(vocabulary_size):
            try:
                text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                continue
            if not text or not text.strip() or any(ord(char) < 32 for char in text):
                continue
            if any(char in text for char in ('"', '`', '<', '>', '#', '*', '“', '”')):
                continue
            allowed.append(token_id)
        return allowed

    def _empty_past(self) -> dict[str, np.ndarray]:
        states = {}
        for name in self.past_names:
            shape = tuple(
                1 if isinstance(dimension, str) and "batch" in dimension else 0 if isinstance(dimension, str) else int(dimension)
                for dimension in self.input_specs[name].shape
            )
            states[name] = np.zeros(shape, dtype=np.float32)
        return states

    def _present_states(self, values: list[np.ndarray]) -> dict[str, np.ndarray]:
        by_name = {name: values[index] for index, name in enumerate(self.output_names)}
        return {past: by_name[present] for past, present in self.present_for_past.items()}

    def _start(self):
        prompt_ids = self.tokenizer.encode(self.prompt, add_special_tokens=False).ids
        if not prompt_ids:
            raise InvalidPack("neural carrier prompt tokenizes to nothing")
        ids = np.asarray([prompt_ids], dtype=np.int64)
        inputs = {
            "input_ids": ids,
            "attention_mask": np.ones((1, len(prompt_ids)), dtype=np.int64),
        }
        if "num_logits_to_keep" in self.input_names:
            inputs["num_logits_to_keep"] = np.asarray(1, dtype=np.int64)
        if "position_ids" in self.input_names:
            inputs["position_ids"] = np.arange(len(prompt_ids), dtype=np.int64)[None, :]
        inputs.update(self._empty_past())
        values = self.session.run(None, inputs)
        logits = values[0][0, -1].astype(np.float64)
        presents = self._present_states(values)
        return presents, logits, len(prompt_ids)

    def _step(self, past: dict[str, np.ndarray], token_id: int, logits_length: int):
        inputs = {
            "input_ids": np.asarray([[token_id]], dtype=np.int64),
            "attention_mask": np.ones((1, logits_length + 1), dtype=np.int64),
        }
        if "num_logits_to_keep" in self.input_names:
            inputs["num_logits_to_keep"] = np.asarray(1, dtype=np.int64)
        if "position_ids" in self.input_names:
            inputs["position_ids"] = np.asarray([[logits_length]], dtype=np.int64)
        inputs.update(past)
        values = self.session.run(None, inputs)
        logits = values[0][0, -1].astype(np.float64)
        presents = self._present_states(values)
        return presents, logits, logits_length + 1

    def _ranked(self, logits: np.ndarray, count: int) -> tuple[np.ndarray, list[int]]:
        quantized = np.rint(logits * 256.0).astype(np.int64)
        order = sorted(self.allowed_ids, key=lambda token_id: (-int(quantized[token_id]), token_id))
        candidates = order[:count]
        if len(candidates) != count:
            raise CapacityExceeded("neural model has too few usable token candidates")
        return quantized, candidates

    def _candidates(self, logits: np.ndarray, mapping_key: bytes, sequence: int, position: int, attempt: int) -> list[int]:
        _, candidates = self._ranked(logits, self.top_k)
        prefix = (
            b"pcc/v2/neural-token\x00" + bytes.fromhex(self.model_id)
            + struct.pack(">QQQ", sequence, position, attempt)
        )
        candidates.sort(key=lambda token_id: hmac.new(mapping_key, prefix + struct.pack(">I", token_id), hashlib.sha256).digest())
        return candidates

    @staticmethod
    def _huffman_lengths(weights: dict[int, int]) -> dict[int, int]:
        heap = []
        serial = 0
        for token_id, weight in sorted(weights.items()):
            heapq.heappush(heap, (weight, serial, token_id))
            serial += 1
        while len(heap) > 1:
            first_weight, _, first = heapq.heappop(heap)
            second_weight, _, second = heapq.heappop(heap)
            heapq.heappush(heap, (first_weight + second_weight, serial, (first, second)))
            serial += 1
        lengths: dict[int, int] = {}

        def walk(node, depth: int) -> None:
            if isinstance(node, int):
                lengths[node] = depth
                return
            walk(node[0], depth + 1)
            walk(node[1], depth + 1)

        walk(heap[0][2], 0)
        return lengths

    def _huffman_book(self, logits: np.ndarray, mapping_key: bytes, sequence: int, position: int, attempt: int):
        quantized, candidates = self._ranked(logits, self.candidate_count)
        peak = max(int(quantized[token_id]) for token_id in candidates)
        weights = {
            token_id: max(1, int(round(math.exp((int(quantized[token_id]) - peak) / 256.0) * 1_000_000)))
            for token_id in candidates
        }
        lengths = self._huffman_lengths(weights)
        codes_by_length: dict[int, list[str]] = defaultdict(list)
        code = 0
        previous_length = 0
        for length, token_id in sorted((length, token_id) for token_id, length in lengths.items()):
            code <<= length - previous_length
            codes_by_length[length].append(format(code, f"0{length}b"))
            code += 1
            previous_length = length
        prefix = (
            b"pcc/v2/neural-huffman\x00" + bytes.fromhex(self.model_id)
            + struct.pack(">QQQ", sequence, position, attempt)
        )
        token_to_code: dict[int, str] = {}
        for length, codes in codes_by_length.items():
            tokens = [token_id for token_id, token_length in lengths.items() if token_length == length]
            tokens.sort(key=lambda token_id: hmac.new(
                mapping_key, prefix + struct.pack(">I", token_id), hashlib.sha256
            ).digest())
            for token_id, token_code in zip(tokens, sorted(codes)):
                token_to_code[token_id] = token_code
        return token_to_code, {token_code: token_id for token_id, token_code in token_to_code.items()}, max(lengths.values())

    def _greedy(self, logits: np.ndarray) -> int:
        quantized = np.rint(logits * 256.0).astype(np.int64)
        return max(self.allowed_ids, key=lambda token_id: (int(quantized[token_id]), -token_id))

    def encode_bits(self, bits: list[int], mapping_key: bytes, sequence: int, attempt: int = 0) -> list[str]:
        if not bits:
            raise CapacityExceeded("neural carrier received no bits")
        past, logits, context_length = self._start()
        token_ids: list[int] = []
        bit_offset = 0
        position = 0
        while bit_offset < len(bits):
            if self.coding == "fixed":
                candidates = self._candidates(logits, mapping_key, sequence, position, attempt)
                label = 0
                for _ in range(self.bits_per_token):
                    label = (label << 1) | (bits[bit_offset] if bit_offset < len(bits) else 0)
                    bit_offset += 1
                token_id = candidates[label]
            else:
                _, code_to_token, max_length = self._huffman_book(logits, mapping_key, sequence, position, attempt)
                prefix = ""
                while prefix not in code_to_token:
                    prefix += str(bits[bit_offset] if bit_offset < len(bits) else 0)
                    bit_offset += 1
                    if len(prefix) > max_length:
                        raise CapacityExceeded("neural Huffman tree is incomplete")
                token_id = code_to_token[prefix]
            token_ids.append(token_id)
            past, logits, context_length = self._step(past, token_id, context_length)
            position += 1
            if position > max(512, len(bits) * 2):
                raise CapacityExceeded("neural carrier exceeded token limit")
        # Greedy continuation closes the carrier at a natural sentence boundary.
        for _ in range(32):
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            if len(text) > 40 and re.search(r"[.!?]['\"]?$", text.rstrip()):
                break
            token_id = self._greedy(logits)
            token_ids.append(token_id)
            past, logits, context_length = self._step(past, token_id, context_length)
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if not text:
            raise CapacityExceeded("neural carrier produced no visible text")
        if self.tokenizer.encode(text, add_special_tokens=False).ids != token_ids:
            raise CapacityExceeded("neural carrier tokenization is not reversible")
        return [text]

    def accepts_cover(self, text: str) -> bool:
        lowered = text.casefold()
        banned = (
            "text assistant", "text-only", "writing about", "the prompt", "these instructions",
            "as an ai", "language model", "i am an assistant", "this conversation",
        )
        if any(phrase in lowered for phrase in banned):
            return False
        if len(text.split()) < 8 or not re.search(r"[.!?]['\"]?$", text.rstrip()):
            return False
        words = [word.casefold() for word in re.findall(r"[A-Za-z][A-Za-z']+", text)]
        if len(words) < 8:
            return False
        for index in range(len(words) - 3):
            if words[index : index + 3] == words[index + 3 : index + 6]:
                return False
        return True

    def decode_bits(self, messages: list[str], mapping_key: bytes, sequence: int, attempt: int = 0) -> list[int]:
        if len(messages) != 1:
            raise PackMismatch("neural carrier expects one pasted generated message")
        token_ids = self.tokenizer.encode(messages[0], add_special_tokens=False).ids
        if not token_ids:
            raise PackMismatch("neural carrier message is empty")
        past, logits, context_length = self._start()
        bits: list[int] = []
        for position, token_id in enumerate(token_ids):
            frame_target = frame_target_from_bits(bits)
            if frame_target is not None and len(bits) >= frame_target[1]:
                break
            if self.coding == "fixed":
                candidates = self._candidates(logits, mapping_key, sequence, position, attempt)
                if token_id not in candidates:
                    raise PackMismatch("text token is outside the neural carrier candidate set")
                label = candidates.index(token_id)
                bits.extend((label >> bit) & 1 for bit in range(self.bits_per_token - 1, -1, -1))
            else:
                token_to_code, _, _ = self._huffman_book(logits, mapping_key, sequence, position, attempt)
                code = token_to_code.get(token_id)
                if code is None:
                    raise PackMismatch("text token is outside the neural Huffman candidate set")
                bits.extend(int(bit) for bit in code)
            past, logits, context_length = self._step(past, token_id, context_length)
        return bits
