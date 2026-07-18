"""Validation and deterministic interpretation of offline dialogue packs."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import string
import struct
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .crypto import DerivedKeys
from .errors import InvalidPack, PackMismatch
from .limits import MAX_CARRIER_TEXT_BYTES


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InvalidPack("pack contains an invalid Unicode surrogate") from exc


def _power_bits(value: int) -> int:
    if value < 2 or value & (value - 1):
        raise InvalidPack("every choice group must contain a power-of-two number of choices")
    return value.bit_length() - 1


def _has_control(text: str) -> bool:
    return any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in text)


@dataclass(frozen=True)
class CompiledSlot:
    name: str
    choices: tuple[str, ...]
    bits: int


@dataclass(frozen=True)
class CompiledTurn:
    role: str
    template: str
    slots: tuple[CompiledSlot, ...]
    bits: int
    parser: re.Pattern[str]


@dataclass(frozen=True)
class CompiledArc:
    name: str
    turns: tuple[CompiledTurn, ...]


class DialoguePack:
    """A deterministic, offline-authored conversation language.

    Each turn is a composition of semantically compatible clause groups. The
    model that authored a pack is not required at runtime. Every choice group
    is independently keyed and reversible.
    """

    def __init__(self, raw: dict[str, Any], pack_id: str, salt: bytes, arcs: tuple[CompiledArc, ...]):
        self.raw = raw
        self.pack_id = pack_id
        self.salt = salt
        self.arcs = arcs
        self.turn_count = len(arcs[0].turns)
        self.capacity_bits = sum(turn.bits for turn in arcs[0].turns)

    @classmethod
    def load(cls, path: str | Path) -> "DialoguePack":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_dict(cls, source: dict[str, Any]) -> "DialoguePack":
        if not isinstance(source, dict) or source.get("schema") != 1:
            raise InvalidPack("unsupported or missing pack schema")
        canonical_source = dict(source)
        declared_id = canonical_source.pop("pack_id", None)
        computed_id = hashlib.sha256(_canonical(canonical_source)).hexdigest()
        if declared_id is not None and declared_id != computed_id:
            raise InvalidPack("pack_id does not match canonical pack contents")
        try:
            salt = base64.b64decode(source["salt"], validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidPack("pack salt must be base64") from exc
        if len(salt) < 16:
            raise InvalidPack("pack salt must contain at least 16 bytes")
        vocabulary = source.get("vocabulary", {})
        if not isinstance(vocabulary, dict):
            raise InvalidPack("vocabulary must be an object")

        raw_arcs = source.get("arcs", [])
        if not isinstance(raw_arcs, list):
            raise InvalidPack("arcs must be an array")
        arcs = tuple(cls._compile_arc(arc, vocabulary) for arc in raw_arcs)
        if not arcs or not arcs[0].turns:
            raise InvalidPack("pack must contain non-empty arcs")
        turn_count = len(arcs[0].turns)
        role_schedule = tuple(turn.role for turn in arcs[0].turns)
        if any(first == second for first, second in zip(role_schedule, role_schedule[1:])):
            raise InvalidPack("adjacent turns must alternate roles")
        if role_schedule[0] == role_schedule[-1]:
            raise InvalidPack("roles must also alternate across arc boundaries")
        for arc in arcs:
            if len(arc.turns) != turn_count:
                raise InvalidPack("all arcs must contain the same number of turns")
            if tuple(turn.role for turn in arc.turns) != role_schedule:
                raise InvalidPack("all arcs must use the same role schedule")
        return cls(source, computed_id, salt, arcs)

    @staticmethod
    def _compile_arc(source: dict[str, Any], vocabulary: dict[str, Any]) -> CompiledArc:
        if not isinstance(source, dict) or not isinstance(source.get("turns"), list):
            raise InvalidPack("arc must contain a turns array")
        turns: list[CompiledTurn] = []
        for turn_source in source["turns"]:
            if not isinstance(turn_source, dict):
                raise InvalidPack("turn must be an object")
            role = turn_source.get("role")
            template = turn_source.get("template")
            slot_sources = turn_source.get("slots")
            if not isinstance(role, str) or not role.strip() or _has_control(role):
                raise InvalidPack("turn role must be non-empty")
            if not isinstance(template, str) or not template.strip():
                raise InvalidPack("turn template must be non-empty")
            if _has_control(template):
                raise InvalidPack("turn template contains a control character")
            if not isinstance(slot_sources, list) or not slot_sources:
                raise InvalidPack("turn must contain slots")

            try:
                parsed_fields = list(string.Formatter().parse(template))
            except ValueError as exc:
                raise InvalidPack("turn template is malformed") from exc
            fields = [field for _, field, _, _ in parsed_fields if field is not None]
            if any(not field.isidentifier() for field in fields):
                raise InvalidPack("templates may only contain simple named fields")
            if len(set(fields)) != len(fields):
                raise InvalidPack("templates may not repeat a field")
            if any(format_spec or conversion for _, _, format_spec, conversion in parsed_fields):
                raise InvalidPack("templates may not use format specs or conversions")

            slots: list[CompiledSlot] = []
            for slot_source in slot_sources:
                if not isinstance(slot_source, dict) or not isinstance(slot_source.get("name"), str):
                    raise InvalidPack("slot must contain a name")
                name = slot_source["name"]
                if "from" in slot_source:
                    source_name = slot_source["from"]
                    if not isinstance(source_name, str):
                        raise InvalidPack(f"slot {name!r} vocabulary reference must be a string")
                    choices = vocabulary.get(source_name)
                else:
                    choices = slot_source.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise InvalidPack(f"slot {name!r} has no choices")
                clean = tuple(choices)
                if any(not isinstance(choice, str) or not choice.strip() for choice in clean):
                    raise InvalidPack(f"slot {name!r} contains an invalid choice")
                if any(_has_control(choice) for choice in clean):
                    raise InvalidPack(f"slot {name!r} contains a control character")
                if len(set(clean)) != len(clean):
                    raise InvalidPack(f"slot {name!r} contains duplicate choices")
                for first in clean:
                    if any(first != second and second.startswith(first) for second in clean):
                        raise InvalidPack(f"slot {name!r} contains prefix-ambiguous choices")
                slots.append(CompiledSlot(name, clean, _power_bits(len(clean))))

            names = [slot.name for slot in slots]
            if set(fields) != set(names) or len(names) != len(fields):
                raise InvalidPack(f"template fields {set(fields)} do not match slot names {set(names)}")
            pattern = ["^"]
            slot_by_name = {slot.name: slot for slot in slots}
            for literal, field, _, _ in parsed_fields:
                pattern.append(re.escape(literal))
                if field is not None:
                    choices = sorted(slot_by_name[field].choices, key=len, reverse=True)
                    pattern.append("(?P<" + field + ">" + "|".join(re.escape(choice) for choice in choices) + ")")
            pattern.append("$")
            parser = re.compile("".join(pattern))
            literal_bytes = sum(len(literal.encode("utf-8")) for literal, _, _, _ in parsed_fields)
            largest_choice_bytes = sum(max(len(choice.encode("utf-8")) for choice in slot.choices) for slot in slots)
            if literal_bytes + largest_choice_bytes > MAX_CARRIER_TEXT_BYTES:
                raise InvalidPack("turn renders text larger than the carrier limit")
            turns.append(CompiledTurn(role, template, tuple(slots), sum(slot.bits for slot in slots), parser))
        return CompiledArc(source.get("name", "arc"), tuple(turns))

    def mapping_order(self, keys: DerivedKeys, arc_index: int, turn_index: int, slot_index: int) -> tuple[int, ...]:
        slot = self.arcs[arc_index].turns[turn_index].slots[slot_index]
        prefix = b"pcc/v1/choice\x00" + bytes.fromhex(self.pack_id)
        prefix += struct.pack(">III", arc_index, turn_index, slot_index)
        ordered = list(range(len(slot.choices)))
        ordered.sort(key=lambda index: hmac.new(keys.mapping, prefix + struct.pack(">I", index), hashlib.sha256).digest())
        return tuple(ordered)

    def arc_index(self, keys: DerivedKeys, sequence: int, segment: int) -> int:
        # The pack author controls the order of authored conversation arcs. A
        # larger pack can replace this simple cycle with an explicit schedule.
        del keys
        return (sequence + segment) % len(self.arcs)

    def render(self, keys: DerivedKeys, arc_index: int, turn_index: int, label: int) -> str:
        turn = self.arcs[arc_index].turns[turn_index]
        if label < 0 or label >= 2**turn.bits:
            raise ValueError("choice label outside turn")
        values: dict[str, str] = {}
        remaining = turn.bits
        for slot_index, slot in enumerate(turn.slots):
            remaining -= slot.bits
            slot_label = (label >> remaining) & ((1 << slot.bits) - 1)
            variant_index = self.mapping_order(keys, arc_index, turn_index, slot_index)[slot_label]
            values[slot.name] = slot.choices[variant_index]
        return turn.template.format(**values)

    def decode_label(self, keys: DerivedKeys, arc_index: int, turn_index: int, text: str) -> int:
        turn = self.arcs[arc_index].turns[turn_index]
        match = turn.parser.fullmatch(text)
        if match is None:
            raise PackMismatch(f"text does not belong to pack turn {turn_index}")
        label = 0
        for slot_index, slot in enumerate(turn.slots):
            variant_index = slot.choices.index(match.group(slot.name))
            slot_label = self.mapping_order(keys, arc_index, turn_index, slot_index).index(variant_index)
            label = (label << slot.bits) | slot_label
        return label
