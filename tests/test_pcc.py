from __future__ import annotations

import copy
import os
import random
import unittest

from pcc.codec import decode_message, encode_message
from pcc.crypto import derive_keys, decrypt, encrypt
from pcc.errors import AuthenticationError, CapacityExceeded, InvalidArgument, InvalidPack, PCCError
from pcc.pack import DialoguePack


ROOT = os.path.dirname(os.path.dirname(__file__))
PACK_PATH = os.path.join(ROOT, "packs", "semantic_demo.json")
KEY = "correct horse battery staple"


class PrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = DialoguePack.load(PACK_PATH)

    def test_pack_has_equal_capacity_and_coherent_surface_shape(self):
        self.assertEqual(self.pack.turn_count, 6)
        self.assertEqual([sum(turn.bits for turn in arc.turns) for arc in self.pack.arcs], [72, 72, 72])
        for arc in self.pack.arcs:
            for turn in arc.turns:
                self.assertNotRegex(turn.template, r"[\r\n\t]")
                self.assertGreater(turn.bits, 0)

    def test_round_trip_common_unicode_and_binary_messages(self):
        messages = [
            b"",
            b"hello",
            "Meet me at the cafe after work.".encode(),
            "caf\u00e9 \u0002".encode("utf-8"),
            bytes(range(256)),
            os.urandom(128),
        ]
        for sequence, message in enumerate(messages):
            with self.subTest(sequence=sequence):
                transcript = encode_message(self.pack, KEY, message, sequence, cover_bucket=0)
                self.assertEqual(decode_message(self.pack, KEY, transcript), message)

    def test_long_message_spans_multiple_authored_arcs(self):
        message = os.urandom(512)
        transcript = encode_message(self.pack, KEY, message, 9, cover_bucket=0)
        self.assertGreater(len(transcript["messages"]), self.pack.turn_count)
        self.assertEqual(decode_message(self.pack, KEY, transcript), message)

    def test_same_inputs_are_deterministic_and_sequence_is_bound(self):
        first = encode_message(self.pack, KEY, b"same message", 2, cover_bucket=0)
        second = encode_message(self.pack, KEY, b"same message", 2, cover_bucket=0)
        self.assertEqual(first, second)
        different_sequence = encode_message(self.pack, KEY, b"same message", 3, cover_bucket=0)
        self.assertNotEqual(first["messages"], different_sequence["messages"])

    def test_cover_padding_varies_visible_length_without_changing_plaintext(self):
        transcripts = [encode_message(self.pack, KEY, b"hello", 7) for _ in range(20)]
        lengths = {len(transcript["messages"]) for transcript in transcripts}
        self.assertGreater(len(lengths), 1)
        for transcript in transcripts:
            self.assertEqual(decode_message(self.pack, KEY, transcript), b"hello")
        with_padding = max(transcripts, key=lambda transcript: len(transcript["messages"]))
        invalid_tail = copy.deepcopy(with_padding)
        invalid_tail["messages"][-1]["text"] = "This is not valid cover padding."
        with self.assertRaises(PCCError):
            decode_message(self.pack, KEY, invalid_tail)
        stripped_tail = copy.deepcopy(with_padding)
        stripped_tail["messages"].pop()
        with self.assertRaises(PCCError):
            decode_message(self.pack, KEY, stripped_tail)

    def test_wrong_key_never_returns_plaintext(self):
        transcript = encode_message(self.pack, KEY, b"private", 4, cover_bucket=0)
        with self.assertRaises(PCCError):
            decode_message(self.pack, "a completely different strong key", transcript)

    def test_tamper_reorder_and_truncation_fail(self):
        transcript = encode_message(self.pack, KEY, b"private", 5, cover_bucket=0)

        tampered = copy.deepcopy(transcript)
        tampered["messages"][0]["text"] = "This is not a member of the dialogue pack."
        with self.assertRaises(PCCError):
            decode_message(self.pack, KEY, tampered)

        reordered = copy.deepcopy(transcript)
        reordered["messages"][0], reordered["messages"][1] = reordered["messages"][1], reordered["messages"][0]
        with self.assertRaises(PCCError):
            decode_message(self.pack, KEY, reordered)

        truncated = copy.deepcopy(transcript)
        truncated["messages"].pop()
        with self.assertRaises(PCCError):
            decode_message(self.pack, KEY, truncated)

    def test_pack_id_and_sequence_are_authenticated(self):
        transcript = encode_message(self.pack, KEY, b"private", 6, cover_bucket=0)
        altered_id = copy.deepcopy(transcript)
        altered_id["pack_id"] = "00" * 32
        with self.assertRaises(PCCError):
            decode_message(self.pack, KEY, altered_id)
        altered_sequence = copy.deepcopy(transcript)
        altered_sequence["sequence"] += 1
        with self.assertRaises(PCCError):
            decode_message(self.pack, KEY, altered_sequence)

    def test_mapping_is_keyed(self):
        keys_a = derive_keys(KEY, self.pack.salt)
        keys_b = derive_keys("another strong passphrase", self.pack.salt)
        text_a = self.pack.render(keys_a, 0, 0, 0)
        text_b = self.pack.render(keys_b, 0, 0, 0)
        self.assertNotEqual(text_a, text_b)

    def test_rendered_labels_parse_back_for_every_turn(self):
        keys = derive_keys(KEY, self.pack.salt)
        rng = random.Random(7)
        for arc_index, arc in enumerate(self.pack.arcs):
            for turn_index, turn in enumerate(arc.turns):
                labels = range(2**turn.bits) if turn.bits <= 8 else [rng.randrange(2**turn.bits) for _ in range(128)]
                for label in labels:
                    with self.subTest(arc=arc_index, turn=turn_index, label=label):
                        text = self.pack.render(keys, arc_index, turn_index, label)
                        self.assertEqual(self.pack.decode_label(keys, arc_index, turn_index, text), label)

    def test_invalid_pack_rejects_non_power_of_two_group(self):
        source = {
            "schema": 1,
            "salt": "MDEyMzQ1Njc4OWFiY2RlZg==",
            "vocabulary": {"choices": ["a", "b", "c"]},
            "arcs": [{"turns": [{"role": "Alice", "template": "{x}", "slots": [{"name": "x", "from": "choices"}]}]}],
        }
        with self.assertRaises(InvalidPack):
            DialoguePack.from_dict(source)

    def test_invalid_pack_rejects_control_and_format_characters(self):
        source = {
            "schema": 1,
            "salt": "MDEyMzQ1Njc4OWFiY2RlZg==",
            "vocabulary": {"choices": ["ok\u0000", "fine", "sure", "yes"]},
            "arcs": [{"turns": [{"role": "Alice", "template": "{x}", "slots": [{"name": "x", "from": "choices"}]}]}],
        }
        with self.assertRaises(InvalidPack):
            DialoguePack.from_dict(source)
        source["vocabulary"]["choices"][0] = "ok\ud800"
        with self.assertRaises(InvalidPack):
            DialoguePack.from_dict(source)

    def test_invalid_pack_rejects_non_alternating_or_inconsistent_roles(self):
        source = {
            "schema": 1,
            "salt": "MDEyMzQ1Njc4OWFiY2RlZg==",
            "vocabulary": {"choices": ["a", "b", "c", "d"]},
            "arcs": [{"turns": [
                {"role": "Alice", "template": "{x}", "slots": [{"name": "x", "from": "choices"}]},
                {"role": "Alice", "template": "{x}", "slots": [{"name": "x", "from": "choices"}]}
            ]}]
        }
        with self.assertRaises(InvalidPack):
            DialoguePack.from_dict(source)
        source["arcs"][0]["turns"] = [
            {"role": "Alice", "template": "{x}", "slots": [{"name": "x", "from": "choices"}]},
            {"role": "Bob", "template": "{x}", "slots": [{"name": "x", "from": "choices"}]},
            {"role": "Alice", "template": "{x}", "slots": [{"name": "x", "from": "choices"}]}
        ]
        with self.assertRaises(InvalidPack):
            DialoguePack.from_dict(source)

    def test_invalid_pack_rejects_oversized_rendered_turn(self):
        huge = "x" * 8_193
        source = {
            "schema": 1,
            "salt": "MDEyMzQ1Njc4OWFiY2RlZg==",
            "vocabulary": {"choices": [huge, "a", "b", "c"]},
            "arcs": [{"turns": [{"role": "Alice", "template": "{x}", "slots": [{"name": "x", "from": "choices"}]}]}],
        }
        with self.assertRaises(InvalidPack):
            DialoguePack.from_dict(source)

    def test_invalid_pack_rejects_wrong_field_types(self):
        source = {"schema": 1, "salt": None, "vocabulary": {}, "arcs": None}
        with self.assertRaises(InvalidPack):
            DialoguePack.from_dict(source)
        source["salt"] = "MDEyMzQ1Njc4OWFiY2RlZg=="
        source["arcs"] = [{"turns": [{"role": "Alice", "template": "{x}", "slots": [{"name": "x", "from": []}]}]}]
        with self.assertRaises(InvalidPack):
            DialoguePack.from_dict(source)

    def test_aead_rejects_wrong_associated_data(self):
        keys = derive_keys(KEY, self.pack.salt)
        sealed = encrypt(keys, self.pack.pack_id, 1, b"private")
        with self.assertRaises(AuthenticationError):
            decrypt(keys, self.pack.pack_id, 2, sealed)

    def test_extreme_cover_parameters_fail_cleanly(self):
        with self.assertRaises(InvalidArgument):
            encode_message(self.pack, KEY, b"hello", 0, cover_bucket=65536)
        with self.assertRaises(CapacityExceeded):
            encode_message(self.pack, KEY, os.urandom(200_000), 0, cover_bucket=0)

    def test_transcript_surrogate_fails_as_protocol_error(self):
        transcript = encode_message(self.pack, KEY, b"hello", 0, cover_bucket=0)
        transcript["messages"][0]["text"] = "\ud800"
        with self.assertRaises(PCCError):
            decode_message(self.pack, KEY, transcript)


if __name__ == "__main__":
    unittest.main()
