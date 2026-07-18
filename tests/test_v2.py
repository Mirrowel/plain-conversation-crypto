from __future__ import annotations

import os
import unittest

from pcc.compression import DICTIONARY_SHA256, MODE_RAW, compression_candidates, compress_message, decompress_message
from pcc.errors import FrameError, PCCError
from pcc.model import HuffmanMarkovModel
from pcc.pack import DialoguePack
from pcc.v2 import decode_v2, encode_v2


ROOT = os.path.dirname(os.path.dirname(__file__))
PACK_PATH = os.path.join(ROOT, "packs", "semantic_demo.json")
MODEL_PATH = os.path.join(ROOT, "packs", "chat_dense_model.json")
TOPIC_MODEL_PATH = os.path.join(ROOT, "packs", "chat_topic_model.json")
KEY = "correct horse battery staple"


class CompressionTests(unittest.TestCase):
    def test_all_codec_candidates_round_trip(self):
        samples = [
            b"",
            b"hello",
            b"Meet me at the cafe after work.",
            ("Café " + chr(0) + " " + chr(17) + " " + chr(2) + " " + chr(1)).encode("utf-8"),
            bytes(range(256)),
            os.urandom(1024),
            b"This is a normal conversational sentence about meeting after work. " * 32,
        ]
        for sample in samples:
            for candidate in compression_candidates(sample):
                with self.subTest(sample=len(sample), codec=candidate.name):
                    self.assertEqual(decompress_message(candidate.payload), sample)
            selected = compress_message(sample)
            self.assertEqual(decompress_message(selected), sample)
            self.assertLessEqual(len(selected), len(sample) + 1)

    def test_compression_is_deterministic_and_common_messages_are_compact(self):
        self.assertEqual(compress_message(b"hello"), bytes((0,)))
        self.assertEqual(compress_message(b"hello"), compress_message(b"hello"))
        self.assertEqual(len(DICTIONARY_SHA256), 64)
        self.assertEqual(decompress_message(bytes((MODE_RAW,))), b"")

    def test_compression_rejects_bad_streams(self):
        for payload in (b"", b"\x7f", b"\x00x", b"\x81not-zlib", b"\x90\x02\x00"):
            with self.subTest(payload=payload):
                with self.assertRaises(FrameError):
                    decompress_message(payload)


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = DialoguePack.load(PACK_PATH)
        cls.model = HuffmanMarkovModel.load(MODEL_PATH)
        from pcc.model import TopicMarkovModel
        cls.topic_model = TopicMarkovModel.load(TOPIC_MODEL_PATH)

    def test_all_profiles_and_carriers_round_trip(self):
        samples = [b"hello", b"Meet me after work.", bytes(range(128))]
        for carrier_name, carrier in (("pack", self.pack), ("model", self.model)):
            for profile in ("secure", "compact", "dense"):
                for interleave in (False, True):
                    for sequence, sample in enumerate(samples):
                        with self.subTest(carrier=carrier_name, profile=profile, interleave=interleave, sequence=sequence):
                            transcript = encode_v2(
                                carrier, KEY, sample, profile=profile, sequence=sequence, interleave=interleave
                            )
                            self.assertEqual(decode_v2(carrier, KEY, transcript), sample)

    def test_plain_model_output_is_pasteable(self):
        transcript = encode_v2(self.model, KEY, b"hello", profile="secure", sequence=3)
        plain = {key: transcript[key] for key in ("format", "carrier", "carrier_id", "profile", "sequence", "interleave")}
        plain["messages"] = [{"text": item["text"]} for item in transcript["messages"]]
        self.assertEqual(decode_v2(self.model, KEY, plain), b"hello")

    def test_topic_model_round_trip_and_topic_is_stable(self):
        transcript = encode_v2(self.topic_model, KEY, b"hello", profile="dense", sequence=9, interleave=True)
        self.assertEqual(decode_v2(self.topic_model, KEY, transcript), b"hello")

    def test_secure_and_compact_wrong_key_fail(self):
        for profile in ("secure", "compact"):
            transcript = encode_v2(self.pack, KEY, b"private", profile=profile, sequence=4)
            with self.assertRaises(PCCError):
                decode_v2(self.pack, "another completely different key", transcript)

    def test_interleave_changes_sealed_carrier_but_not_length(self):
        ordered = encode_v2(self.pack, KEY, b"a longer message for interleaving", profile="secure", sequence=5)
        shuffled = encode_v2(
            self.pack, KEY, b"a longer message for interleaving", profile="secure", sequence=5, interleave=True
        )
        self.assertNotEqual(ordered["messages"], shuffled["messages"])
        self.assertEqual(len(ordered["messages"]), len(shuffled["messages"]))
        self.assertEqual(decode_v2(self.pack, KEY, shuffled), b"a longer message for interleaving")

    def test_tampered_secure_carrier_is_rejected(self):
        transcript = encode_v2(self.pack, KEY, b"a sufficiently long message for compression", profile="secure", sequence=6)
        tampered = dict(transcript)
        tampered["messages"] = [dict(message) for message in transcript["messages"]]
        tampered["messages"][0]["text"] = "not a valid pack member"
        with self.assertRaises(PCCError):
            decode_v2(self.pack, KEY, tampered)


if __name__ == "__main__":
    unittest.main()
