# Plain Conversation Crypto

Standalone prototype for carrying one encrypted logical message inside
ordinary-looking conversational text. This project is separate from the
[original `conversation-steganography` repository](https://github.com/nethical6/conversation-steganography).

The receiver pastes the visible cover bubbles belonging to one logical message
and gets one plaintext message back. A logical message may require multiple
cover bubbles; separate hidden messages are not batched by the basic protocol.

## Install

```sh
python -m pip install -e .
```

On Windows, install the optional DirectML runtime to use a compatible GPU:

```sh
python -m pip install -e ".[gpu-windows]"
```

For a CUDA 12 GPU and driver, install the pinned CUDA runtime profile instead:

```sh
python -m pip install -e ".[gpu-cuda12]"
```

CPU inference is the protocol default. Set `PCC_ONNX_PROVIDER=cuda` or `dml`
only when both peers use the same provider/runtime and intentionally select the
provider-specific carrier ID. Provider outputs are not assumed numerically
identical across hardware. On the tested GTX 1060, the quantized SmolLM graph
was slower on CUDA because provider copies dominated sequential generation.
The carrier ID also binds ONNX Runtime, tokenizer, NumPy, and graph-execution
settings; cross-machine determinism vectors are still required before calling
neural transport portable.

The runtime has no network requirement and no LLM requirement. Optional neural
carriers use local ONNX files. Download one separately:

```sh
python tools/download_models.py smollm360
```

See the [model documentation](https://github.com/Mirrowel/plain-conversation-crypto/blob/main/models/README.md) for model sizes, hashes, licenses, and the tested model
comparison.

## V2 CLI

Encode one message into a local JSON transcript:

```sh
python -m pcc encode-v2 \
  --carrier packs/semantic_demo.json \
  --profile secure \
  --message-file secret.txt \
  --sequence 0 \
  --output transcript.json
```

For copy/paste-oriented output, omit the JSON wrapper:

```sh
python -m pcc encode-v2 \
  --carrier packs/semantic_demo.json \
  --profile secure \
  --message "hello" \
  --sequence 1 \
  --plain-output > cover.txt
```

The key is prompted without echo when `--key` is omitted. Decode pasted
blank-line-separated cover bubbles with:

```sh
python -m pcc decode-v2 \
  --carrier packs/semantic_demo.json \
  --profile secure \
  --sequence 1 \
  --plain-input < cover.txt
```

The same commands work with a model manifest instead of a dialogue pack:

```sh
python -m pcc encode-v2 \
  --carrier models/smollm2-360m-int8/carrier-huffman16.json \
  --profile secure \
  --message "hello" \
  --sequence 1 \
  --plain-output > cover.txt
```

`--key` and `--message` remain available for demonstrations, but expose
secrets to shell history and process listings. Use prompted keys and files for
real use. The passphrase must contain at least 12 bytes; use a random secret,
not a memorable sentence.

## Profiles

| Profile | Encryption | Integrity | Use |
|---|---|---|---|
| `secure` | AES-SIV | 128-bit authenticated encryption | Strong default |
| `compact` | ChaCha20 | 64-bit truncated HMAC | Smaller authenticated messages with a lower forgery margin |
| `dense` | ChaCha20 | None | Passive-confidentiality experiments only |

The compact and dense profiles still use a strong stream cipher. They do not
use a deliberately weak encryption algorithm. The density improvement comes
from removing or reducing authentication overhead. `dense` must not be used
where undetected tampering matters.

The CLI requires `--allow-unsafe-dense` for the unauthenticated dense profile.

Compact and dense mode sequences must never repeat under the same key and
carrier. Reuse repeats the ChaCha20 keystream and can expose relationships
between plaintexts. JSON output generates a random sequence when omitted;
plain copy/paste output requires an explicit sequence that the receiver also
supplies.

Library callers must always pass `sequence` explicitly. The CLI generates a
random sequence only for JSON output, where it is carried in the local wrapper.

All profiles compress before encryption. Keyed interleaving is enabled by
default: it pseudorandomly permutes sealed bytes before carrier encoding
without adding visible data. Both sides can explicitly opt out with
`--no-interleave`. This is obfuscation, not extra cryptographic strength.

## Carriers

### Deterministic dialogue pack

[`packs/semantic_demo.json`](https://github.com/Mirrowel/plain-conversation-crypto/blob/main/packs/semantic_demo.json) is the no-LLM baseline. It contains typed,
human-authored conversation arcs, alternating speakers, and four independent
8-choice clause groups per turn. It carries 12 bits per coherent turn and is
the most predictable quality baseline.

[`packs/dense_semantic_pack.json`](https://github.com/Mirrowel/plain-conversation-crypto/blob/main/packs/dense_semantic_pack.json) is an experimental 21-bit-per-turn pack. Its
larger branching factor improves density, but every generated combination must
still receive human review. It is not selected as the default until that
quality gate passes.

### Compact statistical model

[`packs/chat_dense_model.json`](https://github.com/Mirrowel/plain-conversation-crypto/blob/main/packs/chat_dense_model.json) is a small order-4 word model trained from the
[`model_corpus.txt`](https://github.com/Mirrowel/plain-conversation-crypto/blob/main/packs/model_corpus.txt) corpus. It is lightweight and fast, and remains available as
an option. Its output is denser than the dialogue pack but does not guarantee
conversation-level coherence.

[`packs/chat_topic_model.json`](https://github.com/Mirrowel/plain-conversation-crypto/blob/main/packs/chat_topic_model.json) bundles topic-specific statistical models. It
reduces unrelated topic jumps at the cost of density.

### Local neural model

The best tested quality/density compromise is the local
`SmolLM2-360M-Instruct` INT8 ONNX model with
[`carrier-huffman16.json`](https://github.com/Mirrowel/plain-conversation-crypto/blob/main/models/smollm2-360m-int8/carrier-huffman16.json). It uses model probabilities
to build keyed Huffman choices, rather than forcing all top-k tokens to appear
equally often.

The 135M sibling is lighter but more generic. Huffman-32 is denser but produces
enough odd word choices that it remains experimental. Gemma3-270M, Qwen3-0.6B,
and LFM2.5-350M were also tested. Arithmetic coding improved some samples, but
none of those models is currently accepted as indistinguishable human text.
See the model README and benchmark artifacts. Neural quality checks fail closed
after their configured attempts; they are heuristics, not a proof of human
indistinguishability.

## Compression

V2 evaluates a deterministic portfolio and chooses the shortest self-describing
payload:

- Exact one-byte common-message codes.
- Raw bytes fallback.
- zlib and raw DEFLATE, with and without a fixed chat dictionary.
- BZ2 and LZMA baselines.
- Zstandard levels 3 and 19, with and without a fixed dictionary.
- Brotli qualities 5 and 11.
- Custom phrase-token coding and tokenized DEFLATE/Zstandard hybrids.

Every candidate has a bounded decoder and round-trip tests. Run the codec
benchmark against the authored 626-message chat corpus:

```sh
python tools/benchmark_compression_corpus.py \
  --corpus packs/model_corpus.txt \
  --output benchmarks/compression_chat_corpus.json
```

Observed results on that corpus, including the one-byte mode marker:

| Input | Median input | Median compressed | Median ratio | Main winner |
|---|---:|---:|---:|---|
| 1 sentence | 48 B | 40 B | 0.847 | Dictionary DEFLATE, 92.8% of messages |
| 2 sentences | 97 B | 73 B | 0.765 | Brotli-11 57.8%, dictionary DEFLATE 39.0% |
| 3 sentences | 146 B | 98 B | 0.680 | Brotli-11, 85.1% |

For high-entropy input, raw bytes correctly win. Compression cannot make
random data smaller.

## Benchmarks

Run the carrier comparison:

```sh
python -m pcc benchmark \
  --pack packs/semantic_demo.json \
  --model packs/chat_dense_model.json \
  --output benchmarks/carrier_comparison.json
```

The benchmark reports visible characters, cover bubbles, compression mode,
and encode/decode time for the pack and statistical model across all three
profiles. Neural carriers are benchmarked separately because model startup and
CPU inference dominate their timings.

The current engineering conclusion is:

- The authored pack is the strongest no-runtime-model coherence baseline.
- The statistical model is the lightest higher-density option, but its text
  needs a human-quality gate.
- SmolLM2-360M Huffman-16 is the best tested neural quality/density compromise,
  but it is too slow to call a cheap-phone default without device-specific
  optimization.
- SmolLM2-135M is the practical lightweight neural experiment, with a visible
  quality reduction.
- Higher branching increases nominal density but quickly violates the normal
  text requirement.

The full source-based reference comparison, model sweep, arithmetic-coding
experiment, and GPU timing study are in the
[optimization report](https://github.com/Mirrowel/plain-conversation-crypto/blob/main/docs/optimization_report.md).

## Security Boundary

The plaintext security boundary is the encryption profile, not the pack or
model. Pack/model artifacts may be known to an attacker; the passphrase must
still protect the plaintext.

The secure profile provides:

- Scrypt password derivation and HKDF domain-separated subkeys.
- AES-SIV authenticated encryption from `cryptography`.
- Sequence, carrier identity, profile, and interleave mode bound as associated data.
- Wrong-key, wrong-carrier, tamper, reorder, truncation, and malformed-input tests.

The prototype does not claim resistance to an adversary trained specifically
to detect its generated distribution. It also does not hide timing, participant
identity, message count, or social-graph metadata. Exact visible text matters;
normalization, rewriting, smart punctuation, or editing can break decoding.

## Tests

```sh
python -m unittest discover -s tests -v
```

The test suite covers V1 regression behavior, every compression candidate,
Unicode and binary round trips, all V2 profiles, both carriers, keyed
interleaving, wrong keys, tampering, malformed packs, and topic-model loading.
