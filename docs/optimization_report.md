# Density And Reference Comparison

Date: 2026-07-18

## Executive Summary

V2 is more modular and gives explicit security/density choices, but the
original `conversation-steganography` repository has two carrier improvements
that V2 initially lacked: probability-weighted arithmetic coding and a rolling
conversation prompt. Those ideas are now represented in V2's experimental
neural arithmetic carrier and in this report.

The practical result is not that one implementation universally wins:

- The reference is the stronger conversational architecture. Its chain prompt,
  strict token filtering, naturalness trials, arithmetic coding, and dynamic
  dictionary are all useful.
- V2 is the stronger experiment platform. It separates compression, crypto,
  carrier, model identity, sequence state, interleaving, and profiles, and it
  supports a no-model deterministic carrier.
- The V2 semantic pack is the only tested carrier with an authored coherence
  invariant and a reviewable state/grammar; this is not a formal guarantee.
- The V2 Markov carrier is fast and dense but fails the normal-conversation
  quality bar in sampled output.
- The committed Gemma 3 270M arithmetic-32 dense `hello` spot check produced
  93 characters but repeated its closing phrase, so it was rejected on quality.
- The requested 1-3x visible-text expansion was not achieved. Some measured
  neural paragraph samples were approximately 7x, but their linguistic quality
  was not independently validated and the path is much slower than the
  deterministic alternatives.

## Reference Implementation

The comparison target is `C:\Projects\conversation-steganography`. The source
was inspected directly. A native run was not performed because the workspace
does not have the Go toolchain installed and the reference also requires a
matching local model/runtime. The reference numbers below are therefore
source-derived budgets, not fabricated runtime measurements.

Important source facts:

- `cmd/conversation-stenography/setup.go:267-290` defaults to arithmetic
  coding, `CandidatePool=8`, `CarrierTrials=2`, strict style, temperature
  `1.0`, length bias `0.1`, and 32 finishing tokens. `TopN=8` is selected for
  the Transformers runtime; the other setup path uses `TopN=256`.
- `generative.go:202-260` uses a binary arithmetic decoder and encoder in
  parallel. It stops when the generated symbols have confirmed the complete
  framed payload.
- `arithmetic.go:17-66` converts model scores into a 32768-unit frequency
  table, guaranteeing every candidate a nonzero interval.
- `generative.go:698-741` requests a larger candidate pool, filters unsafe
  visible tokens, and keeps the top configured candidates.
- `generative.go:744-779` rejects labels, metadata words, controls, digits, and
  other visibly non-chat tokens.
- `conversation_chain.go:313-341` puts the previous visible conversation into
  the next model prompt. This is a major coherence advantage over a static
  prompt.
- `conversation_chain.go:150-213` generates multiple carrier trials and picks
  the shortest one within the naturalness slack.
- `message_compression.go:126-175` tries common phrases, static dictionary
  DEFLATE, a dynamic dictionary, fragment coding, compact fragments, and dense
  fragments, selecting the shortest packed result.
- `conversation_chain.go:418-437` builds a rolling compression dictionary from
  the static chat dictionary plus recent visible carrier text.
- `conversation_chain.go:363-369` seals the packed bytes with AES-SIV.
- `conversation_chain.go:88-95` accounts for a variable-length frame; the
  16-byte tag is assigned at `conversation_chain.go:76` and defined by
  `siv.go:12`.
- `conversation_chain.go:130-148` requires exact message ordering and sender
  state, while `conversation_chain.go:347-354` and `440-450` bind the rolling
  chain into authenticated data. Reordering is intentionally rejected.

### Reference `hello` Budget

The reference's `commonMessages` table does not contain exactly `hello`, but
its fragment protocol does. The shortest `hello` packing path is exactly 2
bytes: one mode byte and one fragment index byte. The default arithmetic chain
is explicitly unframed (`conversation_chain.go:309-311`), so AES-SIV adds 16
bytes and no extra frame byte.

```text
2-byte packed message
+ 16-byte SIV tag
= 18 bytes = 144 hidden bits
```

With eight arithmetic candidates, the Transformers setup has a theoretical
ceiling of 3 bits per data token. A best-case lower bound is therefore about
48 data tokens, before
sentence finishing and before the model's nonuniform entropy is considered.

This is comparable to V2's neural arithmetic path, not to the V2 semantic
pack. The reference's output length cannot be stated honestly without running
the same model and tokenizer.

## V2 Results

The measurements are stored under `benchmarks/` and use one logical hidden
message per encode operation.

### Authored Semantic Pack

| Profile | `hello` bubbles | `hello` visible chars | Paragraph visible chars |
|---|---:|---:|---:|
| Secure AES-SIV | 12 | 1,551 | 8,577 |
| Compact ChaCha20 + 64-bit HMAC | 7 | 833 | 7,876 |
| Dense ChaCha20 without authentication | 2 | 223 | 7,259 |

This carrier is verbose because each turn carries only 12 authored choice bits.
Its advantage is that the state and grammar make every generated combination
valid by construction.

### Lightweight Statistical Models

The global order-4 Markov model is approximately 0.8 MB and runs in well under
one second per message on this machine.

| Carrier/profile | `hello` chars | Short chat chars | Paragraph chars |
|---|---:|---:|---:|
| Markov secure | 743 | 1,282 | 4,233 |
| Markov compact | 404 | 1,019 | 3,835 |
| Markov dense | 140 | 718 | 3,655 |

The density is attractive, but sampled output such as three unrelated sentences
about a lemonade stand, a forecast, and an espresso machine fails the normal
conversation requirement. The Markov model remains an option and a useful
lower-bound experiment, not the default carrier.

### Neural Models

The best tested neural profiles use local ONNX models and arithmetic coding.
They produce one visible message per logical message, but generation is
sequential.

| Model/profile | Dense `hello` chars | Dense paragraph chars | Notes |
|---|---:|---:|---|
| SmolLM2-135M arithmetic-32 | 96 | 1,128 | Fastest neural option, weaker wording |
| SmolLM2-360M arithmetic-32 | 133 | 1,372 | Better wording, slower |
| Gemma3-270M arithmetic-32 | 93 exploratory spot check | Not benchmarked | Repetitive sample; rejected |
| Qwen3-0.6B arithmetic-32 | 86 exploratory spot check | Not benchmarked | `OST:` prefix and refusal-like text; rejected |
| LFM2.5-350M arithmetic-32 | Failed spot check | Not usable | Tokenization failure; rejected |

The exact SmolLM2 samples and timings are kept in the neural benchmark JSON
files. The Gemma/Qwen/LFM values are explicitly single-message exploratory
spot checks in `benchmarks/model_spot_checks.json`, not quality benchmarks.
Arithmetic-8 was more conservative but substantially longer. Huffman-32 was
denser in some cases but forced less probable words and was less natural.

## Experiments That Mattered

### Compression portfolio

V2 tests raw bytes, common-message codes, custom phrase tokens, zlib/DEFLATE
with and without dictionaries, BZ2, LZMA, Zstandard, Brotli, and hybrids. On
626 normal chat messages, dictionary DEFLATE dominated one-sentence inputs,
Brotli-11 dominated three-sentence inputs, and raw bytes correctly won for
incompressible data.

The reference's dynamic dictionary is a worthwhile future session feature, but
it is not a universal win: previous visible carrier text only helps when it
shares phrases with the next hidden plaintext. V2 already has a richer static
chat dictionary and can add dynamic history when a conversation context is
available.

### Arithmetic versus Huffman

Reference-style arithmetic coding is more model-distribution-faithful than a
fixed-width top-k carrier and usually improves short natural text. It can also
fall into very low-probability intervals, producing extremely long repetitive
carriers. V2 therefore has bounded token generation and a repetition quality
gate. A production implementation should use multiple temperature/length-bias
trials and reject pathological candidates before sending.

### Interleaving

V2 now enables keyed interleaving by default. It permutes sealed bytes before
language encoding and reverses the permutation after carrier decoding. This
means the encoded secret bytes are not placed sequentially in visible output.
It adds no bytes. It does not make strong encryption stronger, and it does not
yet allow physical cover bubbles to arrive in arbitrary order.

### GPU execution

The test machine has a GTX 1060 6 GB. DirectML loaded the model but was slower
than CPU for one-token sequential inference. CUDA was made to load with CUDA
12.8/cuDNN 9, but the long SmolLM2 test was worse:

| Provider | Load | Long dense encode | Visible chars |
|---|---:|---:|---:|
| CPU | 3.45 s | 28.3 s | 972 |
| CUDA | 5.33 s | 95.4 s | 2,784 |

The CUDA graph inserted many memory-copy operations. Provider-level numerical
differences also changed token choices, proving that provider selection must be
part of the carrier identity. The raw study is committed as
`benchmarks/gpu_provider_study.json`. CPU is therefore the protocol default;
CUDA and DirectML are explicit experiments only.

## Recommended Direction

The strongest next architecture is a hybrid of the best reference ideas and
V2's explicit profile boundary:

1. Keep one logical plaintext per operation.
2. Compress with the current portfolio, plus optional rolling history when the
   user supplies prior cover/context.
3. Keep AES-SIV secure mode and dense stream mode separate.
4. Use probability-weighted arithmetic coding for neural carriers.
5. Add the reference's rolling visible conversation prompt to neural mode.
6. Use strict visible-token filtering and multiple naturalness trials.
7. Keep the semantic pack as the no-model quality fallback.
8. Treat Markov as a fast density baseline, not as a coherence guarantee.
9. Keep interleaving default-on.
10. Benchmark each model/provider with a hard time and token budget.

The fundamental limit remains: compression can reduce the hidden payload, but a
normal-language carrier has limited entropy. A strong 1-3x ratio is plausible
for medium, compressible chat only with a high-quality local model and a good
probability coder. It is not credible for `hello`, independently authenticated
short messages, or incompressible data.

## Reproducibility Limits

The reference runtime was not executed because Go and its exact model runtime
were unavailable in this workspace. V2 runtime measurements are local to the
listed ONNX files, tokenizer, provider, and hardware. The report deliberately
labels source-derived reference budgets separately from measured V2 output.
