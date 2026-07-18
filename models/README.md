# Optional Neural Carriers

The ONNX weights are intentionally not committed. GitHub rejects files over
100 MB, and the model licenses require separate attribution. Download a model
with:

```sh
python tools/download_models.py smollm360
```

Available artifacts:

| Name | Approximate weight size | Manifest | Result |
|---|---:|---|---|
| SmolLM2-135M INT8 | 136 MB | [`carrier-huffman16.json`](https://github.com/Mirrowel/plain-conversation-crypto/blob/main/models/smollm2-135m-int8/carrier-huffman16.json) | Lightest acceptable neural trial |
| SmolLM2-360M INT8 | 363 MB | [`carrier-huffman16.json`](https://github.com/Mirrowel/plain-conversation-crypto/blob/main/models/smollm2-360m-int8/carrier-huffman16.json) | Best tested quality/density balance |
| Qwen3-0.6B INT8 | 618 MB | [`carrier-huffman32.json`](https://github.com/Mirrowel/plain-conversation-crypto/blob/main/models/qwen3-0.6b-int8/carrier-huffman32.json) | Rejected in this carrier experiment |
| Gemma3-270M Q4 | 323 MB plus tokenizer | [`carrier-huffman32.json`](https://github.com/Mirrowel/plain-conversation-crypto/blob/main/models/gemma3-270m-q4/carrier-huffman32.json) | Rejected in this carrier experiment |
| LFM2.5-350M Q4 | 294 MB | [`carrier-huffman32.json`](https://github.com/Mirrowel/plain-conversation-crypto/blob/main/models/lfm2.5-350m-q4/carrier-huffman32.json) | Rejected due to generated repetition |

Arithmetic manifests (`carrier-arithmetic8.json` and
`carrier-arithmetic32.json`) reproduce the
[reference repository's](https://github.com/nethical6/conversation-steganography)
probability-weighted coding experiment. Arithmetic-32 is denser; arithmetic-8
is more conservative but can be much longer. See
[optimization report](https://github.com/Mirrowel/plain-conversation-crypto/blob/main/docs/optimization_report.md)
for measured failures and quality tradeoffs.

The manifests pin model and tokenizer SHA-256 hashes. A model file is protocol
material, not a cryptographic secret. The passphrase remains the security key.

Model sources and licenses:

- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), Apache-2.0
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B), Apache-2.0
- [Gemma 3](https://huggingface.co/google/gemma-3-270m-it), Gemma terms
- [LFM2.5](https://huggingface.co/LiquidAI/LFM2.5-350M), LFM Open License

Gemma 4 was evaluated for inclusion. Its smallest E2B variant is effectively
over 2B parameters and its ONNX decoder artifact exceeds the project’s 1 GB
model limit, so it is not included.
