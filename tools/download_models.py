"""Download optional local neural carrier artifacts without committing weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


MODELS = {
    "smollm135": (
        "onnx-community/SmolLM2-135M-Instruct-ONNX",
        "models/smollm2-135m-int8",
        ["onnx/model_int8.onnx", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "generation_config.json"],
    ),
    "smollm360": (
        "onnx-community/SmolLM2-360M-Instruct-ONNX",
        "models/smollm2-360m-int8",
        ["onnx/model_int8.onnx", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "generation_config.json"],
    ),
    "qwen3": (
        "onnx-community/Qwen3-0.6B-ONNX",
        "models/qwen3-0.6b-int8",
        ["onnx/model_int8.onnx", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "generation_config.json"],
    ),
    "gemma3": (
        "onnx-community/gemma-3-270m-it-ONNX",
        "models/gemma3-270m-q4",
        ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "generation_config.json"],
    ),
    "lfm25": (
        "onnx-community/LFM2.5-350M-ONNX",
        "models/lfm2.5-350m-q4",
        ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data", "tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json"],
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODELS))
    args = parser.parse_args()
    repository, directory, files = MODELS[args.model]
    Path(directory).mkdir(parents=True, exist_ok=True)
    for filename in files:
        print(hf_hub_download(repository, filename=filename, local_dir=directory))
    print("Load the matching carrier-*.json manifest from the same directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
