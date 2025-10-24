#!/usr/bin/env bash
# Download the Tiny Shakespeare dataset used in many toy language modeling setups.

set -euo pipefail

DATA_URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUTPUT_ROOT="${1:-data}"
OUTPUT_PATH="${OUTPUT_ROOT%/}/tiny_shakespeare.txt"

mkdir -p "${OUTPUT_ROOT}"

if [[ -s "${OUTPUT_PATH}" ]]; then
    echo "Tiny Shakespeare dataset already exists at ${OUTPUT_PATH}." >&2
    exit 0
fi

echo "Downloading Tiny Shakespeare dataset to ${OUTPUT_PATH}..." >&2
curl -fL "${DATA_URL}" -o "${OUTPUT_PATH}"

echo "Download complete." >&2
