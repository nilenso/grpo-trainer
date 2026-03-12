#!/bin/bash
set -eo pipefail

# Convert a Hugging Face model to quantized GGUF format.
# Handles both GGUF repos (direct download) and safetensors repos (convert + quantize).
# Outputs the path to the final GGUF file on stdout.

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"

usage() {
    cat << EOF
Usage: $0 -m MODEL [-q QUANT] [--models-dir DIR]

Download and convert a Hugging Face model to GGUF format.

OPTIONS:
    -m, --model MODEL       HF model ID (required)
    -q, --quant TYPE        Quantization type (default: Q8_0)
                            Common: Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, BF16
    --models-dir DIR        Model storage directory (default: $MODELS_DIR)
    -h, --help              Show this help message

Prints the path to the resulting GGUF file on stdout.
EOF
    exit 0
}

MODEL=""
QUANT="Q8_0"

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)  MODEL="$2";      shift 2 ;;
        -q|--quant)  QUANT="$2";      shift 2 ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        -h|--help)   usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required" >&2
    exit 1
fi

mkdir -p "$MODELS_DIR"

MODEL_SAFE_NAME="${MODEL//\//__}"
QUANT_UPPER="$(echo "$QUANT" | tr '[:lower:]' '[:upper:]')"
QUANT_LOWER="$(echo "$QUANT" | tr '[:upper:]' '[:lower:]')"
GGUF_PATH="$MODELS_DIR/${MODEL_SAFE_NAME}-${QUANT_UPPER}.gguf"

if [[ -f "$GGUF_PATH" ]]; then
    echo "$GGUF_PATH"
    exit 0
fi

if [[ "$MODEL" == *-GGUF ]] || [[ "$MODEL" == *-gguf ]]; then
    # GGUF repo: download the specific quantization file directly
    BASE_NAME="${MODEL##*/}"
    BASE_NAME="${BASE_NAME%-GGUF}"
    BASE_NAME="${BASE_NAME%-gguf}"
    FILE_NAME="$(echo "${BASE_NAME}-${QUANT_LOWER}.gguf" | tr '[:upper:]' '[:lower:]')"

    echo "Downloading $FILE_NAME from $MODEL..." >&2
    huggingface-cli download "$MODEL" "$FILE_NAME" --local-dir "$MODELS_DIR" >&2

    if [[ ! -f "$MODELS_DIR/$FILE_NAME" ]]; then
        echo "Error: Expected file $MODELS_DIR/$FILE_NAME not found after download" >&2
        exit 1
    fi
    mv "$MODELS_DIR/$FILE_NAME" "$GGUF_PATH"
else
    # Safetensors repo: download, convert to F16 GGUF, then quantize
    HF_DIR="$MODELS_DIR/hf/$MODEL_SAFE_NAME"
    F16_GGUF="$MODELS_DIR/${MODEL_SAFE_NAME}-F16.gguf"

    if [[ ! -d "$HF_DIR" ]] || [[ ! -f "$HF_DIR/config.json" ]]; then
        echo "Downloading $MODEL from Hugging Face..." >&2
        mkdir -p "$HF_DIR"
        huggingface-cli download "$MODEL" --local-dir "$HF_DIR" >&2
    else
        echo "Using cached HF download: $HF_DIR" >&2
    fi

    if [[ ! -f "$F16_GGUF" ]]; then
        echo "Converting to GGUF (F16)..." >&2
        uv run convert_hf_to_gguf.py "$HF_DIR" --outfile "$F16_GGUF" --outtype f16 >&2
    else
        echo "Using cached F16 GGUF: $F16_GGUF" >&2
    fi

    if [[ "$QUANT_UPPER" == "F16" ]]; then
        GGUF_PATH="$F16_GGUF"
    else
        echo "Quantizing to $QUANT_UPPER..." >&2
        llama-quantize "$F16_GGUF" "$GGUF_PATH" "$QUANT_UPPER" >&2
    fi
fi

echo "$GGUF_PATH"
