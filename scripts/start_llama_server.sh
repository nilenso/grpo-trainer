#!/bin/bash
set -eo pipefail

# Default values
MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
QUANT="Q8_0"
PORT=8080
CONTEXT_SIZE=4096
GPU_LAYERS=99
PARALLEL=4
MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Download a model from Hugging Face, convert to GGUF if needed, and start llama-server.

OPTIONS:
    -m, --model MODEL              HF model ID (default: $MODEL)
    -q, --quant TYPE               Quantization type (default: $QUANT)
                                   Common: Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, BF16
    -p, --port PORT                Port number (default: $PORT)
    -c, --context-size SIZE        Context size (default: $CONTEXT_SIZE)
    -ngl, --gpu-layers N           Layers to offload to GPU (default: $GPU_LAYERS)
    -np, --parallel N              Parallel sequences (default: $PARALLEL)
    --models-dir DIR               Model storage directory (default: $MODELS_DIR)
    -h, --help                     Show this help message

EXAMPLES:
    $0
    $0 --model kiranpg/Qwen2.5-OCamler-1.5B-Instruct --quant Q4_K_M
    $0 --model Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF --quant Q8_0
EOF
    exit 0
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)        MODEL="$2";      shift 2 ;;
        -q|--quant)        QUANT="$2";       shift 2 ;;
        -p|--port)         PORT="$2";        shift 2 ;;
        -c|--context-size) CONTEXT_SIZE="$2"; shift 2 ;;
        -ngl|--gpu-layers) GPU_LAYERS="$2";  shift 2 ;;
        -np|--parallel)    PARALLEL="$2";    shift 2 ;;
        --models-dir)      MODELS_DIR="$2";  shift 2 ;;
        -h|--help)         usage ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Convert/download model to GGUF
GGUF_PATH="$("$SCRIPT_DIR/hf_to_gguf.sh" --model "$MODEL" --quant "$QUANT" --models-dir "$MODELS_DIR")"

echo ""
echo "Starting llama-server..."
echo "   Model: $GGUF_PATH"
echo "   Quant: $(echo "$QUANT" | tr '[:lower:]' '[:upper:]')"
echo "   Port: $PORT"
echo "   Context Size: $CONTEXT_SIZE"
echo "   GPU Layers: $GPU_LAYERS"
echo "   Parallel: $PARALLEL"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "   Extra args: ${EXTRA_ARGS[*]}"
echo ""

exec llama-server \
    -m "$GGUF_PATH" \
    --port "$PORT" \
    --ctx-size "$CONTEXT_SIZE" \
    --n-gpu-layers "$GPU_LAYERS" \
    --parallel "$PARALLEL" \
    "${EXTRA_ARGS[@]}"
