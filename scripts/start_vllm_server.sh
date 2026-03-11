#!/bin/bash
set -eo pipefail

# Default values optimized for Qwen2.5-Coder-1.5B-Instruct evaluation
MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
PORT=8080
DTYPE="bfloat16"
GPU_MEMORY_UTIL=0.95
TENSOR_PARALLEL=1
MAX_NUM_SEQS=256
MAX_MODEL_LEN=4096
DISABLE_LOG_REQUESTS=true

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Start a vLLM OpenAI-compatible API server

OPTIONS:
    -m, --model MODEL              Model name or path (default: $MODEL)
    -p, --port PORT                Port number (default: $PORT)
    -d, --dtype DTYPE              Data type: float16, bfloat16, float32 (default: $DTYPE)
    -g, --gpu-memory FRACTION      GPU memory utilization 0.0-1.0 (default: $GPU_MEMORY_UTIL)
    -t, --tensor-parallel SIZE     Number of GPUs for tensor parallelism (default: $TENSOR_PARALLEL)
    -l, --max-len LENGTH           Maximum context length (default: $MAX_MODEL_LEN)
    -q, --quantization METHOD      Quantization: awq, gptq, squeezellm, fp8
    -s, --max-seqs NUMBER          Max sequences in parallel (default: $MAX_NUM_SEQS)
    --disable-log-requests         Disable request logging (default: enabled)
    --enable-log-requests          Enable request logging
    -h, --help                     Show this help message

EXAMPLES:
    # Start with default settings
    $0

    # Custom model and port
    $0 --model meta-llama/Llama-2-7b-hf --port 8000

    # Performance optimized for 2 GPUs
    $0 --tensor-parallel 2 --gpu-memory 0.95 --disable-log-requests

    # Limit context length for faster inference
    $0 --max-len 4096
EOF
    exit 0
}

# Parse arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -d|--dtype)
            DTYPE="$2"
            shift 2
            ;;
        -g|--gpu-memory)
            GPU_MEMORY_UTIL="$2"
            shift 2
            ;;
        -t|--tensor-parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        -l|--max-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        -q|--quantization)
            EXTRA_ARGS+=("--quantization" "$2")
            shift 2
            ;;
        -s|--max-seqs)
            MAX_NUM_SEQS="$2"
            shift 2
            ;;
        --disable-log-requests)
            DISABLE_LOG_REQUESTS=true
            shift
            ;;
        --enable-log-requests)
            DISABLE_LOG_REQUESTS=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Starting vLLM server..."
echo "   Model: $MODEL"
echo "   Port: $PORT"
echo "   Dtype: $DTYPE"
echo "   GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo "   Tensor Parallel Size: $TENSOR_PARALLEL"
echo "   Max Model Length: $MAX_MODEL_LEN"
echo "   Max Sequences: $MAX_NUM_SEQS"
echo "   Disable Log Requests: $DISABLE_LOG_REQUESTS"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "   Extra args: ${EXTRA_ARGS[*]}"
echo ""

# Build command with conditional arguments
CMD_ARGS=(
    --model "$MODEL"
    --dtype "$DTYPE"
    --port "$PORT"
    --host 0.0.0.0
    --tensor-parallel-size "$TENSOR_PARALLEL"
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len "$MAX_MODEL_LEN"
)

# Add disable-log-requests if enabled
if [[ "$DISABLE_LOG_REQUESTS" == "true" ]]; then
    CMD_ARGS+=(--no-enable-log-requests)
fi

# Add any extra arguments
CMD_ARGS+=("${EXTRA_ARGS[@]}")

exec uv run python -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}"
