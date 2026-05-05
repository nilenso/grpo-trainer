# grpo-trainer

`grpo-trainer` is a toolkit for fine-tuning LLMs to generate high-quality OCaml code. It supports supervised fine-tuning (SFT), RLVR/GRPO training, evaluation, and monitoring with feedback from the OCaml compiler and test suite.

## Prerequisites

- **Nix with flakes enabled** for the development shell.
- **CUDA-capable Linux server** for GPU training.
- **OCaml toolchain** and **llama.cpp** are provided by the Nix shell.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/nilenso/grpo-trainer.git
cd grpo-trainer
```

### 2. Enter the development environment

For local development, enter the Nix shell:

```bash
make shell
```

On Linux, `make shell` enters the CUDA-enabled shell (`nix develop --impure .#cuda`). On other platforms, it enters the default shell.

### 3. Bootstrap a Linux training server

On a fresh Linux training server, run:

```bash
scripts/bootstrap.sh
```

The bootstrap script:

- refuses to run as root;
- installs Nix if `nix` is not already available on `PATH`;
- persists the Nix profile setup in `~/.bashrc`;
- enables Nix flakes and `nix-command`;
- links CUDA driver libraries into `.cuda-driver/`;
- enters the CUDA dev shell and installs CUDA-enabled Python dependencies.

If you are setting up manually instead, install CUDA-enabled Python dependencies inside the Nix shell:

```bash
uv sync --extra cuda
```

### 4. Start a model server

The Nix shell includes llama.cpp. To start a server from a Hugging Face GGUF model:

```bash
llama-server -hf unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF:F16 -c 4096 -ngl -1
```

You can also use the Makefile helpers:

```bash
make llama-server-hf
make vllm-server model=<model_id>
```

## Configuration

Training and evaluation configuration is read from `.envrc` via [direnv](https://direnv.net/). Keep local secrets and machine-specific settings out of source control.

Common settings include model IDs, dataset paths, output directories, LoRA parameters, batch sizes, and evaluation server configuration.

## RLVR / GRPO Training

Train the base model with GRPO:

```bash
make rlvr-train
```

This starts training with the [default OCaml training dataset](https://huggingface.co/datasets/kiranpg/ocaml-training-problems), writes logs to `training.log`, and stores run artifacts in `grpo_runs/`.

### Reward system

Training uses a graduated reward system with learning signals from multiple stages:

| Stage | Max Score | Description |
|-------|-----------|-------------|
| Type check | 0.25 | Graduated credit based on error count; zero errors receives full credit. |
| Compilation | 0.10 | Full credit for successful compilation; partial credit for type-checked code that fails compilation. |
| Tests | 0.65 | Graduated credit: `0.65 × (passed_assertions / total_assertions)`. |

Correct but verbose solutions may receive a style penalty of up to `0.10`:

- Extra code blocks: `0.02` per block beyond the first.
- Trailing prose after the final code fence: `0.03`.

Completions with repetitive content, low code ratio, code block spam, or stub implementations are flagged as degenerate and penalized.

### Metrics

Training logs are written to `training.log`. Structured metrics are written to `grpo_runs/`:

| File | Description |
|------|-------------|
| `metrics.jsonl` | Step-level learning metrics: epoch, loss, grad norm, learning rate, and reward statistics. |
| `batch_metrics.jsonl` | Batch-level reward statistics. |
| `{reward_name}.jsonl` | Per-completion reward outcomes. |

For detailed metric descriptions, see [doc/metrics.md](doc/metrics.md).

### Monitoring

| Service | URL | Description |
|---------|-----|-------------|
| Dashboard | <http://localhost:8080> | Real-time GRPO training metrics. |
| Completions Viewer | <http://localhost:8080/completions> | Browse model completions. |
| TensorBoard | <http://localhost:6006> | Loss curves, learning rate, and gradients. |

## Supervised Fine-Tuning

Pre-train the base model on OCaml code examples before RLVR:

```bash
make sft-train
```

SFT uses TRL's `SFTTrainer` with LoRA on the [OCaml SFT dataset](https://huggingface.co/datasets/kiranpg/ocaml-sft-problems).

### Training strategy

SFT uses completion-only training with TRL's prompt-completion dataset format:

- **Prompt**: problem description and function signature, masked from loss.
- **Completion**: OCaml code in markdown blocks, trained on.

When `SFTTrainer` receives `prompt` and `completion` columns, it masks prompt tokens from the loss (`completion_only_loss=True` by default). This teaches the model to generate OCaml code blocks without spending capacity on predicting prompt text.

### Metrics and monitoring

SFT artifacts are written to:

- `sft_training.log` — SFT training output.
- `sft_runs/` — SFT metrics and checkpoints.

| Service | URL | Description |
|---------|-----|-------------|
| SFT Dashboard | <http://localhost:8080/sft> | SFT training metrics. |
| TensorBoard | <http://localhost:6006> | Loss curves, learning rate, and gradients. |

## Post-training

### Merge the LoRA adapter

After training, merge the LoRA adapter into the base model to create a standalone model:

```bash
make merge-adapter path=<adapter-path>
```

This requires `BASE_MODEL_ID` to be set in `.envrc`. The merged model is saved to `merged_model/`.

### Convert to GGUF

The Nix shell includes llama.cpp tools. To convert the merged model to GGUF:

```bash
# Enter the Nix shell if needed.
make shell

# Convert to GGUF. Requires llama.cpp Python dependencies.
pip install gguf
python -m llama_cpp.convert merged_model --outfile model.gguf

# Quantize for smaller inference artifacts.
llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M
```

### Evaluate a model

Run evaluation against the configured test set:

```bash
make eval model=<model_name>
```

Limit the number of examples when smoke testing:

```bash
make eval model=<model_name> limit=10
```

## Common commands

| Command | Description |
|---------|-------------|
| `make shell` | Enter the Nix development shell. |
| `make deps` | Install Python dependencies with `uv sync --frozen`. |
| `make rlvr-train` | Run RLVR/GRPO training. |
| `make sft-train` | Run supervised fine-tuning. |
| `make eval model=<name>` | Evaluate a model. |
| `make dashboard` | Start the dashboard server. |
| `make test` | Run the pytest suite. |
| `make lint` | Run Ruff on changed Python files. |
| `make fmt` | Format Python files with Ruff. |

## Project structure

```text
grpo-trainer/
├── rlvr/                    # RLVR/GRPO training module
│   ├── train.py             # Main GRPO training script
│   ├── environment.py       # Verifiers-compatible environment
│   ├── reward.py            # Reward computation: type check, compile, tests
│   ├── config.py            # Training configuration
│   ├── data.py              # Dataset loading
│   └── logging.py           # Metrics logging utilities
├── sft/                     # Supervised fine-tuning module
│   ├── train.py             # SFT training with TRL's SFTTrainer
│   ├── config.py            # LoRA configuration
│   ├── data.py              # Dataset loading from Hugging Face
│   └── logging.py           # SFT metrics logging
├── eval/                    # Evaluation module
│   ├── eval.py              # Model evaluation script
│   ├── compare.py           # Compare model outputs
│   └── metrics.py           # Evaluation metrics
├── dashboard/               # Real-time training dashboard
│   ├── server.py            # Dashboard backend
│   ├── index.html           # GRPO metrics dashboard
│   ├── completions.html     # Completions viewer
│   └── sft.html             # SFT metrics dashboard
├── tests/                   # Unit tests
│   ├── test_environment.py  # Environment tests
│   ├── test_reward.py       # Reward computation tests
│   └── test_partial_rewards.py
└── scripts/
    ├── run-sft.sh           # SFT training launcher
    ├── run-rlvr-training.py # RLVR training launcher
    ├── run-eval.sh          # Evaluation script
    ├── merge_adapter.py     # Merge LoRA adapter into base model
    ├── start_vllm_server.sh # Start vLLM inference server
    └── bootstrap.sh         # Linux training server bootstrap script
```
