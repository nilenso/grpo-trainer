# grpo-trainer

**grpo-trainer** is a toolkit for fine-tuning LLMs to generate high-quality OCaml code using RLVR and GRPO.

## Prerequisites

- **Nix** (with flakes enabled): Install it from [here](https://nixos.wiki/wiki/Nix_Installation_Guide).

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/grpo-trainer.git
cd grpo-trainer
```

### 2. Setup Environment with Nix

Enter a development shell with Python, OCaml, uv, and all tools pre-installed:

```bash
make shell
```

For a Linux training server, run the bootstrap script to install/configure Nix, persist the Nix profile setup in `~/.bashrc`, link CUDA driver libraries, enter the CUDA dev shell, and install CUDA-enabled Python dependencies:

```bash
scripts/bootstrap.sh
```

Do not run the bootstrap script as root.

### 2.5 Install pytorch with CUDA support

If you are setting up manually instead of using `scripts/bootstrap.sh`, install PyTorch with CUDA support inside the Nix shell:

```bash
uv sync --extra cuda
```

### 3. Start Model Server

The Nix environment includes llama.cpp pre-installed. Start a model server:

```bash
llama-server -hf unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF:F16 -c 4096 -ngl -1
```

## Configuration

The training parameters and other related configuration is present in `.envrc` which are sourced using [direnv](https://direnv.net/).


## RLVR Training

Train the base model (default: Qwen2.5-Coder:1.5B-Instruct) using GRPO:

```bash
make rlvr-train
```

This starts the model training using the [default training dataset](https://huggingface.co/datasets/kiranpg/ocaml-training-problems) in the background and logs to `training.log`.

### Reward System Architecture

The training uses a graduated reward system that provides learning signals at multiple compilation stages:

| Stage | Max Score | Description |
|-------|-----------|-------------|
| Type Check | 0.25 | Graduated credit based on error count (0 errors = full, 1 error = 0.20, decreasing to 0.01 for 10+) |
| Compilation | 0.10 | Full credit for successful compilation, partial credit (0.01-0.05) for type-checked but failed compile |
| Tests | 0.65 | Graduated credit: `0.65 × (passed_assertions / total_assertions)` |

**Style Penalty:** Verbose but correct solutions receive a penalty up to 0.10:
- Extra code blocks: 0.02 per block beyond the first
- Trailing prose (>30 chars after final code fence): 0.03

**Degenerate Detection:** Completions with repetitive content, low code ratio, code block spam, or stub solutions are flagged and penalized.

### Metrics

The training logs are written to `training.log` in project root. Training output/metrics are written to `grpo_runs` directory:

| File | Description |
|------|-------------|
| `metrics.jsonl` | Essential learning metrics per step (epoch, loss, grad_norm, learning_rate, reward stats) |
| `batch_metrics.jsonl` | Batch-level reward statistics |
| `{reward_name}.jsonl` | Per-completion reward outcomes |

For detailed metric descriptions, see [doc/metrics.md](doc/metrics.md).

### Monitoring

| Service | URL | Description |
|---------|-----|-------------|
| Dashboard | http://localhost:8080 | Real-time GRPO training metrics |
| Completions Viewer | http://localhost:8080/completions | Browse model completions |
| TensorBoard | http://localhost:6006 | Loss curves, learning rate, gradients |

## Post-RLVR-Training

### Merging the Adapter

After training, merge the LoRA adapter into the base model to create a standalone model:

```bash
make merge-adapter path=<adapter-path>
```

This requires `BASE_MODEL_ID` to be set in `.envrc`. The merged model will be saved to `merged_model/`.

### Converting to GGUF

The Nix shell includes llama.cpp tools. To convert the merged model to GGUF format:

```bash
# Enter nix shell if not already
make shell

# Convert to GGUF (requires llama.cpp Python dependencies)
pip install gguf
python -m llama_cpp.convert merged_model --outfile model.gguf

# Quantize for smaller size (llama-quantize is available in nix shell)
llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M
```

### Evaluating Model Performance

Assess model performance against test cases (pass the model name):

```bash
make eval model=<model_name>
```

## Supervised Fine-Tuning

Pre-train the base model on OCaml code examples before RLVR:

```bash
make sft-train
```

This uses TRL's SFTTrainer with LoRA to fine-tune on the [SFT dataset](https://huggingface.co/datasets/kiranpg/ocaml-sft-problems).

### Training Strategy

SFT uses **completion-only training** via TRL's native prompt-completion dataset format:

- **Prompt**: Problem description + function signature (masked from loss)
- **Completion**: Code in markdown blocks ` ```ocaml...``` ` (trained on)

When SFTTrainer receives a dataset with `prompt` and `completion` columns, it automatically masks prompt tokens from the loss (`completion_only_loss=True` by default). The model learns to generate OCaml code blocks without wasting capacity learning to predict prompt tokens.

### Metrics

SFT training data are written to:
- `sft_training.log` - SFT training output
- `sft_runs/` - SFT metrics and checkpoints

### Monitoring

The training script automatically starts monitoring services:

| Service | URL | Description |
|---------|-----|-------------|
| SFT Dashboard | http://localhost:8080/sft | SFT training metrics |
| TensorBoard | http://localhost:6006 | Loss curves, learning rate, gradients |


## Project Structure

```
grpo-trainer/
├── rlvr/                    # RLVR/GRPO training module
│   ├── train.py             # Main GRPO training script
│   ├── environment.py       # Verifiers-compatible environment
│   ├── reward.py            # Reward computation (type check, compile, tests)
│   ├── config.py            # Training configuration
│   ├── data.py              # Dataset loading
│   └── logging.py           # Metrics logging utilities
├── sft/                     # Supervised fine-tuning module
│   ├── train.py             # SFT training with TRL's SFTTrainer
│   ├── config.py            # LoRA configuration
│   ├── data.py              # Dataset loading from HuggingFace
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
    └── bootstrap.sh         # Environment bootstrap script
```
