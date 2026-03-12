UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
  NIX_CMD := nix develop --impure .\#cuda
else
  NIX_CMD := nix develop
endif

.PHONY: shell rlvr-train sft-train eval dashboard grpo-train test lint fmt deps merge-adapter llama-server llama-server-hf help

shell:
	$(NIX_CMD)

rlvr-train:
	./scripts/run-rlvr-training.py

sft-train:
	./scripts/run-sft.sh

dashboard:
	uv run python dashboard/server.py

vllm-server:
	@test -n "$(model)" || (echo "Usage: make vllm-server model=<model_id>" >&2; exit 1)
	./scripts/start_vllm_server.sh -m "$(model)"

llama-server:
	@test -n "$(model)" || (echo "Usage: make llama-server model=<path_to_gguf>" >&2; exit 1)
	llama-server -m "$(model)" $(if $(port),--port $(port),) $(LLAMA_ARGS)

llama-server-hf:
	./scripts/start_llama_server.sh $(if $(model),--model $(model),) $(if $(quant),--quant $(quant),) $(if $(port),--port $(port),)

eval:
	@test -n "$(model)" || (echo "Usage: make eval model=<model_name> [limit=<n>]" >&2; exit 1)
	OPENAI_MODEL="$(model)" uv run python -m eval.eval $(if $(limit),--limit $(limit),)

merge-adapter:
	@test -n "$(path)" || (echo "Usage: make merge-adapter path=<adapter-path>" >&2; exit 1)
	uv run scripts/merge_adapter.py $(path)

test:
	uv run pytest -v

lint:
	@files=$$(git diff --name-only | grep '\.py$$'); \
	if [ -n "$$files" ]; then \
		uv run ruff check $$files; \
	else \
		echo "No changed Python files to lint"; \
	fi

fmt:
	uv run ruff format .

deps:
	uv sync --frozen

help:
	@printf "Targets:\n"
	@printf "  shell           enter nix shell (cuda on Linux)\n"
	@printf "  deps            install dependencies via uv sync\n"
	@printf "  rlvr-train      run RLVR training wrapper\n"
	@printf "  sft-train       run SFT training wrapper\n"
	@printf "  eval            run eval.eval with model=<name> [limit=<n>]\n"
	@printf "  merge-adapter   merge LoRA adapter with path=<adapter-path>\n"
	@printf "  dashboard       run dashboard server\n"
	@printf "  vllm-server     start vLLM server with model=<model_id>\n"
	@printf "  llama-server    start llama.cpp server with model=<path_to_gguf> [port=<n>] [LLAMA_ARGS=...]\n"
	@printf "  llama-server-hf download/convert HF model and start llama-server [model=<id>] [quant=<type>] [port=<n>]\n"
	@printf "  test            run test suite with pytest\n"
	@printf "  lint            run ruff check\n"
	@printf "  fmt             run ruff format\n"
