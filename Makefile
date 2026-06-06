IMAGE_NAME   := heretic
OUTPUT_DIR   := $(HOME)/output
HF_TOKEN     ?=

# Extract short model name from full HF path (e.g. google/gemma-4-E2B-it -> gemma-4-E2B-it)
MODEL        := $(word 2, $(MAKECMDGOALS))
MODEL_SLUG   := $(notdir $(MODEL))

DECENSORED_DIR  := $(OUTPUT_DIR)/decensored
F16_GGUF        := $(OUTPUT_DIR)/$(MODEL_SLUG)-heretic-f16.gguf
FINAL_GGUF      := $(OUTPUT_DIR)/$(MODEL_SLUG)-heretic-q4_0.gguf

DOCKER_RUN := docker run --rm \
	--gpus all \
	-v $(OUTPUT_DIR):/output \
	-e HF_HOME=/output/hf_cache \
	$(if $(HF_TOKEN),-e HF_TOKEN=$(HF_TOKEN),) \
	$(IMAGE_NAME)

# ─── Colours ──────────────────────────────────────────────────────────────────
BOLD  := \033[1m
CYAN  := \033[1;36m
GREEN := \033[1;32m
RED   := \033[1;31m
RESET := \033[0m

.PHONY: heretic $(MODEL) check-model build-image decensor convert quantize clean-intermediate

# ─── Entry point ──────────────────────────────────────────────────────────────
heretic: check-model build-image decensor convert quantize clean-intermediate
	@echo ""
	@echo "$(GREEN)$(BOLD)✔ Done!$(RESET)"
	@echo "$(GREEN)Output: $(FINAL_GGUF)$(RESET)"

# Absorb the model argument as a no-op target so make doesn't complain
$(MODEL):
	@true

# ─── Guard: require a model argument ──────────────────────────────────────────
check-model:
	@if [ -z "$(MODEL)" ]; then \
		echo "$(RED)Error: no model specified.$(RESET)"; \
		echo "Usage: make heretic <hf-org/model-name>"; \
		exit 1; \
	fi
	@echo "$(CYAN)$(BOLD)» Model   :$(RESET) $(MODEL)"
	@echo "$(CYAN)$(BOLD)» Output  :$(RESET) $(OUTPUT_DIR)"
	@mkdir -p $(OUTPUT_DIR)

# ─── Step 0: build Docker image if not present ────────────────────────────────
build-image:
	@if docker image inspect $(IMAGE_NAME) > /dev/null 2>&1; then \
		echo "$(CYAN)$(BOLD)» [0/3] Docker image '$(IMAGE_NAME)' already exists, skipping build.$(RESET)"; \
	else \
		echo "$(CYAN)$(BOLD)» [0/3] Building Docker image '$(IMAGE_NAME)'...$(RESET)"; \
		docker build -t $(IMAGE_NAME) . || { \
			echo "$(RED)Docker build failed.$(RESET)"; exit 1; \
		}; \
		echo "$(GREEN)Image built successfully.$(RESET)"; \
	fi

# ─── Step 1: decensor with Heretic ────────────────────────────────────────────
decensor:
	@echo ""
	@echo "$(CYAN)$(BOLD)» [1/3] Decensoring $(MODEL)...$(RESET)"
	@$(DOCKER_RUN) \
		heretic $(MODEL) --output-dir /output/decensored --save-model || { \
		echo "$(RED)Heretic decensor step failed.$(RESET)"; exit 1; \
	}
	@echo "$(GREEN)Decensored model saved to $(DECENSORED_DIR)$(RESET)"

# ─── Step 2: convert safetensors -> f16 GGUF ──────────────────────────────────
convert:
	@echo ""
	@echo "$(CYAN)$(BOLD)» [2/3] Converting to f16 GGUF...$(RESET)"
	@$(DOCKER_RUN) \
		python /llama.cpp/convert_hf_to_gguf.py \
			/output/decensored \
			--outfile /output/$(MODEL_SLUG)-heretic-f16.gguf \
			--outtype f16 || { \
		echo "$(RED)GGUF conversion failed.$(RESET)"; exit 1; \
	}
	@echo "$(GREEN)f16 GGUF written.$(RESET)"

# ─── Step 3: quantize f16 GGUF -> q4_0 ───────────────────────────────────────
quantize:
	@echo ""
	@echo "$(CYAN)$(BOLD)» [3/3] Quantizing to q4_0...$(RESET)"
	@$(DOCKER_RUN) \
		llama-quantize \
			/output/$(MODEL_SLUG)-heretic-f16.gguf \
			/output/$(MODEL_SLUG)-heretic-q4_0.gguf \
			q4_0 || { \
		echo "$(RED)Quantization failed.$(RESET)"; exit 1; \
	}
	@echo "$(GREEN)Quantized model written.$(RESET)"

# ─── Cleanup: remove intermediate f16 GGUF ────────────────────────────────────
clean-intermediate:
	@echo ""
	@echo "$(CYAN)Removing intermediate f16 GGUF...$(RESET)"
	@rm -f $(F16_GGUF)