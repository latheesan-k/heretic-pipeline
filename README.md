# heretic-pipeline

A fully automated pipeline for decensoring and quantizing HuggingFace language models using [Heretic](https://github.com/p-e-w/heretic) and [llama.cpp](https://github.com/ggerganov/llama.cpp), packaged in Docker.

Given any HuggingFace model ID, the pipeline will:

1. Decensor the model using Heretic (directional ablation / abliteration)
2. Convert the output to GGUF (f16)
3. Quantize to `q4_0` GGUF
4. Clean up intermediates and report the final file path

---

## Requirements

| Requirement | Notes |
|---|---|
| Ubuntu Linux | Tested on Ubuntu 22.04+ |
| NVIDIA GPU | Minimum ~16GB VRAM for 7–8B models, 24GB (e.g. RTX 4090) recommended |
| NVIDIA Driver | 555+ (the image uses the CUDA 12.6 runtime via `nvidia/cuda:12.6.2-devel`) |
| Docker | With `nvidia-container-toolkit` installed |
| `make` | Standard on Ubuntu (`sudo apt install make`) |

### Install nvidia-container-toolkit (if not already installed)

```bash
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify Docker can see your GPU
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

---

## Setup

Clone this repo (or copy the `Dockerfile` and `Makefile` into a directory):

```
your-dir/
├── Dockerfile
├── Makefile
└── run_heretic.py   # non-interactive driver for Heretic (required)
```

No other setup is needed. The Docker image is built automatically on first use.

> **Note:** Heretic is normally an interactive program — after optimizing it
> opens menus to pick a trial and save the model. `run_heretic.py` drives the
> real Heretic pipeline non-interactively, automatically selecting the
> fewest-refusals trial and saving the decensored model. This is why there is
> no `--save-model` flag in the `decensor` step.

---

## Usage

```bash
make heretic <hf-org/model-name>
```

### Example

```bash
make heretic google/gemma-4-E2B-it-qat-q4_0-unquantized
```

### With a HuggingFace token (required for gated models)

```bash
HF_TOKEN=hf_your_token_here make heretic google/gemma-4-E2B-it-qat-q4_0-unquantized
```

You can also export it so you don't have to repeat it:

```bash
export HF_TOKEN=hf_your_token_here
make heretic google/gemma-4-E2B-it-qat-q4_0-unquantized
```

---

## Output

All output is written to `~/output/`:

```
~/output/
├── hf_cache/                              # HuggingFace model cache (persists between runs)
├── decensored/                            # Decensored model in safetensors format
└── <model-slug>-heretic-q4_0.gguf        # ✅ Final quantized model
```

The intermediate f16 GGUF is automatically deleted after quantization.

---

## Pipeline steps

```
[0/3] Build Docker image (skipped if already built)
        ↓
[1/3] Decensor model with Heretic
        ↓
[2/3] Convert safetensors → f16 GGUF
        ↓
[3/3] Quantize f16 GGUF → q4_0 GGUF
        ↓
      Remove intermediate f16 GGUF
        ↓
      ✔ Done — print path to final file
```

---

## Performance

Tested on an RTX 4090 (24GB VRAM):

| Model size | Approximate runtime |
|---|---|
| 2–4B | ~15–20 min |
| 7–8B | ~40–50 min |
| 13B | ~90 min |

Most of the time is spent in the Heretic optimization step. Conversion and quantization are fast (a few minutes).

---

## Rebuilding the Docker image

The image is built automatically if it doesn't exist. To force a rebuild (e.g. after updating the `Dockerfile`):

```bash
docker rmi -f heretic
make heretic <model>
```

---

## Troubleshooting

**`nvidia-smi` works but Docker can't see the GPU**
```bash
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Out of VRAM on large models**
Pass extra Heretic flags via `HERETIC_ARGS`, e.g.:
```bash
HERETIC_ARGS="--quantization bnb_4bit" make heretic <model>
```
This reduces VRAM usage during the abliteration pass at a small quality cost. You can also lower the trial count for a faster run, e.g. `HERETIC_ARGS="--n-trials 50"`.

**HuggingFace 401 / access denied**
The model is gated. Pass your token: `HF_TOKEN=hf_xxx make heretic <model>`

**`torch.cuda.is_available()` is False / runs on CPU**
This usually means a dependency downgraded PyTorch to a CPU build. The
`Dockerfile` installs the llama.cpp converter's non-torch dependencies first and
installs Heretic **last** so its GPU `torch` and version pins win. If you edit
the `Dockerfile`, keep that ordering. Verify with:
```bash
docker run --rm --gpus all heretic python3.11 -c "import torch; print(torch.cuda.is_available())"
```

---

## Credits

- [Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann — abliteration / censorship removal
- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov — GGUF conversion and quantization