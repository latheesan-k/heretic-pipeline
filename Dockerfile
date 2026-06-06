FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl build-essential cmake python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Make python3 the default
RUN ln -s /usr/bin/python3 /usr/bin/python

# PyTorch
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Heretic
RUN pip install -U heretic-llm

# llama.cpp with CUDA support
RUN git clone https://github.com/ggerganov/llama.cpp /llama.cpp && \
    cd /llama.cpp && \
    pip install -r requirements.txt && \
    cmake -B build -DGGML_CUDA=ON && \
    cmake --build build --config Release -j$(nproc)

RUN mkdir /output
WORKDIR /workspace