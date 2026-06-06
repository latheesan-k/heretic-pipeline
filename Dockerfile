FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
        tar && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    python3.11 -m pip install --upgrade pip setuptools wheel

RUN python3.11 -m pip install \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch torchvision

RUN git clone --depth=1 https://github.com/ggml-org/llama.cpp /llama.cpp-src && \
    python3.11 -m pip install \
        "gguf>=0.1.0" \
        "sentencepiece>=0.1.98,<0.3.0" \
        "protobuf>=4.21.0,<5.0.0"

RUN python3.11 -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        hf-transfer heretic-llm

RUN python3.11 -m pip install "kernels==0.14.0"

ARG LLAMA_BUILD=b9542
RUN curl -L "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/llama-${LLAMA_BUILD}-bin-ubuntu-x64.tar.gz" \
        -o /tmp/llama.tar.gz && \
    mkdir -p /llama.cpp && \
    tar -xzf /tmp/llama.tar.gz -C /llama.cpp && \
    rm /tmp/llama.tar.gz && \
    QUANTIZE="$(find /llama.cpp -name 'llama-quantize' | head -1)" && \
    echo "llama-quantize found at: ${QUANTIZE}" && \
    chmod +x "${QUANTIZE}" && \
    ln -s "${QUANTIZE}" /usr/local/bin/llama-quantize && \
    LIBDIR="$(dirname "${QUANTIZE}")" && \
    echo "${LIBDIR}" > /etc/ld.so.conf.d/llama.conf && ldconfig

RUN printf '#!/usr/bin/env bash\nexec python3.11 /llama.cpp-src/convert_hf_to_gguf.py "$@"\n' \
        > /usr/local/bin/convert-hf-to-gguf && \
    chmod +x /usr/local/bin/convert-hf-to-gguf

COPY run_heretic.py /usr/local/bin/run_heretic.py
RUN chmod +x /usr/local/bin/run_heretic.py

RUN python3.11 -c "from importlib.metadata import version; \
    print('torch        =', version('torch')); \
    print('transformers =', version('transformers')); \
    print('huggingface-hub =', version('huggingface-hub')); \
    print('kernels      =', version('kernels')); \
    print('heretic-llm  =', version('heretic-llm'))"

RUN mkdir -p /output
WORKDIR /workspace
