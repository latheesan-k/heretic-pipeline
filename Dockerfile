FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl python3 python3-pip tar && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.6.0 \
        torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir heretic-llm

RUN curl -L "https://github.com/ggml-org/llama.cpp/releases/download/b9528/llama-b9528-bin-ubuntu-x64.tar.gz" \
        -o /tmp/llama.tar.gz && \
    mkdir -p /llama.cpp && \
    tar -xzf /tmp/llama.tar.gz -C /llama.cpp && \
    find /llama.cpp -name "llama-quantize" -exec chmod +x {} \; && \
    rm /tmp/llama.tar.gz

RUN QUANTIZE=$(find /llama.cpp -name "llama-quantize" | head -1) && \
    echo "llama-quantize found at: $QUANTIZE" && \
    ln -s "$QUANTIZE" /usr/local/bin/llama-quantize

RUN git clone --depth=1 https://github.com/ggml-org/llama.cpp /llama.cpp-src && \
    pip install -r /llama.cpp-src/requirements/requirements-convert_hf_to_gguf.txt

RUN mkdir /output
WORKDIR /workspace