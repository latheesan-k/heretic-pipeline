FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl python3 python3-pip tar && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

RUN pip install -U heretic-llm

# Download pre-built llama.cpp CUDA binary (hardcoded release — update tag as needed)
RUN curl -L "https://github.com/ggml-org/llama.cpp/releases/download/b9528/llama-b9528-bin-ubuntu-x64.tar.gz" \
        -o /tmp/llama.tar.gz && \
    mkdir -p /llama.cpp && \
    tar -xzf /tmp/llama.tar.gz -C /llama.cpp && \
    chmod +x /llama.cpp/build/bin/* && \
    rm /tmp/llama.tar.gz

# Shallow clone just for the Python conversion script
RUN git clone --depth=1 https://github.com/ggml-org/llama.cpp /llama.cpp-src && \
    pip install -r /llama.cpp-src/requirements/requirements-convert_hf_to_gguf.txt

RUN mkdir /output
WORKDIR /workspace