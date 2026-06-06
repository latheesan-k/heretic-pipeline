FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl python3 python3-pip unzip && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

RUN pip install -U heretic-llm

RUN RELEASE=$(curl -s https://api.github.com/repos/ggerganov/llama.cpp/releases/latest | grep '"tag_name"' | cut -d'"' -f4) && \
    echo "Downloading llama.cpp $RELEASE" && \
    curl -L "https://github.com/ggerganov/llama.cpp/releases/download/${RELEASE}/llama-${RELEASE}-bin-ubuntu-x64.zip" \
        -o /tmp/llama.zip && \
    unzip /tmp/llama.zip -d /llama.cpp && \
    chmod +x /llama.cpp/build/bin/* && \
    rm /tmp/llama.zip

# Python scripts for conversion (no build needed, just the repo scripts)
RUN pip install gguf numpy

RUN mkdir /output
WORKDIR /workspace