FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake tmux vim && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY env.yml /tmp/environment.yml
RUN conda env update -n base -f /tmp/environment.yml && \
    conda clean -aqy && \
    rm -rf /opt/miniconda/pkgs && \
    find / -type d -name __pycache__ -prune -exec rm -rf {} \; && \
    rm /tmp/environment.yml

RUN git clone https://github.com/nvidia/apex /tmp/apex && \
    pip install -v --disable-pip-version-check --no-cache-dir \
    --global-option="--cpp_ext" --global-option="--cuda_ext" \
    --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
    --global-option="--fast_multihead_attn"  /tmp/apex/ && \
    rm -rf /tmp/apex

