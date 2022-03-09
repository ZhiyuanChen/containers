FROM nvcr.io/nvidia/pytorch:22.02-py3

ENV LD_LIBRARY_PATH=/opt/hpcx/nccl_rdma_sharp_plugin/lib:$LD_LIBRARY_PATH
ENV NCCL_ASYNC_ERROR_HANDLING=1
ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

##############################################################################
# Blobfuse required packages for 20.04
##############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pkg-config cmake libcurl4-gnutls-dev libgnutls28-dev uuid-dev \
    libgcrypt20-dev libboost-all-dev gcc g++ tmux vim && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install mpi4py (be sure to link existing /opt/openmpi-xxx)
RUN CC=mpicc MPICC=mpicc pip install mpi4py --no-binary mpi4py

# Build apex
RUN git clone https://github.com/nvidia/apex /tmp/apex && \
    pip install -v --disable-pip-version-check --no-cache-dir \
    --global-option="--cpp_ext" --global-option="--cuda_ext" \
    --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
    --global-option="--fast_multihead_attn"  /tmp/apex/ && \
    rm -rf /tmp/apex

# Install accelerate and other packages
RUN pip install git+https://github.com/huggingface/accelerate.git einops pytorch-lightning

USER root
WORKDIR /home/aiscuser
LABEL maintainer="Zhiyuan Chen <this@zyc.ai>"
#
# Information about this image
#
ENV BASE_IMAGE=pytorch1.11-cuda11.6-cudnn8.3:Alice
