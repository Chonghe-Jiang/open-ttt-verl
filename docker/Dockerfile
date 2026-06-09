ARG BASE_IMAGE=slimerl/slime:latest
FROM ${BASE_IMAGE}

ENV WORKSPACE=/root/workspace \
    REPO_DIR=/root/workspace/erdos \
    PYTHONUNBUFFERED=1 \
    PYTHONBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_XET=1

RUN python3 -m pip install --no-cache-dir --upgrade \
    "numpy<2" \
    "huggingface_hub[cli]" \
    hf_transfer \
    scipy \
    cvxpy

WORKDIR /root/workspace/erdos
