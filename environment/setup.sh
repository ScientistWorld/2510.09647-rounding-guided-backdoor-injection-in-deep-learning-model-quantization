#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# Ubuntu 22.04 with CUDA 12.4.
# PyTorch 2.5.1 installed from PyTorch wheel index (CUDA 12.4 support).

set -e

# Install uv if not available
if ! command -v uv &> /dev/null; then
    pip3 install --no-cache-dir uv
fi

# Install PyTorch and dependencies using uv
uv pip install --system \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    numpy \
    scipy \
    tqdm
