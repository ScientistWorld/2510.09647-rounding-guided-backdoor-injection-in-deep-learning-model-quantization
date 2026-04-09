#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# Ubuntu 20.04 base image with Python 3.8 from apt.
# PyTorch 2.5.1 installed from PyTorch wheel index (CUDA 11.8 support).

set -e

pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118 \
    numpy \
    scipy \
    tqdm
