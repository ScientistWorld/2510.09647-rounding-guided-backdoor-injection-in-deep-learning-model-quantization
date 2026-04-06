#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# Put ALL pip installs here, NOT in container.def.
# Only use container.def for the base Docker image and apt-get packages.
#
# IMPORTANT: Install into the SYSTEM Python, not a virtual environment.
# The overlay may be recreated on container rebuilds — a venv inside
# it will be lost and all subsequent jobs will fail.
# Use: uv pip install --system <packages>
# Do NOT create a venv (python -m venv, virtualenv, etc.).

set -e

uv pip install --system \
    torch==2.3.1 torchvision==0.18.1 \
    timm \
    scipy \
    matplotlib \
    numpy \
    pandas \
    scikit-learn \
    tqdm \
    pyyaml \
    easydict \
    transformers==4.20.0 \
    protobuf==3.20.3 \
    onnx==1.13.1 \
    prettytable \
    datasets

# Install mqbench from QURA repo
cd /home/user/QuRA_repo
uv pip install --system -e . 2>/dev/null || python3 setup.py develop 2>/dev/null || echo "MQBench setup skipped"
