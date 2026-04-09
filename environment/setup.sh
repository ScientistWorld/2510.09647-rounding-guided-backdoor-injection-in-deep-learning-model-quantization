#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# The base image already has PyTorch 2.5.0 + CUDA 12.4 pre-installed
# via conda at /opt/conda/envs/ptca. We only install additional packages.
#
# IMPORTANT: Install into the SYSTEM Python, not a virtual environment.
# Use: uv pip install --system <packages>

set -e

uv pip install --system \
    numpy \
    scipy \
    tqdm
