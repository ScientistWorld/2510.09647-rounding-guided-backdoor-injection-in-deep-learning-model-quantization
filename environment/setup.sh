#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# PyTorch 2.4.0 + CUDA 11.8 + Python 3.10 is pre-installed in the base image
# at /opt/conda/envs/ptca. We only install lightweight additional packages.
#
# IMPORTANT: Install into the SYSTEM Python, not a virtual environment.
# Use: uv pip install --system <packages>

set -e

uv pip install --system \
    numpy \
    scipy \
    tqdm
