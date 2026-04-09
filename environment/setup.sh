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
# Use: uv pip install --system <packages>

set -e

uv pip install --system \
    torch==2.3.1 \
    torchvision==0.18.1 \
    numpy \
    scipy \
    tqdm
