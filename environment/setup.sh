#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system inside the container with a persistent
# overlay. Changes persist across job submissions without rebuilding.
#
# The nvidia/cuda container has PyTorch available via pip. This script
# installs additional packages needed for QURA.

set -e

# Install PyTorch with CUDA support and additional packages
pip3 install --no-cache-dir torch torchvision

# Install additional packages
pip3 install --no-cache-dir numpy scipy tqdm

# Ensure pip points to the right location
export PATH=/usr/local/bin:$PATH