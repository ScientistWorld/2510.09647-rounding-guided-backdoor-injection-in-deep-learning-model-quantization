#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system inside the container with a persistent
# overlay. Changes persist across job submissions without rebuilding.
#
# The nvidia/cuda container has PyTorch available via pip. This script
# installs additional packages needed for QURA.

set -e

# Install Python dependencies into the system environment.
uv pip install --system torch torchvision numpy scipy tqdm
