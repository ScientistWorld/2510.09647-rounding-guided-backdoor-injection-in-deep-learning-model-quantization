#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# The MCR PyTorch image already has PyTorch 2.5.0 + CUDA 12.4 installed.

set -e

# Activate conda environment if available
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate ptca 2>/dev/null || true
fi

# Install additional packages using pip
pip install --no-cache-dir numpy scipy tqdm
