#!/bin/bash
# Download all data and models needed for this environment.
#
# CIFAR-10 data is in the shared datasets directory. We link it to the
# workspace data directory. Total size: ~170 MB.
#
# The data is in Python pickle format (.pkl embedded in .batch files).

set -e

cd "$(dirname "$0")/.."

# CIFAR-10: link from shared datasets directory
if [ -d "/home/user/shared/datasets/cifar-10" ]; then
    echo "CIFAR-10 found in shared directory. Creating symlink..."
    mkdir -p /home/user/data
    ln -sf /home/user/shared/datasets/cifar-10 /home/user/data/cifar-10
    echo "CIFAR-10 ready at /home/user/data/cifar-10"
else
    echo "ERROR: CIFAR-10 not found in shared directory"
    exit 1
fi

echo "All downloads complete."
