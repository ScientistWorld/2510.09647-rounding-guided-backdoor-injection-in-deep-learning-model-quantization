#!/bin/bash
# Download all data and models needed for this environment.
#
# Estimated total download size: ~170 MB
# Estimated disk usage after extraction: ~200 MB
#
# CIFAR-10 is copied from the shared datasets directory (/home/user/shared/datasets).
# The shared directory already contains the extracted CIFAR-10 data.
# We create a symlink to the workspace data directory.

set -e

cd "$(dirname "$0")"

# CIFAR-10 is in the shared datasets directory
if [ -d "/home/user/shared/datasets/cifar-10" ]; then
    echo "CIFAR-10 found in shared directory. Creating symlink..."
    mkdir -p /home/user/data
    if [ ! -L "/home/user/data/cifar-10" ]; then
        ln -sf /home/user/shared/datasets/cifar-10 /home/user/data/cifar-10
    fi
    echo "CIFAR-10 ready at /home/user/data/cifar-10"
else
    echo "CIFAR-10 not in shared directory. Downloading..."
    mkdir -p /home/user/data/cifar-10
    wget -nc -q https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz \
        -P /home/user/data/cifar-10/
    tar -xzf /home/user/data/cifar-10/cifar-10-python.tar.gz \
        -C /home/user/data/cifar-10/
    rm /home/user/data/cifar-10/cifar-10-python.tar.gz
    echo "CIFAR-10 downloaded."
fi

echo "All downloads complete."
