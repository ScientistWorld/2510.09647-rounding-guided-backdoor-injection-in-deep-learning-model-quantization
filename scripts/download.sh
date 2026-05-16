#!/bin/bash
# Download all data and models needed for this environment.
#
# Downloads required data. Estimated download: 170 MB compressed,
# about 350 MB including extracted files.

set -e

cd "$(dirname "$0")/.."

# CIFAR-10 canonical archive, verified by torchvision.
mkdir -p data/downloads/cifar-10
ARCHIVE="data/downloads/cifar-10-python.tar.gz"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

if [ ! -e "$ARCHIVE" ]; then
    curl --fail --location --retry 3 "$URL" --output "$ARCHIVE"
fi

if [ ! -d "data/downloads/cifar-10/cifar-10-batches-py" ]; then
    tar -xzf "$ARCHIVE" -C data/downloads/cifar-10
fi

echo "All downloads complete."
