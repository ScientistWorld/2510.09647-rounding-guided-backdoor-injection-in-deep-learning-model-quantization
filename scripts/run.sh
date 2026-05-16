#!/bin/bash
# Job script submitted via action.yaml.
#
# Compute nodes have no internet; scripts/download.sh must have populated
# data/downloads before this runs.

set -e

cd /home/user

if [ -d /home/user/pkgs ]; then
    export PYTHONPATH="/home/user/pkgs:$PYTHONPATH"
fi

if [ ! -d /home/user/data/downloads/cifar-10/cifar-10-batches-py ]; then
    echo "Missing CIFAR-10 at /home/user/data/downloads/cifar-10/cifar-10-batches-py."
    echo "Run scripts/download.sh on the login node before submitting."
    exit 2
fi

rm -f /home/user/scoring/scores_train.json /home/user/scoring/scores_test.json

bash /home/user/scripts/evaluate_train.sh
bash /home/user/scripts/evaluate_test.sh
