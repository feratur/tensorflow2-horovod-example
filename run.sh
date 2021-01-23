#!/usr/bin/env bash
set -euxo pipefail

apt update && apt install -y cmake openmpi-bin

HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir scipy scikit-learn horovod[tensorflow]

NUM_WORKERS=3

cd $(dirname "${BASH_SOURCE[0]}")

python3 load_data.py $NUM_WORKERS

horovodrun -np $NUM_WORKERS -H localhost:$NUM_WORKERS python3 hvd_train.py
