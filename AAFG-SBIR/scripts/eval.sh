#!/usr/bin/env bash
set -e
CFG=$1
SEED=$2
TAG=$3

CKPT="outputs/checkpoints/$(basename $CFG .yaml)_${TAG}_seed${SEED}.pt"

python -m src.engine.eval   --config "$CFG"   --seed "$SEED"   --tag "$TAG"   --ckpt "$CKPT"   --save_csv "outputs/results_csv/$(basename $CFG .yaml)_${TAG}_seed${SEED}.csv"   | tee "logs/eval_$(basename $CFG .yaml)_${TAG}_seed${SEED}.txt"
