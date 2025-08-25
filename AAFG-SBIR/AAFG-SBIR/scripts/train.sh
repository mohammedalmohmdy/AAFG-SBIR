#!/usr/bin/env bash
set -e
CFG=$1
SEED=$2
TAG=${3:-run}
shift 3 || true
OVERRIDES="$@"

python -m src.engine.train   --config "$CFG"   --seed "$SEED"   --tag "$TAG"   $OVERRIDES   | tee "logs/train_$(basename $CFG .yaml)_${TAG}_seed${SEED}.txt"
