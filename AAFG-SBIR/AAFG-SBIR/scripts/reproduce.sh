#!/usr/bin/env bash
set -e
SEEDS=(0 1 2)

for S in "${SEEDS[@]}"; do
  bash scripts/train.sh configs/sketchy.yaml $S baseline_triplet model.self_attention=false model.cross_attention=false
  bash scripts/eval.sh  configs/sketchy.yaml $S baseline_triplet

  bash scripts/train.sh configs/sketchy.yaml $S sa_only model.self_attention=true model.cross_attention=false
  bash scripts/eval.sh  configs/sketchy.yaml $S sa_only

  bash scripts/train.sh configs/sketchy.yaml $S ca_only model.self_attention=false model.cross_attention=true
  bash scripts/eval.sh  configs/sketchy.yaml $S ca_only

  bash scripts/train.sh configs/sketchy.yaml $S ours_sa_ca model.self_attention=true model.cross_attention=true
  bash scripts/eval.sh  configs/sketchy.yaml $S ours_sa_ca
done
python scripts/aggregate_results.py --glob "outputs/results_csv/sketchy_*_seed*.csv"   --out outputs/results_csv/sketchy_agg.json

# TU-Berlin
for S in "${SEEDS[@]}"; do
  bash scripts/train.sh configs/tu_berlin.yaml $S ours_sa_ca model.self_attention=true model.cross_attention=true
  bash scripts/eval.sh  configs/tu_berlin.yaml $S ours_sa_ca
done
python scripts/aggregate_results.py --glob "outputs/results_csv/tu_berlin_*_seed*.csv"   --out outputs/results_csv/tu_berlin_agg.json

# QMUL-Shoe-V2
for S in "${SEEDS[@]}"; do
  bash scripts/train.sh configs/qmul_shoe_v2.yaml $S baseline_triplet model.self_attention=false model.cross_attention=false
  bash scripts/eval.sh  configs/qmul_shoe_v2.yaml $S baseline_triplet

  bash scripts/train.sh configs/qmul_shoe_v2.yaml $S sa_only model.self_attention=true model.cross_attention=false
  bash scripts/eval.sh  configs/qmul_shoe_v2.yaml $S sa_only

  bash scripts/train.sh configs/qmul_shoe_v2.yaml $S ca_only model.self_attention=false model.cross_attention=true
  bash scripts/eval.sh  configs/qmul_shoe_v2.yaml $S ca_only

  bash scripts/train.sh configs/qmul_shoe_v2.yaml $S ours_sa_ca model.self_attention=true model.cross_attention=true
  bash scripts/eval.sh  configs/qmul_shoe_v2.yaml $S ours_sa_ca
done
python scripts/aggregate_results.py --glob "outputs/results_csv/qmul_shoe_v2_*_seed*.csv"   --out outputs/results_csv/qmul_shoe_v2_agg.json

# QMUL-Chair
for S in "${SEEDS[@]}"; do
  bash scripts/train.sh configs/qmul_chair.yaml $S ours_sa_ca model.self_attention=true model.cross_attention=true
  bash scripts/eval.sh  configs/qmul_chair.yaml $S ours_sa_ca
done
python scripts/aggregate_results.py --glob "outputs/results_csv/qmul_chair_*_seed*.csv"   --out outputs/results_csv/qmul_chair_agg.json
