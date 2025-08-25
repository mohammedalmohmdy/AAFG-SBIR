## Hardware & Software
GPU: <A100 40GB> | CUDA 12.1 | Driver <xxx.xx>
PyTorch 2.3.1 | TorchVision 0.18.1 | Commit: <HASH>

## Runs
- Sketchy: seeds {0,1,2}. Aggregates in outputs/results_csv/sketchy_agg.json.
- TU-Berlin: seeds {0,1,2}. Aggregates in outputs/results_csv/tu_berlin_agg.json.
- QMUL-Shoe-V2: seeds {0,1,2}. Aggregates in outputs/results_csv/qmul_shoe_v2_agg.json.
- QMUL-Chair: seeds {0,1,2}. Aggregates in outputs/results_csv/qmul_chair_agg.json.

## Wall-clock
Sketchy train per seed: ~<H:mm> on <GPU>, batch=64, epochs=100.
