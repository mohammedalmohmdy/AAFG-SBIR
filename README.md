# AAFG-SBIR: Attention-Augmented Fine-Grained Sketch-Based Image Retrieval

This repository provides a clean, reproducible **PyTorch** implementation of the AAFG‑SBIR framework (self‑attention + cross‑modal attention over a shared ResNet‑50). It includes:

- Siamese‑style backbone with shared **ResNet‑50**
- **Self-attention** (intra‑modal) and **cross‑modal attention** modules
- **Triplet ranking** with **batch‑hard** mining
- **Deterministic** training, logging (**TensorBoard + CSV**), checkpointing
- Evaluation: **mAP** and **Precision@K**

> We do **not** redistribute datasets. Please follow official sources and place data as described below.

---

## 1) Environment

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional sanity:
```bash
python - <<'PY'
import torch, random, numpy as np
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
PY
```

## 2) Data layout

We support CSV indexes (relative paths). Create this structure—or symlink to your storage:

```
DATA/
  Sketchy/
    images/              # natural images
    sketches/            # hand-drawn sketches
    index_train.csv      # columns: path_sketch,path_image,label
    index_val.csv
    index_test.csv
  TUBerlin/
    images/
    sketches/
    index_train.csv
    index_val.csv
    index_test.csv
  QMUL-Shoe-V2/
    ...
  QMUL-Chair/
    ...
```

Each CSV row provides a **paired** (sketch, positive-photo) for the anchor/positive; negatives are mined **within-batch**.

You can start from the templates in `data_templates/` and edit the paths/labels to your copies.

## 3) Quick start (Sketchy example)

**Train**
```bash
python scripts/train.py   --data_root DATA/Sketchy   --train_csv index_train.csv   --val_csv index_val.csv   --log_dir runs/Sketchy   --epochs 60 --batch_size 32 --lr 1e-4 --margin 0.2   --embed_dim 512 --attn_reduction 1 --amp
```

**Evaluate**
```bash
python scripts/eval.py   --data_root DATA/Sketchy   --test_csv index_test.csv   --ckpt checkpoints/best.pt   --batch_size 64
```

**Smoke test (tiny subset)**
```bash
# create a tiny CSV (e.g., 100 rows) and run 2 epochs
python scripts/train.py --data_root DATA/Sketchy     --train_csv tiny_train.csv --val_csv tiny_val.csv     --epochs 2 --batch_size 8 --log_dir runs/smoke
```

## 4) Logs & artifacts
- TensorBoard: `tensorboard --logdir runs/Sketchy`
- CSV logs: `runs/Sketchy/metrics.csv`
- Checkpoints: `checkpoints/epoch_XX.pt` and `checkpoints/best.pt`

## 5) Reproducibility
We fix seeds and prefer deterministic ops where feasible. Exact bit‑wise determinism across driver/CUDA stacks may require pinning specific versions.

## 6) License
MIT (see `LICENSE`). For academic review and research use.
