
# GUIDE: Running AAFG-SBIR in Google Colab

This guide shows step-by-step how to open and run the project in Google Colab.  
Use it for reviewers to quickly validate the code.

---

## 1. Open the Notebook
- After pushing this repo to GitHub, copy the link to `AAFG-SBIR-Colab.ipynb`.
- Go to [Google Colab](https://colab.research.google.com/) → **File → Open Notebook → GitHub** → paste the link.

📸 *Screenshot: Colab "File → Open Notebook → GitHub"*

---

## 2. Setup Repository
In the first cell you will see:

```python
GIT_REPO = "https://github.com/USERNAME/REPO.git"  # change to your repo
GIT_DIR  = ""  # leave empty
```

- Replace `USERNAME/REPO` with your actual GitHub repo link.  
- Run the cell ✅.

📸 *Screenshot: First cell edited with repo URL*

---

## 3. Install Requirements
Run:

```python
!python -m pip install -q -r requirements.txt
```

📸 *Screenshot: successful pip install output*

---

## 4. Check Environment
The cell will display Python version, Torch version, and GPU info.

📸 *Screenshot: CUDA available: True*

---

## 5. Create Tiny Dataset
This will generate a small set of sample images + CSVs for testing.

Output message:  
`Created tiny dataset under DATA/Sketchy`

📸 *Screenshot: folder DATA/Sketchy with files*

---

## 6. Unit Test (Forward Pass)
Run:

```bash
!python tests/test_forward.py
```

Expected output: `forward ok` ✅

📸 *Screenshot: terminal output forward ok*

---

## 7. Smoke Training Run
Run for 2 epochs:

```bash
!python scripts/train.py   --data_root DATA/Sketchy   --train_csv tiny_train.csv   --val_csv tiny_val.csv   --epochs 2 --batch_size 2
```

📸 *Screenshot: tqdm training bar with loss values*

---

## 8. Evaluate
Finally:

```bash
!python scripts/eval.py   --data_root DATA/Sketchy   --test_csv index_test.csv   --ckpt checkpoints/best.pt
```

Expected output:

```
{'mAP': 0.xxx, 'P@100': 0.xxx, 'P@200': 0.xxx}
```

📸 *Screenshot: metrics printed*

---

✅ With this guide, reviewers can open Colab, run cells, and see working results immediately.
