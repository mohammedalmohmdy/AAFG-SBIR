# AAFG-SBIR: Attention-Augmented Fine-Grained Sketch-Based Image Retrieval

Official implementation of **AAFG-SBIR** for our submission to *The Visual Computer (Springer)*.  
The model integrates **self-attention** (intra-modal) and **cross-attention** (inter-modal) to reduce the sketch–photo domain gap with ~8% FLOPs overhead while preserving interpretability.


##  📁 Project Structure

AAFG-SBIR/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ configs/
│  └─ default.yaml
├─ src/
│  ├─ datasets/
│  │  ├─ sketchy.py
│  │  ├─ tuberlin.py
│  │  ├─ qmul_shoe_v2.py
│  │  └─ qmul_chair.py
│  ├─ models/
│  │  ├─ backbone_resnet50.py
│  │  ├─ attention_sa.py          # self-attention
│  │  ├─ attention_ca.py          # cross-attention
│  │  └─ aafg_sbir.py             # build_model()
│  ├─ losses/
│  │  └─ triplet.py               # margin=0.2
│  ├─ utils/
│  │  ├─ seed.py
│  │  ├─ metrics.py               # mAP, P@100
│  │  ├─ preprocess.py
│  │  └─ train_utils.py
│  ├─ train.py                    # main training entry
│  └─ test.py                     # main evaluation entry
├─ scripts/
│  ├─ train.sh
│  ├─ eval.sh
│  ├─ demo_infer.py               # single sketch → top-K results
│  └─ compute_flops.py            # optional
├─ figures/
│  ├─ perf_vs_flops.png
│  └─ attention_examples/
└─ results/                        # (ignored by git)
   ├─ checkpoints/
   └─ eval/


##  Environment

- Python 3.10, PyTorch 2.2.1, CUDA 12.1
```bash


pip install -r requirements.txt

torch==2.2.1
torchvision==0.17.1
numpy
scipy
pyyaml
tqdm
opencv-python
pillow
matplotlib
scikit-learn


## 📂 Datasets

- **ShoeV2 / ChairV2**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/1frltfiEd9ymnODZFHYrbg741kfys1rq1/view)

- **Sketchy**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/11GAr0jrtowTnR3otyQbNMSLPeHyvecdP/view)

- **TU-Berlin**  
  [TU-Berlin Official Website](https://www.tu-berlin.de/)  
  [Google Drive Download](https://drive.google.com/file/d/12VV40j5Nf4hNBfFy0AhYEtql1OjwXAUC/view)


##  Training
python src/train.py --config configs/default.yaml


##  Demo Inference
python scripts/demo_infer.py --sketch path/to/sketch.png --gallery path/to/gallery_dir



##  cff-version: 1.2.0
title: "AAFG-SBIR: Attention-Augmented Fine-Grained Sketch-Based Image Retrieval"
authors:
  - family-names: "Al-Mohamadi"
    given-names: "Mohammed A.S."
  - family-names: "C. J"
    given-names: ".Prabhakar "
repository-code: "https://github.com/mohammedalmohmdy/AAFG-SBIR"
message: "If you use this code, please cite the associated manuscript submitted to The Visual Computer (Springer)."
