
AAFG-SBIR: Attention-Augmented Framework for Fine-Grained Sketch-Based Image Retrieval

This repository contains the official implementation of the paper:
“Bridging the Sketch–Photo Domain Gap: An Attention-Augmented Framework for Fine-Grained Image Retrieval”
Mohammed A. S. Al-Mohamadi and Prabhakar C. J.
Submitted to The Visual Computer (Springer, 2025)

Overview:
AAFG-SBIR is an attention-augmented deep learning framework for Fine-Grained Sketch-Based Image Retrieval (FG-SBIR). 
It integrates self-attention and cross-attention modules to enhance feature alignment between sketches and photos while maintaining interpretability and computational efficiency.

Key Features:
- Dual Attention Design — Combines self-attention and cross-attention for better sketch–photo correspondence.
- Lightweight Efficiency — Adds only ~8% computational overhead compared to ResNet-50 baseline.
- Explainability — Generates interpretable attention heatmaps and attention intensity distributions.
- Cross-Dataset Generalization — Validated on four major FG-SBIR benchmarks (Sketchy, TU-Berlin, QMUL-Shoe-V2, QMUL-Chair).


Installation:
git clone https://github.com/mohammedalmohmdy/AAFG-SBIR.git
cd AAFG-SBIR
pip install -r requirements.txt

Requirements: Python >= 3.10, PyTorch >= 2.2.1, CUDA >= 12.1, NumPy, Pillow, tqdm, scikit-learn.

Training:
bash scripts/train.sh
or
python src/train.py --config configs/default.yaml --dataset sketchy

Evaluation:
bash scripts/eval.sh
or
python src/test.py --dataset tuberlin --checkpoint results/checkpoints/aafg_sbir_best.pth

Visualization Demo:
python scripts/demo_infer.py --input sample_sketch.png --topk 5

Data Availability:
### 📂 Datasets

- **ShoeV2 / ChairV2**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/1frltfiEd9ymnODZFHYrbg741kfys1rq1/view)

- **Sketchy**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/11GAr0jrtowTnR3otyQbNMSLPeHyvecdP/view)

- **TU-Berlin**  
  [TU-Berlin Official Website](https://www.tu-berlin.de/)  
  [Google Drive Download](https://drive.google.com/file/d/12VV40j5Nf4hNBfFy0AhYEtql1OjwXAUC/view)


Citation:
If you use this code, please cite:

  title   = {Bridging the Sketch–Photo Domain Gap: An Attention-Augmented Framework for Fine-Grained Image Retrieval},
  
  author  = {Mohammed A. S. Al-Mohamadi and Prabhakar C. J.},
  
  journal = {The Visual Computer},
  year    = {2025}
}

License:
This project is released under the MIT License.

Contact:
almohmdy30@gmail.com
GitHub: https://github.com/mohammedalmohmdy

