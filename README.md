# AAFG-SBIR: Attention-Augmented Fine-Grained Sketch-Based Image Retrieval

Official implementation of **AAFG-SBIR** for our submission to *The Visual Computer (Springer)*.  
The model integrates **self-attention** (intra-modal) and **cross-attention** (inter-modal) to reduce the sketch–photo domain gap with ~8% FLOPs overhead while preserving interpretability.




##  Environment

- Python 3.10, PyTorch 2.2.1, CUDA 12.1
```bash
pip install -r requirements.txt



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


Training
python src/train.py --config configs/default.yaml


Demo Inference
python scripts/demo_infer.py --sketch path/to/sketch.png --gallery path/to/gallery_dir

requirements.txt
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

