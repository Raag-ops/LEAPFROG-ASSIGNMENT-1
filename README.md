# Diabetes EDA Notebook

This project provides a professional, interview-ready EDA notebook for a multi-class diabetes dataset.

## Quick Start
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Open the notebook:
```bash
jupyter notebook
```

Notebook file:
- `edaBook.ipynb`

## Dataset
Expected path:
- `dataset/Dataset of Diabetes .csv`

If your dataset name or path differs, update the `DATA_PATH` in `edaBook.ipynb`.

---

## CUDA / GPU Setup (Windows)
**Detected on this machine (via `nvidia-smi` and `nvcc`):**
- GPU: NVIDIA GeForce RTX 3060 (6GB)
- Driver: 596.36
- CUDA Driver API Version: 13.2
- CUDA Toolkit (nvcc): 13.1

### Verified commands
```bash
nvidia-smi
nvcc --version
```

### Python GPU packages (optional but recommended)
```bash
pip install xgboost
```
Then configure:
```python
XGBClassifier(tree_method="gpu_hist", predictor="gpu_predictor")
```

Optional GPU stacks (if you need deep learning or GPU arrays):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install cupy-cuda12x
```

### Installed GPU Python stack (verified)
- PyTorch: 2.6.0+cu124 (CUDA 12.4)
- TorchVision: 0.21.0+cu124
- TorchAudio: 2.6.0+cu124
- CuPy: 14.0.1 (CUDA 12.x)

Verification command:
```bash
python -c "import torch, torchvision, torchaudio, cupy; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('cupy', cupy.__version__)"
```
