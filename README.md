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
**Note:** These are instructions only. No system changes were executed.

### 1) Install NVIDIA Driver
- Download and install the latest NVIDIA GPU driver for your card.
- Verify:
```bash
nvidia-smi
```

### 2) Install CUDA Toolkit
- Download CUDA Toolkit (12.x recommended) from NVIDIA.
- Verify:
```bash
nvcc --version
```

### 3) Optional Python GPU packages
For GPU acceleration with XGBoost, you can use:
```bash
pip install xgboost
```
Then configure:
```python
XGBClassifier(tree_method="gpu_hist", predictor="gpu_predictor")
```

If you want GPU-enabled PyTorch or CuPy, add:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install cupy-cuda12x
```

