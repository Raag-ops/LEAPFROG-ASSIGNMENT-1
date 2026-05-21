# Diabetes Machine Learning Project

This project uses Jupyter notebooks to prepare a diabetes dataset, run exploratory data analysis, train classification and clustering models, and evaluate the saved results.

## Project Structure

```text
.
|-- dataset/
|   `-- Dataset of Diabetes .csv
|-- processed_arrays/
|   `-- Saved NumPy arrays created by dataPrep.ipynb
|-- models/
|   `-- Saved trained models and clustering labels
|-- images/
|   `-- Generated plots and reports
|-- dataPrep.ipynb
|-- edaBook.ipynb
|-- classificationBook.ipynb
|-- clusteringBook.ipynb
|-- evalClassModel.ipynb
|-- evalClusteringModel.ipynb
`-- requirements.txt
```

## Prerequisites

- Python 3.10 or newer
- pip
- Conda
- Jupyter Notebook or JupyterLab

This project includes GPU-related dependencies such as `cupy-cuda12x`, `torch`, `torchvision`, and `torchaudio`. If you do not have a CUDA 12 compatible GPU setup, installing `cupy-cuda12x` may fail. In that case, remove or comment out `cupy-cuda12x` from `requirements.txt` and install the remaining dependencies.

## Step-by-Step Setup

### 1. Open the Project Folder

```powershell
cd C:\Users\mahar\PycharmProjects\PythonProject2
```

### 2. Create and Activate the Conda Environment

Create a Conda environment for this project:

```powershell
conda create -n diabetes-ml python=3.10
```

Activate it:

```powershell
conda activate diabetes-ml
```

### 3. Install Dependencies in the Conda Environment

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Alternative: Create a Python Virtual Environment

Use this only if you are not using Conda.

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Register the Jupyter Kernel

```powershell
python -m ipykernel install --user --name diabetes-ml --display-name "Python (diabetes-ml)"
```

### 5. Start Jupyter

To use JupyterLab:

```powershell
jupyter lab
```

Or to use classic Jupyter Notebook:

```powershell
jupyter notebook
```

When Jupyter opens in the browser, select the `Python (diabetes-ml)` kernel for each notebook.

## Running the Project

Run the notebooks in this order.

### 1. Prepare the Data

Open and run:

```text
dataPrep.ipynb
```

This notebook loads:

```text
dataset/Dataset of Diabetes .csv
```

It cleans and standardizes the dataset, engineers features, splits the data, and saves arrays into:

```text
processed_arrays/
```

The generated files include `X_train.npy`, `X_val.npy`, `X_test.npy`, `y_train.npy`, `y_val.npy`, `y_test.npy`, `X_full.npy`, `y_full.npy`, `feature_names.npy`, and `target_classes.npy`.

### 2. Run Exploratory Data Analysis

Open and run:

```text
edaBook.ipynb
```

This notebook loads the original CSV file, creates plots, and writes generated visual outputs to:

```text
images/
```

It also creates the Sweetviz report:

```text
images/eda_sweetviz_report.html
```

### 3. Train Classification Models

Open and run:

```text
classificationBook.ipynb
```

This notebook loads the processed arrays and trains:

- Random Forest
- XGBoost
- SVM

It saves trained models into:

```text
models/
```

Expected model outputs include:

```text
models/random_forest.joblib
models/xgboost_model.json
models/svm_model.joblib
```

### 4. Train Clustering Models

Open and run:

```text
clusteringBook.ipynb
```

This notebook loads the full processed dataset and trains or evaluates:

- K-Means
- Hierarchical clustering
- Gaussian Mixture Model

It saves outputs into:

```text
models/
```

Expected clustering outputs include:

```text
models/kmeans_model.joblib
models/hierarchical_labels.npy
models/gmm_labels.npy
```

### 5. Evaluate Classification Models

Open and run:

```text
evalClassModel.ipynb
```

This notebook loads the saved classification models and processed arrays, then generates classification metrics, confusion matrices, ROC curves, precision-recall curves, and comparison plots.

### 6. Evaluate Clustering Models

Open and run:

```text
evalClusteringModel.ipynb
```

This notebook loads saved clustering outputs and calculates clustering metrics such as silhouette score, Davies-Bouldin score, and Calinski-Harabasz score.

## Quick Run Checklist

```text
1. Create and activate the Conda environment
2. Install dependencies
3. Register the Jupyter kernel
4. Start Jupyter
5. Run dataPrep.ipynb
6. Run edaBook.ipynb
7. Run classificationBook.ipynb
8. Run clusteringBook.ipynb
9. Run evalClassModel.ipynb
10. Run evalClusteringModel.ipynb
```

## Notes

- The `processed_arrays/` and `models/` folders already contain generated outputs. You can use them directly for evaluation, but rerun `dataPrep.ipynb`, `classificationBook.ipynb`, and `clusteringBook.ipynb` if you want to regenerate everything from the raw CSV.
- Keep the dataset filename unchanged unless you also update the notebook paths:

```text
dataset/Dataset of Diabetes .csv
```

- Most generated plots and reports are saved in `images/`.

## Troubleshooting

### Jupyter does not show the project environment

Run this again while the Conda environment is active:

```powershell
python -m ipykernel install --user --name diabetes-ml --display-name "Python (diabetes-ml)"
```

Then restart Jupyter and select `Python (diabetes-ml)`.

### Dependency installation fails on `cupy-cuda12x`

If your machine does not have a compatible CUDA 12 setup, edit `requirements.txt`, remove or comment out:

```text
cupy-cuda12x
```

Then rerun:

```powershell
pip install -r requirements.txt
```

### Dataset file not found

Make sure the dataset exists at:

```text
dataset/Dataset of Diabetes .csv
```

The filename contains spaces and a space before `.csv`, so keep the name exactly the same.
