# Phase 1: Problem Setup and Data Understanding

## Goal
Build two modeling tracks:
- **Classification (supervised):** Predict `CLASS` (N/P/Y) from medical features.
- **Clustering (unsupervised):** Group patients without labels and compare clusters to true classes.

## Dataset Summary
- **Target:** `CLASS`
  - `N` = Non-diabetic
  - `P` = Prediabetic
  - `Y` = Diabetic
- **Key Features:** `Gender`, `AGE`, `Urea`, `Cr`, `HbA1c`, `Chol`, `TG`, `HDL`, `LDL`, `VLDL`, `BMI`
- **Medical context:**
  - `HbA1c` is a primary glycemic marker and often the strongest predictor of diabetes.
  - `BMI` and `TG` capture metabolic risk; dyslipidemia patterns are common in prediabetes/diabetes.

## Environment and Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
# Optional (bonus):
# pip install lightgbm catboost
```

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
```

## Load Dataset
```python
df = pd.read_csv("dataset/Dataset of Diabetes .csv")

# Basic inspection
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
```

## Data Understanding (Explain Features in Notebook)
| Feature | Meaning |
|---|---|
| HbA1c | Main diabetes indicator |
| BMI | Obesity level |
| TG | Triglycerides |
| HDL | Good cholesterol |
| LDL | Bad cholesterol |
| Chol | Total cholesterol |
| Urea | Kidney function marker |
| Cr | Creatinine, kidney function marker |
| AGE | Age in years |
| Gender | Sex |

**Why this matters:**
- Explaining features elevates report quality and shows domain understanding.
- It sets expectations for how medical markers should behave across classes.

## Expected Workflow Outline
1. Data Cleaning
2. EDA
3. Feature Engineering
4. Preprocessing
5. Train-Test Split
6. Classification Models
7. Clustering Models
8. Evaluation & Comparison
9. Error Analysis
10. Interpretation
11. Future Improvements
12. Conclusion

