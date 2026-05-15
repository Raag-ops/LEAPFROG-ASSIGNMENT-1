# Phase 3: Feature Engineering and Preprocessing

## Feature Engineering
### A. Cardiovascular Risk Ratio
```python
# Example engineered feature
if 'Chol' in df.columns and 'HDL' in df.columns:
    df['Chol_HDL_Ratio'] = df['Chol'] / df['HDL']
```
**Why important:** Chol/HDL ratio is a standard cardiovascular risk indicator and may improve predictive power.

### B. Optional: BMI Categories
```python
# Optional categorical feature
bins = [0, 18.5, 25, 30, np.inf]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df['BMI_Category'] = pd.cut(df['BMI'], bins=bins, labels=labels)
```
**Why important:** BMI categories can capture non-linear risk trends.

---

## Data Preprocessing
### A. Split Features and Target
```python
X = df.drop('CLASS', axis=1)
y = df['CLASS']
```

### B. Standardization
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
**Why important:** Standardization prevents large-scale features from dominating, improves SVM/KMeans/PCA performance.

---

## Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```
**Why important:** Stratified split preserves class distribution in train/test sets.

