# Phase 2: Data Cleaning and EDA

## Data Cleaning
### A. Missing Values
```python
print(df.isnull().sum())

# If missing values exist
if df.isnull().sum().sum() > 0:
    df.fillna(df.median(numeric_only=True), inplace=True)
```
**Why important:** Missing values can bias results or break models. Median imputation is robust for skewed lab values.

### B. Remove Duplicates
```python
before = df.shape[0]
df.drop_duplicates(inplace=True)
print("Removed", before - df.shape[0], "duplicates")
```
**Why important:** Duplicates can inflate apparent signal and distort metrics.

### C. Encode Gender
```python
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
# Often: F -> 0, M -> 1 (verify with le_gender.classes_)
```
**Why important:** Models require numeric input; encoding keeps gender information.

### D. Encode Target
```python
le_class = LabelEncoder()
df['CLASS'] = le_class.fit_transform(df['CLASS'])
# Likely: N -> 0, P -> 1, Y -> 2
print(le_class.classes_)
```
**Why important:** Supervised models need numeric targets; label mapping must be documented.

---

## Exploratory Data Analysis (EDA)
**Purpose:** Understand distributions, detect outliers, evaluate class imbalance, and inspect correlations.

### A. Class Distribution
```python
sns.countplot(x='CLASS', data=df)
plt.title("Class Distribution")
plt.show()
```
**Insight:** Assess class imbalance; severe imbalance requires stratified split and possibly resampling.

### B. Correlation Heatmap
```python
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```
**Insight:** HbA1c is expected to correlate strongly with diabetes. Lipid relationships (TG, VLDL, Chol) should align medically.

### C. Distribution Plots
```python
features = ['HbA1c', 'BMI', 'Chol', 'TG']
for col in features:
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.show()
```
**Insight:** Skewness (e.g., TG) may suggest log-transform or robust scaling.

### D. Boxplots for Outliers
```python
for col in features:
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()
```
**Insight:** Identify extreme values that might be clinical outliers or data entry issues.

### E. Pairplot
```python
sns.pairplot(df, hue='CLASS')
plt.show()
```
**Insight:** Visual class separation hints at model feasibility; overlap between P and Y is common.

---

## Medical Interpretations
- **HbA1c:** Primary diabetes marker, should show strong separation across classes.
- **BMI & TG:** Metabolic risk markers; elevated in prediabetes/diabetes.
- **Chol/HDL/LDL:** Dyslipidemia patterns often appear in diabetic populations.

