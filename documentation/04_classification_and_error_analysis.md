# Phase 4: Classification Models, Evaluation, and Error Analysis

## Classification Models
### A. Logistic Regression
```python
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
```

### B. Random Forest
```python
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
```

### C. Support Vector Machine (RBF)
```python
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
```

### D. XGBoost (Recommended)
```python
xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=3
)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
```

---

## Evaluation Function
```python
def evaluate_model(y_test, y_pred, name):
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(name)
    plt.show()
```

```python
evaluate_model(y_test, y_pred_lr, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_svm, "SVM")
evaluate_model(y_test, y_pred_xgb, "XGBoost")
```

**Why important:** Multiple models reveal robustness and reduce risk of overfitting to one approach.

---

## Feature Importance (Interpretation)
```python
importance = rf.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feat_imp)
plt.show()
```
**Expected top features:** `HbA1c`, `BMI`, `TG`, `Chol`.

**Medical interpretation:** These markers represent glucose control and metabolic syndrome risk, aligning with diabetes pathology.

---

## Error Analysis
### A. Misclassified Samples
```python
errors = X_test[y_test != y_pred_rf]
print("Misclassified samples:", errors.shape[0])
```
**Insight:** Misclassifications often cluster between P and Y due to overlapping metabolic profiles.

### B. Confusion Between P and Y
**Medical reasoning:** Prediabetes and diabetes have gradual transitions; HbA1c and BMI overlap is expected.

### C. Outlier Impact
**Potential issues:** Extreme lipid or creatinine values can skew decision boundaries; consider outlier handling if needed.

