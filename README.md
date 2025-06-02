Churn Prediction Model

### 🚦 1️⃣ Data Flow:

✅ Load input data → `val_df.csv`  
✅ Apply full pipeline:
- Feature engineering
- tenure bucket
- dummies + column alignment
- clustering
- calibrated model predictions
- risk bucketing

---

### 🏗️ 2️⃣ Modeling Architecture:

- **KMeans Clustering** → creates 2 behavioral cohorts:
    - **Cluster 0:** High-risk, short-tenure, high charges
    - **Cluster 1:** Long-tenure, lower-risk customers
- **Separate models for each cluster**:
    - **Cluster 0 → CatBoost Classifier + calibration**
    - **Cluster 1 → Logistic Regression + scaler + calibration**
- **Calibration**:
    - Ensures consistent probability outputs across clusters
- **Risk Bucketing**:
    - Divides final probabilities into:
        - Low
        - Moderate
        - High
        - Very High

---

## 🎨 Streamlit App Features

✅ Upload **val_df.csv**  
✅ Pipeline auto-runs:
- Preprocessing
- Clustering
- Cluster-specific predictions
- Risk bucketing

- <img width="601" alt="image" src="https://github.com/user-attachments/assets/d22b09fa-efa5-4a53-9d3a-ffe08881d1b7" />




