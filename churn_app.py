# app.py — Full Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.graph_objects as go
import os
from sklearn.preprocessing import LabelEncoder

# ---- Load Artifacts ----
# REMOVE the decorator:
# @st.cache_resource
def load_artifacts():
    base_path = os.path.dirname(os.path.abspath(__file__))
    artifacts_path = os.path.join(base_path, "artifacts")

    with open(os.path.join(artifacts_path, "label_encoders.pkl"), 'rb') as f:
        feature_columns = pickle.load(f)

    kmeans_model = joblib.load(os.path.join(artifacts_path, "kmeans_model.pkl"))

    with open(os.path.join(artifacts_path, "calibrated_model_cluster_01_CatBoost.pkl"), 'rb') as f:
        calibrated_cat = pickle.load(f)

    with open(os.path.join(artifacts_path, "calibrated_model_cluster_2_LR.pkl"), 'rb') as f:
        calibrated_lr = pickle.load(f)

    with open(os.path.join(artifacts_path, "scaler_cluster_2.pkl"), 'rb') as f:
        scaler_cluster_2 = pickle.load(f)

    return feature_columns, kmeans_model, calibrated_cat, calibrated_lr, scaler_cluster_2



# ---- Streamlit App ----
st.title("🚀 Churn Prediction App")
st.write("Upload your `val_df.csv` → Get churn risk prediction & customer insights!")

uploaded_file = st.file_uploader("Upload val_df.csv", type=["csv"])

if uploaded_file is not None:
    # ---- Load Artifacts ----
    feature_columns, kmeans_model, calibrated_cat, calibrated_lr, scaler_cluster_2 = load_artifacts()

    # ---- Load CSV ----
    val_df = pd.read_csv(uploaded_file)
    st.success(f"Uploaded {uploaded_file.name} ✅")
    st.write("### Preview of uploaded data", val_df.head(3))

    # ---- Preprocess ----
    val_df['tenure_bucket'] = np.where(val_df['tenure'] <= 6, 1,
                            np.where(val_df['tenure'] <= 20, 2,
                            np.where(val_df['tenure'] <= 50, 3,
                            4)))

    val_df['TotalCharges'] = np.where((val_df['TotalCharges'] == " ") | (val_df['TotalCharges'].isna()), 0, val_df['TotalCharges'])
    val_df['TotalCharges'] = pd.to_numeric(val_df['TotalCharges'])

    val_df_processed = val_df.drop(columns=['customerID', 'Churn', 'tenure'], errors='ignore')

    top_feature_names = ['Contract', 'TotalCharges', 'MonthlyCharges', 'OnlineSecurity', 'tenure_bucket',
                         'TechSupport', 'PaymentMethod', 'InternetService', 'MultipleLines', 'PaperlessBilling']

    val_df_processed = val_df_processed[top_feature_names]

    
    val_df_dummies = val_df_processed.copy()
    cate_cols = val_df_dummies.select_dtypes(include='object').columns
    
    le = LabelEncoder()
    for col in cate_cols:
        val_df_dummies[col] = le.fit_transform(val_df_dummies[col])
        
    
    X_val_full = val_df_dummies.reindex(columns=top_feature_names, fill_value=0)


    val_df['cohort_cluster'] = kmeans_model.predict(X_val_full)

    # ---- Predict probabilities ----
    probas = []
    X_val_scaled_cluster_2 = scaler_cluster_2.transform(X_val_full)

    for idx, row in X_val_full.iterrows():
        cluster = val_df.loc[idx, 'cohort_cluster']
        row_array = row.values.reshape(1, -1)

        if cluster == 0:
            proba = calibrated_cat.predict_proba(row_array)[0,1]
        else:
            row_scaled = X_val_scaled_cluster_2[idx, :].reshape(1, -1)
            proba = calibrated_lr.predict_proba(row_scaled)[0,1]

        probas.append(proba)

    val_df['pred_proba'] = probas

    # ---- Risk bucketing ----
    val_df['risk_bucket'] = pd.qcut(val_df['pred_proba'], q=4, labels=['Low', 'Moderate', 'High', 'Very High'])

    # ---- Final Output ----
    final_df = val_df[['customerID', 'cohort_cluster', 'pred_proba', 'risk_bucket']]
    st.write("### Final Predictions", final_df)

    # ---- Gauge Chart ----
    st.write("### 📈 Customer Risk Gauge")
    selected_customer = st.selectbox("Select a customerID:", final_df['customerID'].unique())

    selected_row = final_df[final_df['customerID'] == selected_customer]
    selected_proba = float(selected_row['pred_proba'].values[0])
    selected_risk = selected_row['risk_bucket'].values[0]

    # Plotly Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = selected_proba * 100,
        title = {'text': f"Churn Probability (%)\nRisk: {selected_risk}"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if selected_proba > 0.5 else "green"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
