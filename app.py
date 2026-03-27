import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load Model + Features
# -------------------------------
with open("models/lung_cancer_model.pkl", "rb") as f:
    model_package = pickle.load(f)

model = model_package["model"]
features = model_package["features"]

# Model metrics
metrics = {
    "accuracy": 0.91,
    "precision": 0.89,
    "recall": 1.00,
    "f1": 0.94,
    "roc_auc": 0.98
}

label_map = {
    0: "Lung Neuroendocrine Tumor",
    1: "Non-Small Cell Lung Cancer"
}

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="MetaboAI Lung Cancer Platform",
    layout="wide"
)

st.title("MetaboAI : Lung Cancer Subtype Classification Platform")

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Platform Modules")

module = st.sidebar.radio(
    "Select Module",
    [
        "Model Stats",
        "Upload Data",
        "AI Prediction",
        "Biomarker Importance",
        "Top Biomarkers",
        "Feature Heatmap"
    ]
)

# -------------------------------
# Download Template
# -------------------------------
st.sidebar.subheader("Input Template")

template = pd.DataFrame(columns=features)

st.sidebar.download_button(
    "Download Input Template",
    template.to_csv(index=False),
    "metabolomics_template.csv"
)

# -------------------------------
# Model Stats
# -------------------------------
if module == "Model Stats":

    st.title("Model Performance Dashboard")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
    col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
    col4.metric("F1 Score", f"{metrics['f1']*100:.2f}%")
    col5.metric("ROC-AUC", f"{metrics['roc_auc']*100:.2f}%")

# -------------------------------
# Upload Data
# -------------------------------
if module == "Upload Data":

    uploaded_file = st.file_uploader("Upload Metabolomics CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(data.head())
        st.session_state["data"] = data

# -------------------------------
# AI Prediction
# -------------------------------
if module == "AI Prediction":

    if "data" not in st.session_state:
        st.warning("Upload dataset first")

    else:
        data = st.session_state["data"]

        missing = [f for f in features if f not in data.columns]

        if missing:
            st.warning(f"Missing features filled with 0: {missing}")
            for col in missing:
                data[col] = 0

        data = data.reindex(columns=features)

        if st.button("Run Prediction"):

            prediction = model.predict(data)
            prob = model.predict_proba(data)

            pred_labels = [label_map[p] for p in prediction]

            results = pd.DataFrame({
                "Predicted_Type": pred_labels,
                "Cancer_Probability": prob[:,1]
            })

            st.subheader("Prediction Results")
            st.dataframe(results)

            st.metric(
                "Dominant Cancer Type",
                results["Predicted_Type"].mode()[0]
            )

            st.session_state["data_processed"] = data

# -------------------------------
# Biomarker Importance
# -------------------------------
if module == "Biomarker Importance":

    st.subheader("Biomarker Importance (Logistic Coefficients)")

    coefficients = model.named_steps['model'].coef_[0]

    feat_imp = pd.DataFrame({
        "Metabolite": features,
        "Importance": np.abs(coefficients)
    }).sort_values("Importance", ascending=False)

    st.dataframe(feat_imp.head(20))
    st.bar_chart(feat_imp.set_index("Metabolite").head(20))

# -------------------------------
# Top Biomarkers
# -------------------------------
if module == "Top Biomarkers":

    coefficients = model.named_steps['model'].coef_[0]

    feat_imp = pd.DataFrame({
        "Metabolite": features,
        "Importance": np.abs(coefficients)
    }).sort_values("Importance", ascending=False)

    top = feat_imp.head(20)

    st.subheader("Top Biomarkers")
    st.dataframe(top)
    st.bar_chart(top.set_index("Metabolite")["Importance"])


# -------------------------------
# Feature Heatmap With Class Annotation (Final Layout)
# -------------------------------
if module == "Feature Heatmap":

    if "data_processed" not in st.session_state:
        st.warning("Run prediction first")
    else:

        import seaborn as sns
        import matplotlib.pyplot as plt

        data = st.session_state["data_processed"]

        prediction = model.predict(data)
        pred_labels = [label_map[p] for p in prediction]

        coefficients = model.named_steps['model'].coef_[0]

        feat_imp = pd.DataFrame({
            "Metabolite": features,
            "Importance": np.abs(coefficients)
        })

        top_features = feat_imp.sort_values(
            "Importance", ascending=False
        ).head(20)["Metabolite"].tolist()

        heatmap_data = data[top_features].copy()

        # Z-score normalization
        heatmap_data = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()

        # Add class labels
        heatmap_data["Class"] = pred_labels
        heatmap_data = heatmap_data.sort_values("Class")

        class_labels = heatmap_data["Class"]
        heatmap_data = heatmap_data.drop("Class", axis=1)

        # Color mapping
        class_color_map = {
            "Non-Small Cell Lung Cancer": "#1f77b4",
            "Lung Neuroendocrine Tumor": "#d62728"
        }

        class_colors = class_labels.map(class_color_map)

        st.subheader("Clustered Heatmap of Top Biomarkers")

        g = sns.clustermap(
            heatmap_data.T,
            cmap="RdBu_r",
            col_colors=class_colors,
            figsize=(16, 10),
            xticklabels=False,
            yticklabels=True,
            dendrogram_ratio=(0.1, 0.15),
            cbar_pos=(0.02, 0.8, 0.02, 0.15)
        )

        # Adjust layout
        g.fig.subplots_adjust(left=0.25, right=0.9, top=0.92, bottom=0.05)

        # Add small legend text manually (top right)
        g.fig.text(
            0.88, 0.95,
            "Cancer Type",
            fontsize=10,
            fontweight="bold"
        )

        g.fig.text(
            0.88, 0.92,
            "Blue = Non-Small Cell Lung Cancer",
            fontsize=8
        )

        g.fig.text(
            0.88, 0.89,
            "Red = Lung Neuroendocrine Tumor",
            fontsize=8
        )

        st.pyplot(g.fig)