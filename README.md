# MetaboAI – Lung Cancer Subtype Classification Using Metabolomics

MetaboAI is a machine learning platform for lung cancer subtype classification using metabolomics data.  
The platform predicts cancer subtype, identifies important metabolite biomarkers, and visualizes metabolic patterns using clustering heatmaps through an interactive Streamlit web application.

---

## Project Overview

This project applies machine learning to metabolomics data to classify lung cancer subtypes:

- Non-Small Cell Lung Cancer
- Lung Neuroendocrine Tumor

The workflow includes:
- Data preprocessing
- Machine learning model training
- Model evaluation
- Biomarker identification
- Heatmap visualization
- Streamlit web application deployment

---

## Features

- Upload metabolomics CSV data
- Predict lung cancer subtype
- Show prediction probabilities
- Identify important metabolite biomarkers
- Visualize top biomarkers
- Cluster heatmap with cancer subtype annotation
- Model performance dashboard
- Download input template CSV
- Interactive machine learning dashboard

---

## Machine Learning Model

Model used:
- Logistic Regression
- StandardScaler
- Stratified K-Fold Cross Validation
- Hyperparameter tuning using GridSearchCV

### Model Performance

| Metric | Score |
|-------|------|
| Accuracy | 0.91 |
| Precision | 0.89 |
| Recall | 1.00 |
| F1 Score | 0.94 |
| ROC-AUC | 0.98 |

---

## Project Structure
MetaboAI-LungCancer-Classification/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│ lung_cancer_model.pkl
│
├── data/
│ example_input.csv
│
├── figures/
│ heatmap.png
│
└── notebook/
Lung_cancer_binary_Class.ipynb

---

## Input Data Format

Upload a CSV file with metabolite names as columns and samples as rows.


The column names must match the training dataset metabolite names.

---

## How to Run the App Locally

1. Clone the repository:
git clone git clone https://github.com/kirankumar88/MetaboAI-LungCancer-Classification.git

2. Navigate to project folder:
cd MetaboAI-LungCancer-Classification

3. Install requirements:
pip install -r requirements.txt

4. Run Streamlit app:
streamlit run app.py


---

 Docker
- AWS / GCP / Azure

---

## Workflow

---

## Deployment link

This application is deployed using:
- Streamlit Cloud - https://metaboai-lungcancer-classification-5n327xz9pwnvtkpzkqn5eq.streamlit.app/

---

## Workflow
Metabolomics Data
↓
Feature Processing
↓
Machine Learning Model
↓
Cancer Subtype Prediction
↓
Biomarker Identification
↓
Heatmap Visualization
↓
Streamlit Web Application


---

## Applications

- Cancer subtype classification
- Metabolomics biomarker discovery
- Precision oncology research
- Machine learning in bioinformatics
- Clinical metabolomics analysis

---

## Author

Kiran Kumar  
Bioinformatics | Machine Learning | Metabolomics | Genomics

---

## License

This project is for research and educational purposes.
