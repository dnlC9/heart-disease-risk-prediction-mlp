
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Neural%20Network%20from%20Scratch-013243?logo=numpy&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-blueviolet)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Project-Production%20Ready-success)

![Cross Validation](https://img.shields.io/badge/Evaluation-K--Fold%20CV-blue)
![ROC AUC](https://img.shields.io/badge/Metric-ROC--AUC-important)
![Regularization](https://img.shields.io/badge/Regularization-Dropout%20%7C%20LR%20Decay-orange)
![Model Serving](https://img.shields.io/badge/Model-Export%20%26%20Inference-9cf)


# 🫀 Heart Disease Risk Prediction — Neural Network from Scratch

This project presents an **end-to-end Machine Learning pipeline** for predicting the risk of heart disease using a **fully custom neural network implemented from scratch**, without relying on high-level deep learning frameworks (e.g. TensorFlow, PyTorch).

Beyond model training, the project focuses on **professional ML practices**: data cleaning, model evaluation, regularization, export/import of trained parameters, and an **interactive Streamlit dashboard** designed for real-world usage and stakeholder communication.

---

## 🎯 Project Objectives

- Build a neural network **from first principles** (forward pass, backpropagation, optimization).
- Apply **professional data cleaning** techniques (hidden missing values, imputation, outliers).
- Evaluate the model rigorously using **cross-validation and ROC-AUC**.
- Prevent overfitting using **Dropout and learning-rate decay**.
- Export the trained model and reuse it for **real-time inference**.
- Deliver an **interactive, production-style Streamlit application** with:
  
  - data exploration,
  - model diagnostics,
  - risk prediction,
  - what-if analysis.

---

## 📁 Dataset

- **Source:** UCI Heart Disease Dataset (via Kaggle)
- **Samples:** 303 patients
- **Features:** 13 clinical variables
- **Target:** Binary classification
    
  - `0` → No heart disease  
  - `1` → Heart disease present

Key features include:

- Age, sex
- Chest pain type (`cp`)
- Resting blood pressure (`trestbps`)
- Cholesterol (`chol`)
- Maximum heart rate (`thalach`)
- Exercise-induced angina (`exang`)
- ST depression (`oldpeak`)
- Number of major vessels (`ca`)
- Thalassemia (`thal`)

---

## 🧹 Data Cleaning & Preparation

Professional cleaning steps were applied:

- Detection of **hidden missing values** (invalid encodings such as `thal=0`, `ca=4`)
- Conversion of incorrect values to `NaN`
- Imputation using **mode** for categorical-like features
- Outlier inspection using **IQR analysis**
- Final dataset validation (`df.info`, `df.describe`)

The same cleaning logic is reused during inference to ensure **training–serving consistency**.

---

## 🧠 Neural Network Architecture

Custom implementation with NumPy:

Input (13 features)
↓
Dense Layer (16 neurons)
↓
ReLU activation
↓
Dropout (regularization)
↓
Dense Layer (2 neurons)
↓
Softmax activation

### Training Details
- Loss: **Categorical Cross-Entropy**
- Optimizer: **Adam with learning-rate decay**
- Regularization: **Dropout**
- Epochs: 500
- Train/Validation split with **K-Fold Cross-Validation (K=5)**

---

## 📈 Model Performance

### Learning Curves
- Stable convergence
- No significant overfitting thanks to Dropout

### ROC-AUC (Cross-Validation)
- Consistently high AUC across folds
- Strong discriminative power between positive and negative classes

> The evaluation artifacts (learning curves and ROC data) are stored and reused directly in the Streamlit application.

---

## 💾 Model Export & Inference

- Trained weights and biases are exported to a **JSON checkpoint**
- A dedicated inference module:
  - reconstructs the network architecture,
  - loads parameters,
  - exposes clean `predict` and `predict_proba` APIs
- No retraining is required for deployment or visualization

---

## 📊 Streamlit Dashboard

The project culminates in a **multi-page Streamlit application**, structured as follows:

### 1️⃣ Dataset Overview
- Clean dataset preview
- Statistical summaries
- Feature dictionary
- High-level metrics (records, features, prevalence)

### 2️⃣ Model Training & Validation
- Learning curves (train vs validation)
- ROC-AUC curves per fold
- Explanation of model stability and generalization

### 3️⃣ Data Exploration
- Interactive filters (age, sex, cholesterol, etc.)
- Dynamic Plotly visualizations (scatter, histogram, box, bar)
- Exploratory analysis similar to real analytics dashboards

### 4️⃣ Prediction & What-If Analysis
- Patient profile form
- Real-time risk prediction
- Clear probability output
- Local **what-if analysis**:
  - age variation
  - cholesterol normalization
  - exercise-induced angina removal
  - ST depression reduction

This replicates how ML models are communicated to **non-technical stakeholders**.

---

## 🚀 How to Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
