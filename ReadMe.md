# Credit Risk Prediction Web App

![Streamlit App](https://img.shields.io/badge/Streamlit-App-green) ![Python Version](https://img.shields.io/badge/Python-3.13-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Dataset](#dataset)  
4. [Architecture & Model Pipeline](#architecture--model-pipeline)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Model Explainability](#model-explainability)  
8. [Folder Structure](#folder-structure)  
9. [Technologies Used](#technologies-used)  
10. [License](#license)  

---

## Project Overview

This project implements a **Customer Credit Risk Prediction** system using **XGBoost** and a **preprocessing pipeline**. Users can input customer details via a **Streamlit web interface** to get:  

- Predicted credit risk (`Low Risk / High Risk`)  
- Probability of default  
- Feature importance explanation  
- SHAP-based local and global interpretability  

The system is designed to assist **financial institutions** in making informed decisions while maintaining interpretability of machine learning predictions.  

---

## Features

### Tab 1 – Predict Risk
- Input customer details:
  - Age, Sex, Job, Housing, Savings, Checking Account, Credit Amount, Duration, Purpose  
- Predict risk: `High Risk (Bad)` or `Low Risk (Good)`  
- Show probability of risk  

### Tab 2 – Feature Importance
- Displays **overall feature importance** of the model  
- Top 10 most impactful features visualized using bar charts  

### Tab 3 – SHAP Explanation
- **Local explanation**: Waterfall plot showing feature contributions for a specific customer  
- **Global explanation**: Summary plot showing average feature importance across the dataset  
- Helps users understand **why the model made a prediction**  

---

## Dataset

- Source: German Credit Data CSV  
- Columns: 20+ features including numerical (`Age`, `Credit amount`, `Duration`, etc.) and categorical (`Sex`, `Housing`, `Purpose`, etc.)  
- Target variable: `Risk` (`good` = 0, `bad` = 1)  
- Missing values are handled:
  - `"Saving accounts"` and `"Checking account"` filled with `"unknown"`  

**Data preprocessing includes:**
- Standard scaling for numerical features  
- One-hot encoding for categorical features  

---

## Architecture & Model Pipeline

### Preprocessing
- Numerical features → `StandardScaler`  
- Categorical features → `OneHotEncoder(handle_unknown="ignore")`  
- Combined using `ColumnTransformer`  

### Model
- **XGBoost Classifier**
- Hyperparameter tuning using `RandomizedSearchCV`:
  - `n_estimators`, `max_depth`, `learning_rate`  
- Integrated into an **imbalanced pipeline** with `SMOTE` (if needed)  
- Final model saved using `joblib`  

### Explainability
- SHAP (`KernelExplainer`) for local and global feature contributions  
- Matplotlib-based plots for Streamlit compatibility  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Samuel-Hailemariam-Seifu/credit-risk-app.git
cd credit-risk-app
