# Predicting Mental Health Treatment Seeking in Tech: A Machine Learning Approach

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Deep Learning](https://img.shields.io/badge/DL-TensorFlow%20%2F%20MLP-red.svg)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

## Project Overview
Mental health challenges are profoundly widespread within the technology sector, with **51% of tech professionals** reporting a diagnosed mental health condition. Despite high prevalence rates, a significant portion of the workforce refrains from seeking professional treatment due to systemic workplace stigma, lack of organizational support, or fear of negative career repercussions.

This project utilizes advanced machine learning pipelines to predict whether a technology worker will actively seek mental health treatment. By analyzing organizational structures, workplace culture indicators, personal history, and demographics, these predictive models identify key drivers of help-seeking behavior.

The ultimate objective is to provide actionable, data-driven insights that empower HR departments and executives to build healthier, more supportive corporate ecosystems.

---

## 🎯 Project Objectives
* **Predictive Classification:** Construct robust classification models to predict employee treatment-seeking behavior (`treatment_sought`).
* **Architecture Evaluation:** Compare the performance, data efficiency, and generalization caps of classical machine learning algorithms against multi-layer deep learning architectures.
* **Feature Explainability:** Uncover structural corporate determinants (e.g., policy communication, anonymity protections) that act as catalysts or barriers to mental health care access.

---

## Dataset Profile
* **Source:** Open Sourcing Mental Illness (OSMI) Mental Health in Tech Survey
* **Sample Size:** 1,433 respondents
* **Feature Dimensionality:** 63 original survey features
* **Target Feature:** `treatment_sought` (Binary: `0` = No, `1` = Yes)

---

## Data Engineering & Preparation Pipeline

### 1. Cleaning & Imputation
* **High-Null Filtering:** Structurally dropped features containing more than 80% missing responses to eliminate sparsity issues.
* **Categorical Imputation:** Handled missing data entries using local mode imputation across matching demographic bands.
* **Feature Text Standardization:** Mapped chaotic, free-form inputs (such as raw gender strings) into standardized structural classifications (`Male`, `Female`, `Other`).

### 2. Encoding & Dimensionality Expansion
* **One-Hot Encoding:** Categorical features were expanded via dummy variables, generating a high-dimensional feature space of **320 dense inputs**.
* **Boolean Normalization:** Converted structural boolean outputs cleanly to standard 0/1 integer states.

### 3. Resampling, Scaling, & Dimensionality Reduction
* **Socio-Demographic Scaling:** Applied standard Z-score normalizations to numerical variances like `age`.
* **Synthetic Over-sampling (SMOTE):** Leveraged Synthetic Minority Over-sampling Technique on the training partition to eliminate minority class under-representation artifacts.
* **Principal Component Analysis (PCA):** Compressed the high-dimensional 320 feature space into **10 orthogonal principal components**, successfully retaining **95% of the cumulative variance** for optimization within the Feedforward Neural Network.

---

## Model Architecture & Performance Evaluation

The modeling pipeline split the data using an 80/20 train-test allocation coupled with **5-Fold Stratified Cross-Validation** to ensure baseline assessment reliability.

### Model Comparison Summary

| Model Architecture | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Overfit Gap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **87.11%** | **0.87** | **0.88** | **0.87** | **-0.13%** |
| **Feedforward Neural Net (FNN)** | 86.76% | 0.86 | 0.86 | 0.86 | **0.04%** |
| **Logistic Regression** | 85.71% | 0.85 | 0.85 | 0.85 | 1.61% |
| **Gradient Boosting** | 85.71% | 0.85 | 0.85 | 0.85 | 8.95% |

### Deep Learning Architectural Blueprint (FNN)
```text
Input Layer (10 PCA Features) 
      │
      ▼
  Dense Layer (256 Neurons, ReLU) ──► Dropout (0.5) ──► L2 Decay
      │
      ▼
  Dense Layer (128 Neurons, ReLU) ──► Dropout (0.5)
      │
      ▼
  Dense Layer (64 Neurons, ReLU)
      │
      ▼
Output Layer (1 Neuron, Sigmoid) ──► Binary Cross-Entropy Loss
