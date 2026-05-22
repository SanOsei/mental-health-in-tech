# Predicting Mental Health Treatment Seeking in Tech: A Machine Learning Approach

**Authors:** Osei Sandra & Anim Priscilla  
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Deep Learning](https://img.shields.io/badge/DL-MLPClassifier--Tanh-red.svg)](https://scikit-learn.org/)

## 1. Project Overview
Mental health remains a critical yet often overlooked issue in the technology industry. Despite fast-paced work environments and inherent structural stress, many tech professionals hesitate to seek professional care due to systemic workplace stigma, lack of organizational support, and fear of negative career repercussions. **Approximately 51% of tech professionals** have been diagnosed with a mental health condition at some point in their career.

### Problem Statement
While many tech workers actively struggle with mental health issues, they frequently choose not to seek treatment. This creates an organizational blind spot: companies and HR departments cannot easily identify who needs support or what systemic factors block care access. This project implements machine learning and deep learning architectures to predict treatment-seeking behavior based on an employee's personal history, workplace culture, and organizational protections.

###  Project Goals
1. **Predictive Classification:** Construct robust predictive models to classify whether tech workers will seek mental health treatment.
2. **Architecture Evaluation:** Compare baseline classical machine learning models with advanced deep learning approaches to demonstrate exactly when complex techniques add real value.
3. **Actionable HR Insights:** Uncover corporate environmental factors that drive or hinder help-seeking behavior to inform policy changes.

---

##  2. Data Engineering & Preprocessing Pipeline
* **Dataset Profile:** OSMI Mental Health in Tech Survey (1,433 respondents, 63 original raw features).
* **Target Feature:** `treatment_sought` (Binary: `0` = No, `1` = Yes).
* **Column Standardization:** Restructured long, complex survey questions into short, interpretable algorithmic variable names (e.g., mapping raw text queries to fields like `anonymity_protected`, `employer_formal_mh_discussion`).
* **Sparsity & Missing Values:** Completely dropped features containing more than **80% missing data**. Applied local **mode imputation** for remaining missing categorical values.
* **Gender Harmonization:** Standardized chaotic, free-form text inputs from the survey's open gender field into clean, structured groups (`Male`, `Female`, `Other`).
* **High-Dimensional Expansion:** One-hot encoded all categorical attributes (including variable Likert scales), resulting in a wide feature space of **320 dense inputs**.
* **Scaling & Partitioning:** Scaled numerical variations (`age`) using `StandardScaler` and enforced a strict **80/20 train-test split** using stratified partitioning to prevent class distribution shift.
* **Validation Setup:** Implemented a robust **5-Fold Stratified $K$-Fold Cross-Validation** layer to evaluate generalizability across all baseline models.

---

##  3. Baseline Classical Machine Learning Models

Three distinct classical machine learning architectures were selected based on the specific survey dynamics of our dataset:

1. **Logistic Regression (Tuned):** Selected to extract highly interpretable coefficients showing exactly how specific environmental factors mathematically scale odds ratios.
2. **Random Forest (Tuned):** Selected to naturally process mixed categorical/ordinal types and map complex, non-linear interactions across workplace variables without manual feature engineering.
3. **Gradient Boosting (Tuned):** Selected to inherently counter the dataset's slight class imbalance (58% seeking treatment vs. 42% not) by iteratively weighting misclassified instances.

###  The Impact of Hyperparameter Tuning
Initial raw baselines showed major bottlenecks—specifically, the un-tuned **Random Forest memorized training partitions completely ($100\%$ Train Acc)** but dropped significantly to **$83.28\%$ on the test set**, signaling an alarming **$16.72\%$ overfitting gap**. 

To resolve this, systematic hyperparameter optimization was executed utilizing `GridSearchCV` and `RandomizedSearchCV` (tuning trees, max depth, `ccp_alpha` pruning scales, sub-sampling rates, and learning constraints). Tuning successfully reined in the variance, transforming the overfitted architectures into stable, highly generalized predictors.

---

## 4. Deep Learning Architecture (Feedforward Neural Network)

To demonstrate when advanced models add true business value, a custom **Multi-Layer Perceptron (MLP)** was integrated alongside specialized pre-neural pipeline components.

###  Architectural Blueprint
* **Dimensionality Reduction (PCA):** Compressed the wide 320 feature space down to **10 principal components**. This removed noise and bypassed the curse of dimensionality while successfully **retaining ~95% of the cumulative variance**.
* **Class Balancing (SMOTE):** Applied Synthetic Minority Over-Sampling Technique on the training partition to protect minority-class feature distributions.
* **Layer Topology:** Enforced a three-layer fully connected architecture:
  $$\text{Input (10 PCA Components)} \longrightarrow \text{Dense (256)} \longrightarrow \text{Dense (128)} \longrightarrow \text{Dense (64)} \longrightarrow \text{Output (1, Sigmoid)}$$
* **Regularization & Optimization:** Configured with a `tanh` activation function, **0.5 Dropout factors** after hidden layers, and an **L2 weight decay penalty of 0.0001**. Optimization was driven by the **Adam optimizer** (initial learning rate of 0.001, batch size of 32) backed by early stopping with a patience threshold of 10 epochs.

---

##  5. Final Model Performance Matrix

The following comprehensive evaluation matrix contrasts our baseline classical implementations against the tuned deep learning model:

| Model Architecture | Train Acc | Test Acc | Precision | Recall | F1-Score | CV Score | Overfit Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | 86.98% | **87.11%** | 0.8655 | 0.8780 | 0.8685 | 86.10% | **-0.13%** |
| **Feedforward Neural Net** | 86.80% | **86.76%** | — | — | — | **86.37%** | **0.04%** |
| **Gradient Boosting (Tuned)** | 90.38% | 86.76% | 0.8609 | 0.8665 | 0.8633 | 86.32% | 3.63% |
| **Logistic Regression (Tuned)** | 86.89% | 85.71% | 0.8501 | 0.8563 | 0.8527 | 86.31% | 1.17% |

###  Model Selection Rationale
* **For Real-World Deployment:** The **Feedforward Neural Network** is the preferred candidate. While its test accuracy ($86.76\%$) sits slightly below the tuned Random Forest, its **exceptionally small overfitting gap ($0.04\%$)** and highly consistent cross-validation mean score ($86.37\%$) prove that the PCA + Neural Network pipeline generalizes best to entirely unseen data patterns.
* **For Explanatory Analytics:** If an organization requires direct transparent audits of feature weights, **Logistic Regression** or **Gradient Boosting** are recommended due to their low complexity and structural interpretability.

---

##  6. Strategic Key Insights
* **Culture Wins Over Rules:** Simply putting a mental health policy on an intranet page does not alter employee behavior. However, companies that host **active, open corporate wellness discussions** see a massive statistical drop in care access barriers.
* **Anonymity Protections:** A perceived threat to data privacy locks down help-seeking pathways. Clearly communicated, structurally verified anonymity protections act as a major baseline catalyst for treatment tracking.
* **Environmental Covariates:** An employee's personal/family history of mental health challenges remains the single strongest baseline demographic predictor, though organizational features like company scale, remote status, and direct supervisor sentiment heavily influence final choices.

---

##  7. Suggestions & Recommendations

### Technical Extensions
1. **Structured Survey Engineering:** Future industry surveys must replace open-text data inputs with forced multiple-choice options to eradicate blank entries and sparse data pools.
2. **Volume Expansion:** Deep learning models require massive scale to truly outperform tree ensembles. Future iterations should scale data collection targets to **8,000–10,000 entries**.
3. **Model Explainability:** Integrate black-box interpretation frameworks like **SHAP (Shapley Additive exPlanations)** or **LIME** directly over the PCA-neural pipeline to translate the principal components back into human-readable corporate risk vectors.

### Organizational Action
* **Proactive Resource Engines:** Technology enterprises should design data-driven, privacy-preserving recommendation systems integrated with internal HR portals. This safe framework can anonymously surface tailored mental health tools to workers based on matched organizational risk factors without exposing individual identities.

---

##  8. Project Learnings & Tasks

### Key Project Takeaways
* **Data Preprocessing at Scale:** Mastered the complex pipeline mechanics needed to safely impute, encode, scale, and clean cross-sectional survey data featuring high categorical density (320 features).
* **Generalization > Precision:** Learned that tracking the overfitting gap via rigorous validation strategies is far more critical for safe business deployments than chasing fractional gains in training accuracy.
* **Interpretability Trade-offs:** Gained insights into navigating the friction between complex deep learning models and simpler, transparent algorithms like Logistic Regression.



---

##  9. Limitations
* **Temporal Friction:** The source dataset reflects corporate climates from **2016**. Shifts in industry dynamics (such as the widespread normalization of remote work ecosystems post-2020) mean data drift could impact current application.
* **Selection Bias:** The dataset features self-reported responses from specific sub-communities, presenting potential selection and non-response bias.
* **No Linear Causation:** All statistical outputs represent correlations and localized classification probabilities; **they do not establish causal lines**.

---
