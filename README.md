# 🧠 Chronic Kidney Disease (CKD) Prediction

This repository contains a simple Machine Learning project built as part of an academic learning exercise. The goal is to predict whether a patient has Chronic Kidney Disease (CKD) based on medical features.

> 📚 **Academic Project**  
> Submitted as part of academic coursework for Machine Learning.

---

## 📌 Project Description

This project involves data preprocessing, visualization, and training multiple machine learning models on a CKD dataset. The models are evaluated based on precision, recall, F1 score, and confusion matrix.

---

## 📂 Dataset

- **Source:** [Kaggle – Chronic Kidney Disease Dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease)
- **Total records:** 400
- **Target Column:** `class` — whether the patient has CKD (`ckd`) or not (`notckd`)

---

## ⚙️ Tech Stack

| Language / Library       | Purpose                                      |
|--------------------------|----------------------------------------------|
| `Python`                 | Programming language                         |
| `Pandas`                 | Data manipulation and preprocessing          |
| `NumPy`                  | Numerical operations                         |
| `Matplotlib`, `Seaborn`  | Data visualization                           |
| `scikit-learn`           | ML model building and evaluation             |

---

## 📊 Features

Here are the key features from the dataset:

### ➕ Medical Attributes

- `age`, `blood_pressure`, `specific_gravity`, `albumin`, `sugar`
- `blood_glucose_random`, `blood_urea`, `serum_creatinine`
- `sodium`, `potassium`, `haemoglobin`, `packed_cell_volume`, `white blood_cell_count`, `red blood_cell_count`

### 🩺 Categorical Attributes

- `red_blood_cells`, `pus_cell`, `pus_cell_clumps`, `bacteria`
- `hypertension`, `diabetes_mellitus`, `coronary_artery_disease`
- `appetite`, `peda_edema`, `anomia`

> ✅ After preprocessing and encoding, these were used to train the models.

---

## 🤖 ML Algorithms Used

| Algorithm                 | Accuracy | Precision | Recall | F1 Score | Confusion Matrix     |
|--------------------------|----------|-----------|--------|----------|----------------------|
| **Gaussian Naive Bayes** | 95.00%   | 1.000     | 0.920  | 0.959    | `[[37, 0], [5, 58]]`  |
| **K-Nearest Neighbors**  | 76.00%   | 0.882     | 0.714  | 0.789    | `[[31, 6], [18, 45]]` |
| **Random Forest**        | 98.00%   | 0.969     | 1.000  | 0.984    | `[[35, 2], [0, 63]]`  |
| **Decision Tree**        | 97.00%   | 0.969     | 0.984  | 0.976    | `[[35, 2], [1, 62]]`  |
| **Support Vector Machine** | 95.00% | 0.953     | 0.968  | 0.961    | `[[34, 3], [2, 61]]`  |

---

## 📉 Sample Output Explained

Example: For `Random Forest Classifier`

