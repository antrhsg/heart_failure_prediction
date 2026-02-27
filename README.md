# Heart Failure Prediction

A machine learning project that predicts the likelihood of heart disease from clinical features. Built as a personal project to demonstrate data science and ML skills for biomedical informatics and AI research applications.

The project includes a full analysis notebook (EDA, preprocessing, modeling) and a Streamlit web app where users can enter patient data and receive a risk prediction from the best-performing model (XGBoost).

---

## Live Demo

> To run locally, follow the setup instructions below.

---

## Project Overview

**Dataset:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data) — 918 patients, 11 clinical features, binary target (Heart Disease: Yes/No).

**Models trained:**
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost
- Multilayer Perceptron / MLP (PyTorch)

**Key techniques used:**
- Exploratory Data Analysis with 8+ visualizations
- Missing value imputation (Cholesterol zeros, RestingBP entry errors)
- Combined sex + target stratified K-Fold cross-validation to account for gender imbalance
- Hyperparameter tuning with GridSearchCV
- Feature engineering (HR Efficiency, EKG Score, Stress Impact)
- Full model evaluation: confusion matrix, ROC curves, F1, recall, precision, AUC

---

## Repository Structure

```
heart-failure-prediction/
│
├── Heart_Failure_Prediction.ipynb   # Full analysis notebook (EDA → modeling)
├── train.py                          # Trains all models, saves to ./models/
├── app.py                            # Streamlit web app
├── requirements.txt
├── README.md
│
├── heart.csv                         # Dataset (included)
└── models/                           # Pre-trained model files
    ├── scaler.pkl
    ├── feature_names.pkl
    ├── xgb_model.pkl
    └── (other model files from train.py)
```

---

## Setup and Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/heart-failure-prediction.git
cd heart-failure-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Then activate it:

- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```

You should see `(.venv)` appear at the start of your terminal prompt, confirming the environment is active.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the web app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Patient age in years |
| Sex | Categorical | M = Male, F = Female |
| ChestPainType | Categorical | ASY / ATA / NAP / TA |
| RestingBP | Numeric | Resting blood pressure (mmHg) |
| Cholesterol | Numeric | Serum cholesterol (mg/dL) |
| FastingBS | Binary | 1 if fasting blood sugar > 120 mg/dL |
| RestingECG | Categorical | Normal / ST / LVH |
| MaxHR | Numeric | Maximum heart rate achieved during exercise |
| ExerciseAngina | Binary | Y = Yes, N = No |
| Oldpeak | Numeric | ST depression induced by exercise |
| ST_Slope | Categorical | Up / Flat / Down |

**Engineered features (created during preprocessing):**

| Feature | Formula | Rationale |
|---|---|---|
| HR_Efficiency | MaxHR / (220 − Age) | Compares actual vs. age-predicted max HR (Tanaka formula) |
| EKG_Score | Oldpeak + slope_weight | Additive combination of two strongest EKG predictors |
| Stress_Impact | Oldpeak + 2 × ExerciseAngina | Combined exercise stress signal |

---

## Key Findings

- **ST_Slope** (Flat/Down), **ExerciseAngina**, and **Oldpeak** are the strongest predictors, all reflecting abnormal cardiac responses under physical stress.
- **Asymptomatic chest pain (ASY)** is paradoxically the highest-risk chest pain category — patients who feel no chest pain form the largest HD-positive group.
- **Cholesterol** and **RestingBP** appeared weak in the raw correlation analysis, but this was due to ~18% of cholesterol values being zero (missing data), not clinical irrelevance.
- **XGBoost** achieves the best overall performance; Logistic Regression is competitive and highly interpretable.
- A **combined sex + target stratification key** was used in all cross-validation splits to account for the dataset's 79%/21% male/female imbalance.

---

## Tech Stack

- Python 3.10
- pandas, NumPy, scikit-learn
- XGBoost
- PyTorch
- Streamlit
- Matplotlib, Seaborn

---

## Disclaimer

This project is for educational purposes only. It is not a medical device and should not be used for clinical diagnosis or treatment decisions. Always consult a licensed healthcare professional.

