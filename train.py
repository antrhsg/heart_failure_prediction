"""
train.py
--------
Run this script ONCE to preprocess the data, train all four models,
and save everything to the /models directory.

Usage:
    python train.py

Outputs (saved to ./models/):
    scaler.pkl          - fitted StandardScaler
    lr_model.pkl        - Logistic Regression
    svm_model.pkl       - Support Vector Machine
    xgb_model.pkl       - XGBoost
    mlp_model.pt        - MLP weights (PyTorch)
    feature_names.pkl   - column names used during training
    mlp_input_dim.pkl   - input dimension for MLP reconstruction
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
np.random.seed(SEED)
os.makedirs("models", exist_ok=True)


# ------------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv("heart.csv")


# ------------------------------------------------------------------
# 2. Preprocessing
# ------------------------------------------------------------------
print("Preprocessing...")
df_clean = df.copy()

# Impute Cholesterol zeros with sex-group median
for sex in df_clean["Sex"].unique():
    median_val = df_clean[(df_clean["Sex"] == sex) & (df_clean["Cholesterol"] > 0)]["Cholesterol"].median()
    df_clean.loc[(df_clean["Sex"] == sex) & (df_clean["Cholesterol"] == 0), "Cholesterol"] = median_val

# Impute RestingBP zero (data entry error)
bp_median = df_clean[df_clean["RestingBP"] > 0]["RestingBP"].median()
df_clean.loc[df_clean["RestingBP"] == 0, "RestingBP"] = bp_median

# Encode binary categoricals
df_clean["Sex"] = (df_clean["Sex"] == "M").astype(int)
df_clean["ExerciseAngina"] = (df_clean["ExerciseAngina"] == "Y").astype(int)

# One-hot encode multi-class categoricals
df_clean = pd.get_dummies(df_clean, columns=["ChestPainType", "RestingECG", "ST_Slope"], drop_first=False)


# ------------------------------------------------------------------
# 3. Feature Engineering
# ------------------------------------------------------------------
print("Engineering features...")

# HR_Efficiency: how close is the patient to their age-predicted max HR
df_clean["HR_Efficiency"] = df_clean["MaxHR"] / (220 - df_clean["Age"])

# EKG_Score: additive combination of Oldpeak and ST_Slope risk weight
slope_weight = df_clean["ST_Slope_Flat"] * 1 + df_clean["ST_Slope_Down"] * 2
df_clean["EKG_Score"] = df_clean["Oldpeak"] + slope_weight

# Stress_Impact: additive combination of Oldpeak and ExerciseAngina
df_clean["Stress_Impact"] = df_clean["Oldpeak"] + 2 * df_clean["ExerciseAngina"]


# ------------------------------------------------------------------
# 4. Train / Validation / Test Split
# ------------------------------------------------------------------
print("Splitting data...")
X = df_clean.drop("HeartDisease", axis=1)
y = df_clean["HeartDisease"]

# Combined stratification key preserves sex ratio AND HD ratio in every split
strat_key = X["Sex"].astype(str) + "_" + y.astype(str)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=strat_key
)

strat_key_tv = X_train_val["Sex"].astype(str) + "_" + y_train_val.astype(str)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=SEED, stratify=strat_key_tv
)

# Save feature names for inference
feature_names = list(X.columns)
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


# ------------------------------------------------------------------
# 5. Scaling (fit on train only)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("  Scaler saved.")


# ------------------------------------------------------------------
# 6. Cross-Validation Setup (combined sex+target stratification)
# ------------------------------------------------------------------
strat_key_train  = X_train["Sex"].astype(str) + "_" + y_train.astype(str)
strat_encoded    = LabelEncoder().fit_transform(strat_key_train)
cv               = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_splits        = list(cv.split(X_train_sc, strat_encoded))
cv_splits_xgb    = list(cv.split(X_train, strat_encoded))


# ------------------------------------------------------------------
# 7. Logistic Regression
# ------------------------------------------------------------------
print("Training Logistic Regression...")
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=SEED),
    {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "liblinear"]},
    cv=cv_splits, scoring="roc_auc", n_jobs=-1
)
lr_grid.fit(X_train_sc, y_train)
lr_best = lr_grid.best_estimator_
print(f"  Best params: {lr_grid.best_params_} | CV AUC: {lr_grid.best_score_:.4f}")

with open("models/lr_model.pkl", "wb") as f:
    pickle.dump(lr_best, f)
print("  LR saved.")


# ------------------------------------------------------------------
# 8. Support Vector Machine
# ------------------------------------------------------------------
print("Training SVM...")
svm_grid = GridSearchCV(
    SVC(probability=True, random_state=SEED),
    {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
    cv=cv_splits, scoring="roc_auc", n_jobs=-1
)
svm_grid.fit(X_train_sc, y_train)
svm_best = svm_grid.best_estimator_
print(f"  Best params: {svm_grid.best_params_} | CV AUC: {svm_grid.best_score_:.4f}")

with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(svm_best, f)
print("  SVM saved.")


# ------------------------------------------------------------------
# 9. XGBoost
# ------------------------------------------------------------------
print("Training XGBoost...")
xgb_grid = GridSearchCV(
    xgb.XGBClassifier(eval_metric="logloss", random_state=SEED, verbosity=0),
    {"n_estimators": [100, 200], "max_depth": [3, 5],
     "learning_rate": [0.05, 0.1, 0.2], "subsample": [0.8, 1.0]},
    cv=cv_splits_xgb, scoring="roc_auc", n_jobs=-1
)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
print(f"  Best params: {xgb_grid.best_params_} | CV AUC: {xgb_grid.best_score_:.4f}")

with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_best, f)
print("  XGBoost saved.")


# ------------------------------------------------------------------
# 10. MLP (PyTorch)
# ------------------------------------------------------------------
print("Training MLP...")

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(1)


def make_tensors(X, y):
    return TensorDataset(torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)))


train_ds = make_tensors(X_train_sc, y_train)
val_ds   = make_tensors(X_val_sc,   y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64)

input_dim = X_train_sc.shape[1]
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp       = MLP(input_dim).to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCELoss()

best_val_loss    = float("inf")
patience         = 15
patience_counter = 0
best_weights     = None

for epoch in range(150):
    mlp.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        criterion(mlp(Xb), yb).backward()
        optimizer.step()

    mlp.eval()
    v_loss = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            v_loss += criterion(mlp(Xb), yb).item()
    v_loss /= len(val_loader)

    if v_loss < best_val_loss:
        best_val_loss    = v_loss
        best_weights     = {k: v.clone() for k, v in mlp.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

mlp.load_state_dict(best_weights)
torch.save(mlp.state_dict(), "models/mlp_model.pt")

with open("models/mlp_input_dim.pkl", "wb") as f:
    pickle.dump(input_dim, f)
print("  MLP saved.")


print("\nAll models trained and saved to ./models/")
print("You can now run:  streamlit run app.py")
