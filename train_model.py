"""
train_model.py
--------------
Trains multiple classifiers on the social media / mental-health dataset,
selects the best model by cross-validated accuracy, and saves the full
sklearn pipeline to models/model.pkl.

Run:  python train_model.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "dataset.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# ── Feature / target split ────────────────────────────────────────────────────
TARGET = "wellbeing_band"
DROP_COLS = ["participant_id"]   # identifier – not a predictor

df = df.drop(columns=DROP_COLS, errors="ignore")
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Encode target to integers (needed by some estimators; we store the mapping)
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"  Classes: {list(le.classes_)}")

# ── Identify column types ─────────────────────────────────────────────────────
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"  Numeric features  ({len(numeric_cols)}): {numeric_cols}")
print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer",  SimpleImputer(strategy="most_frequent")),
    ("encoder",  OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer,  numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# ── Models to compare ─────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

# ── Cross-validation ──────────────────────────────────────────────────────────
print("\nCross-validating models (5-fold stratified) …")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, clf in MODELS.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])
    scores = cross_val_score(pipe, X, y_enc, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<25}  mean={scores.mean():.4f}  std={scores.std():.4f}")

# ── Select best model ─────────────────────────────────────────────────────────
best_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\nBest model: {best_name}  (mean CV accuracy = {cv_results[best_name].mean():.4f})")

best_clf = MODELS[best_name]
best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   best_clf),
])

# ── Train / test split & final evaluation ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

print("\nTest-set performance:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Save pipeline + metadata ──────────────────────────────────────────────────
artifact = {
    "pipeline":          best_pipeline,
    "label_encoder":     le,
    "feature_names":     X.columns.tolist(),
    "numeric_cols":      numeric_cols,
    "categorical_cols":  categorical_cols,
    "best_model_name":   best_name,
    "cv_results":        {k: v.tolist() for k, v in cv_results.items()},
    # store unique category values so Streamlit can build dropdowns
    "cat_options":       {col: sorted(df[col].dropna().unique().tolist())
                          for col in categorical_cols},
    # store numeric range hints
    "num_stats":         df[numeric_cols].describe().to_dict(),
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(artifact, f)

print(f"\nModel saved to {MODEL_PATH}")
print("Training complete.")
