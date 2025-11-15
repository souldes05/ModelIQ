#!/usr/bin/env python3
"""
retrain_simple_model.py — Production-Ready Retraining Script
------------------------------------------------------------
- Handles column name mapping automatically
- Coerces numeric and binary values
- Imputes missing data
- Trains Logistic or Decision Tree
- Saves model, scaler, meta.json
- Backs up old artifacts safely
- Full logging for audit
"""

import os
import sys
import json
import shutil
import joblib
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ---------------- CONFIG ----------------
DATA_PATH = Path(os.getenv("DATA_PATH", "../data/cleaned_data.csv"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "../models"))
MODEL_FILE = MODEL_DIR / "model.pkl"
SCALER_FILE = MODEL_DIR / "scaler.pkl"
META_FILE = MODEL_DIR / "meta.json"
MODEL_TYPE = os.getenv("MODEL_TYPE", "logistic").lower()  # or 'decision_tree'
TEST_SIZE = 0.2
RANDOM_STATE = 42
BACKUP = True
LOG_FILE = "retrain.log"

# Expected features for the API / model
EXPECTED_FEATURES = [
    "gender_binary", "SeniorCitizen", "Partner_binary", "Dependents_binary",
    "tenure", "PhoneService_binary", "PaperlessBilling_binary",
    "MonthlyCharges", "TotalCharges"
]
TARGET_COL = "Churn_binary"

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("retrain")

# ---------------- Helpers ----------------
def backup_file(path: Path):
    if path.exists() and BACKUP:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_suffix(path.suffix + f".backup.{ts}")
        shutil.copy2(path, backup_path)
        logger.info(f"🔁 Backed up {path} -> {backup_path}")

def safe_map_binary(series):
    """Map common boolean/yes/no/male/female values to 0/1"""
    def mapper(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)) and x in (0, 1):
            return int(x)
        s = str(x).strip().lower()
        if s in ("yes", "y", "true", "1", "male", "m"):
            return 1
        if s in ("no", "n", "false", "0", "female", "f"):
            return 0
        return np.nan
    return series.map(mapper)

# ---------------- Main ----------------
def main():
    logger.info("🚀 Starting retraining process...")

    if not DATA_PATH.exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        sys.exit(2)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(DATA_PATH)
        logger.info(f"✅ Loaded dataset: {DATA_PATH} | shape={df.shape}")
    except Exception as e:
        logger.exception("Failed to read CSV file")
        sys.exit(3)

    # --- Map actual dataset columns to expected binary columns ---
    COLUMN_MAPPING = {
        "gender_Male": "gender_binary",
        "Partner_Yes": "Partner_binary",
        "Dependents_Yes": "Dependents_binary",
        "PhoneService_Yes": "PhoneService_binary",
        "PaperlessBilling_Yes": "PaperlessBilling_binary",
        "Churn_Yes": "Churn_binary"
    }

    for src, target in COLUMN_MAPPING.items():
        if src in df.columns:
            df[target] = safe_map_binary(df[src])
    logger.info("🧩 Mapped dataset columns to expected binary features where possible.")

    # Ensure numeric fields are proper
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Validate that all expected columns are present ---
    missing = [c for c in EXPECTED_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        logger.error(f"Missing columns even after mapping/fix: {missing}")
        sys.exit(4)

    # --- Subset and impute missing values ---
    X = df[EXPECTED_FEATURES].copy()
    y = df[TARGET_COL].copy()

    # Impute missing numeric values with median, binary with mode
    for col in X.columns:
        if X[col].isna().sum() > 0:
            if X[col].dtype.kind in "biufc":  # numeric
                med = X[col].median()
                X[col].fillna(med, inplace=True)
            else:
                mode_val = int(X[col].mode(dropna=True).iloc[0]) if not X[col].mode(dropna=True).empty else 0
                X[col].fillna(mode_val, inplace=True)
    if y.isna().sum() > 0:
        mask = ~y.isna()
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
        logger.warning(f"Dropped {len(df) - len(y)} rows with missing target.")

    # --- Split and scale ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train model ---
    if MODEL_TYPE == "decision_tree":
        model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    else:
        model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = float(accuracy_score(y_test, y_pred))
    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_scaled)[:, 1]
            roc_auc = float(roc_auc_score(y_test, probs))
    except Exception:
        logger.warning("Failed to compute ROC AUC.")

    logger.info(f"✅ Model trained: Accuracy={acc:.4f}, ROC_AUC={roc_auc}")

    # --- Backup and save artifacts ---
    backup_file(MODEL_FILE)
    backup_file(SCALER_FILE)
    backup_file(META_FILE)
    joblib.dump(model, MODEL_FILE, compress=3)
    joblib.dump(scaler, SCALER_FILE, compress=3)

    meta = {
        "feature_names": EXPECTED_FEATURES,
        "target": TARGET_COL,
        "model_type": model.__class__.__name__,
        "accuracy": acc,
        "roc_auc": roc_auc,
        "n_samples": int(len(X)),
        "trained_at_utc": datetime.utcnow().isoformat()
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    logger.info("✅ Retraining completed successfully.")
    logger.info(f"Model: {MODEL_FILE}, Scaler: {SCALER_FILE}, Meta: {META_FILE}")
    print("✅ Retraining complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception during retraining")
        sys.exit(10)
