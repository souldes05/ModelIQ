import bentoml
import pandas as pd
import joblib
import json
import numpy as np
from bentoml.io import JSON

# -----------------------------
# Load artifacts
# -----------------------------
MODEL_PATH = "models/churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"
META_PATH = "models/churn_meta.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(META_PATH, "r") as f:
    meta = json.load(f)

feature_cols = meta["feature_columns"]

# -----------------------------
# Define BentoML service
# -----------------------------
svc = bentoml.Service("telco_churn_service")

@svc.api(input=JSON(), output=JSON())
def predict(request):
    df = pd.DataFrame(request["data"])
    df = df.reindex(columns=feature_cols)
    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)
    return {"predictions": preds.tolist()}

