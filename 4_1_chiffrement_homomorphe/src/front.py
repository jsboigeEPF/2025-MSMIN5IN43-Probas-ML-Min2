from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from concrete.ml.deployment import FHEModelClient

app = FastAPI(title="FHE Credit Risk - Front (Local Encrypt/Decrypt)")
templates = Jinja2Templates(directory="templates")

# --- Paths / artifacts ---
_root = Path(__file__).resolve().parents[1]
_artifacts = _root / "artifacts"
_fhe_dir = _artifacts / "fhe"
_key_dir = _artifacts / "client_keys"
_key_dir.mkdir(exist_ok=True)

# --- Load preprocessor & meta ---
pre = joblib.load(_artifacts / "preprocessor.joblib")
meta = json.loads((_artifacts / "meta.json").read_text(encoding="utf-8"))

# Extract original feature list (before preprocessing)
feature_names = meta["feature_names"]
cat_cols = meta["cat_cols"]
num_cols = meta["num_cols"]

# Extract categories for categorical fields from OneHotEncoder inside ColumnTransformer
# ColumnTransformer named transformer "cat"
ohe = pre.named_transformers_["cat"]
categories_by_col = {col: list(cats) for col, cats in zip(cat_cols, ohe.categories_)}

# --- Load FHE client + keys once at startup ---
client = FHEModelClient(path_dir=str(_fhe_dir), key_dir=str(_key_dir))
client.load()
client.generate_private_and_evaluation_keys(force=False)
eval_keys = client.get_serialized_evaluation_keys()

# Where is the blind server?
BLIND_SERVER = "http://127.0.0.1:8000"


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "categories_by_col": categories_by_col,
            "default_server": BLIND_SERVER,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    server_url: str = Form(BLIND_SERVER),

    # numeric fields (all optional in signature, we validate later)
    duration: float = Form(...),
    credit_amount: float = Form(...),
    installment_commitment: float = Form(...),
    residence_since: float = Form(...),
    age: float = Form(...),
    existing_credits: float = Form(...),
    num_dependents: float = Form(...),

    # categorical fields
    checking_status: str = Form(...),
    credit_history: str = Form(...),
    purpose: str = Form(...),
    savings_status: str = Form(...),
    employment: str = Form(...),
    personal_status: str = Form(...),
    other_parties: str = Form(...),
    property_magnitude: str = Form(...),
    other_payment_plans: str = Form(...),
    housing: str = Form(...),
    job: str = Form(...),
    own_telephone: str = Form(...),
    foreign_worker: str = Form(...),
):
    # 1) Build one-row "raw" input (same schema as training BEFORE preprocessing)
    row = {
        "checking_status": checking_status,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_status": savings_status,
        "employment": employment,
        "installment_commitment": installment_commitment,
        "personal_status": personal_status,
        "other_parties": other_parties,
        "residence_since": residence_since,
        "property_magnitude": property_magnitude,
        "age": age,
        "other_payment_plans": other_payment_plans,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "num_dependents": num_dependents,
        "own_telephone": own_telephone,
        "foreign_worker": foreign_worker,
    }

    # 2) Preprocess locally (client side)
    df = pd.DataFrame([row], columns=feature_names)
    x = pre.transform(df).astype(np.float32)

    # 3) Encrypt locally
    t0 = time.perf_counter()
    encrypted_data = client.quantize_encrypt_serialize(x)
    enc_time = time.perf_counter() - t0

    payload = {
        "encrypted_data_b64": base64.b64encode(encrypted_data).decode("ascii"),
        "evaluation_keys_b64": base64.b64encode(eval_keys).decode("ascii"),
    }

    print(f"encrypted data sent to the backend in the payload : {payload}")

    # 4) Call blind server
    t1 = time.perf_counter()
    r = requests.post(f"{server_url}/run_fhe", json=payload, timeout=600)
    fhe_time = time.perf_counter() - t1
    r.raise_for_status()

    encrypted_result = base64.b64decode(r.json()["encrypted_result_b64"])

    # 5) Decrypt locally
    t2 = time.perf_counter()
    proba = client.deserialize_decrypt_dequantize(encrypted_result)
    dec_time = time.perf_counter() - t2

    p_bad = float(proba[0, 1])  # class=1
    pred = "BAD (risqué)" if p_bad >= 0.5 else "GOOD (solvable)"

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "server_url": server_url,
            "pred": pred,
            "p_bad": p_bad,
            "timings": {"encrypt_s": enc_time, "server_fhe_s": fhe_time, "decrypt_s": dec_time},
            "row": row,
        },
    )
