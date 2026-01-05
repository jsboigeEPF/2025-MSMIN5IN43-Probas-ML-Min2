from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from concrete.ml.deployment import FHEModelDev
from concrete.ml.sklearn import LogisticRegression

from .data import load_arff, split_dataset, to_xy
from .schema import CAT_COLS, NUM_COLS


def build_preprocessor():
    """
    Préprocesseur hardcodé selon le schéma ARFF.
    - Catégoriel -> OneHotEncoder
    - Numérique -> StandardScaler
    """

    # Transforme les colonnes de type : checking_status { "<0", "0<=X<200", ">=200", "no checking" } en plusieurs colonnes 
    # checking_status_<0              = 1
    # checking_status_0<=X<200        = 0
    # checking_status_>=200           = 0
    # checking_status_no checking     = 0 

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre, CAT_COLS, NUM_COLS


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "dataset_31_credit-g.arff"

    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    fhe_dir = artifacts_dir / "fhe"
    # FHEModelDev.save exige un dossier vide (sinon exception)
    if fhe_dir.exists():
        for p in fhe_dir.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in sorted([x for x in fhe_dir.rglob("*") if x.is_dir()], reverse=True):
            p.rmdir()
        fhe_dir.rmdir()
    fhe_dir.mkdir()

    print(f"[i] Loading dataset: {data_path}")
    df = load_arff(data_path)
    ds = to_xy(df)
    train, test = split_dataset(ds)

    # Preprocessing
    pre, cat_cols, num_cols = build_preprocessor()
    pre.fit(train.X)

    X_train = pre.transform(train.X).astype(np.float32)
    X_test = pre.transform(test.X).astype(np.float32)
    y_train = train.y
    y_test = test.y
    #print("X_train", X_train[:1])
    #print("Y_train", y_train)
    print("train class ratio (mean y):", float(np.mean(y_train)))
    print("test class ratio (mean y):", float(np.mean(y_test)))

    # Modèle Concrete-ML (sklearn-like)
    # n_bits : profondeur/precision de quantification (trade-off perf/qualité)
    model = LogisticRegression(n_bits=8)

    print("[i] Training clear model...")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    print("[i] Evaluating clear model...")
    t1 = time.perf_counter()
    proba_clear = model.predict_proba(X_test)[:, 1]
    infer_clear_time = time.perf_counter() - t1

    auc = roc_auc_score(y_test, proba_clear)
    acc = accuracy_score(y_test, (proba_clear >= 0.5).astype(int))

    print(f"[OK] Clear metrics: roc_auc_score={auc:.3f} accuracy_score={acc:.3f}")
    print(f"[OK] Train time={train_time:.3f}s | Clear inference time={infer_clear_time:.3f}s")

    # Compilation FHE (calibration sur un échantillon représentatif)
    calib = X_train[: min(len(X_train), 200)]
    print("[i] Compiling to FHE (this can take time)...")
    t2 = time.perf_counter()
    model.compile(calib)
    compile_time = time.perf_counter() - t2
    print(f"[OK] Compile time={compile_time:.3f}s")

    print("[i] Saving FHE artifacts (client.zip / server.zip)...")
    dev = FHEModelDev(path_dir=str(fhe_dir), model=model)
    dev.save(via_mlir=True)

    # Sauvegarde du préprocesseur pour le client (et uniquement le client)
    pre_path = artifacts_dir / "preprocessor.joblib"
    joblib.dump(pre, pre_path)
    print(f"[OK] Saved preprocessor: {pre_path}")

    meta = {
        "target": ds.target_name,
        "feature_names": ds.feature_names,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "n_bits": 8,
        "metrics_clear": {"auc": float(auc), "acc": float(acc)},
        "timings": {"train_s": train_time, "compile_s": compile_time, "infer_clear_s": infer_clear_time},
    }
    (artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] Saved meta: {artifacts_dir/'meta.json'}")

    print("\nNext:")
    print("  1) Run server:  uvicorn src.api:app --reload --port 8000")
    print("  2) Run client:  uvicorn src.front:app --reload --port 8500")


if __name__ == "__main__":
    main()
