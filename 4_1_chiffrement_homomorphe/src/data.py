from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

from .schema import FEATURES, TARGET, LABEL_MAP


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    feature_names: list[str]
    target_name: str


def load_arff(path: str | Path) -> pd.DataFrame:
    data, meta = arff.loadarff(str(path))
    df = pd.DataFrame(data)

    # scipy.io.arff charge souvent les strings en bytes -> decode
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)

    return df


def infer_target_column(df: pd.DataFrame) -> str:
    # Heuristiques simples : souvent "class" ou dernière colonne
    lowered = {c.lower(): c for c in df.columns}
    for key in ("class", "target", "label", "y"):
        if key in lowered:
            return lowered[key]
    return df.columns[-1]


def to_xy(df: pd.DataFrame, target_col: str | None = None):
    target_col = target_col or TARGET

    # Impose l'ordre et l'ensemble de colonnes attendues
    missing = [c for c in FEATURES + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[FEATURES].copy()
    y_raw = df[target_col].astype(str).to_numpy()

    # Mapping explicite good/bad 
    unknown = sorted(set(y_raw) - set(LABEL_MAP.keys()))
    if unknown:
        raise ValueError(f"Unknown labels in target: {unknown}. Expected {list(LABEL_MAP.keys())}")

    y = np.array([LABEL_MAP[v] for v in y_raw], dtype=np.int64)

    return Dataset(X=X, y=y, feature_names=list(X.columns), target_name=target_col)


def split_dataset(
    ds: Dataset,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dataset, Dataset]:
    X_train, X_test, y_train, y_test = train_test_split(
        ds.X, ds.y, test_size=test_size, random_state=random_state, stratify=ds.y
    )
    train = Dataset(X=X_train, y=y_train, feature_names=ds.feature_names, target_name=ds.target_name)
    test = Dataset(X=X_test, y=y_test, feature_names=ds.feature_names, target_name=ds.target_name)
    return train, test
