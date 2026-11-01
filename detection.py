"""
detection.py
Detection + evaluation engine for GoldenEye Detector - Evaluation Edition.

Loads the trained Random Forest model + selected-features list, aligns an
uploaded CSV to the model's expected input, runs predictions, and (when a
ground-truth Label column is present) computes classification metrics so
the system can evaluate how well the model generalizes to traffic captured
in different years.
"""

import re
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_final.pkl"
FEATURES_PATH = BASE_DIR / "models" / "selected_features.pkl"

LABEL_MAP = {0: "BENIGN", 1: "DoS GoldenEye"}
PROTOCOL_MAP = {"6": "TCP", "17": "UDP", "1": "ICMP", "0": "HOPOPT"}
CLASSES = ["BENIGN", "DoS GoldenEye"]


@lru_cache(maxsize=1)
def load_model_and_features():
    """Load the RandomForest model and the selected feature list (cached once)."""
    model = joblib.load(MODEL_PATH)
    features = list(joblib.load(FEATURES_PATH))
    return model, features


def _clean(name: str) -> str:
    return str(name).strip().lower()


# ----------------------------------------------------------------------
# Feature alignment
# ----------------------------------------------------------------------
def align_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Build the exact feature matrix the model expects (case/whitespace-insensitive)."""
    col_lookup = {_clean(c): c for c in df.columns}
    data = {}
    for feat in features:
        key = _clean(feat)
        if key in col_lookup:
            data[feat] = pd.to_numeric(df[col_lookup[key]], errors="coerce")
        else:
            data[feat] = 0.0
    X = pd.DataFrame(data, columns=features)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def missing_features(df: pd.DataFrame, features: list) -> list:
    col_lookup = {_clean(c) for c in df.columns}
    return [f for f in features if _clean(f) not in col_lookup]


def get_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract or synthesize display metadata (Source/Destination IP & Port, Protocol)."""
    col_lookup = {_clean(c): c for c in df.columns}

    def find(*names):
        for n in names:
            key = _clean(n)
            if key in col_lookup:
                return df[col_lookup[key]]
        return None

    n = len(df)
    src_ip = find("Source IP", "Src IP", "SrcIP")
    dst_ip = find("Destination IP", "Dst IP", "DstIP")
    src_port = find("Source Port", "Src Port")
    dst_port = find("Destination Port", "Dst Port")
    protocol = find("Protocol")
    flow_duration = find("Flow Duration")

    rng = np.random.default_rng(42)

    if src_ip is None:
        octet4 = rng.integers(2, 250, size=n)
        octet2 = rng.integers(1, 4, size=n)
        src_ip = pd.Series([f"192.168.{o2}.{o4}" for o2, o4 in zip(octet2, octet4)])
    if dst_ip is None:
        dst_ip = pd.Series(["192.168.10.124"] * n)
    if src_port is None:
        src_port = pd.Series(rng.integers(1024, 65000, size=n))
    if dst_port is None:
        dst_port = pd.Series([80] * n)
    if protocol is None:
        protocol = pd.Series(["TCP"] * n)
    else:
        protocol = protocol.apply(lambda v: PROTOCOL_MAP.get(str(v).strip(), str(v).strip()))
    if flow_duration is None:
        flow_duration = pd.Series([0.0] * n)

    return pd.DataFrame({
        "Source IP": pd.Series(src_ip).astype(str).values,
        "Source Port": pd.to_numeric(pd.Series(src_port), errors="coerce").fillna(0).astype(int).values,
        "Destination IP": pd.Series(dst_ip).astype(str).values,
        "Destination Port": pd.to_numeric(pd.Series(dst_port), errors="coerce").fillna(0).astype(int).values,
        "Protocol": pd.Series(protocol).astype(str).values,
        "Flow Duration": pd.to_numeric(pd.Series(flow_duration), errors="coerce").fillna(0).values,
    })


# ----------------------------------------------------------------------
# Ground-truth label handling
# ----------------------------------------------------------------------
def find_label_column(df: pd.DataFrame):
    """Find a ground-truth label column, tolerating name variants/whitespace."""
    col_lookup = {_clean(c): c for c in df.columns}
    for candidate in ("label", "labels", "class", "target", " label"):
        if candidate in col_lookup:
            return col_lookup[candidate]
    return None


def normalize_label(value):
    """
    Map a raw ground-truth value to one of CLASSES, or None if unrecognized
    (e.g. a different attack type present in a raw multi-class CICIDS file).
    """
    s = str(value).strip()
    if s in ("0", "0.0"):
        return "BENIGN"
    if s in ("1", "1.0"):
        return "DoS GoldenEye"
    su = s.upper()
    if "BENIGN" in su:
        return "BENIGN"
    if "GOLDENEYE" in su or "GOLDEN EYE" in su or "GOLDEN_EYE" in su:
        return "DoS GoldenEye"
    return None


# ----------------------------------------------------------------------
# Year extraction from filename
# ----------------------------------------------------------------------
def extract_year(filename: str) -> str:
    match = re.search(r"(19|20)\d{2}", filename)
    return match.group(0) if match else "Unknown"


# ----------------------------------------------------------------------
# Core pipeline
# ----------------------------------------------------------------------
def run_detection(df: pd.DataFrame):
    """
    Run the model on a traffic dataframe. Returns:
      result_df   - metadata + Prediction + Probability (+ TrueLabel if present)
      feature_df  - the exact aligned feature matrix
      missing     - model features not found in the CSV
      label_col   - name of the ground-truth column found (or None)
      unmapped_n  - count of rows whose ground-truth value could not be mapped
    """
    model, features = load_model_and_features()

    X = align_features(df, features)
    meta = get_meta_columns(df)
    miss = missing_features(df, features)

    preds = model.predict(X)
    proba = model.predict_proba(X)
    max_proba = proba.max(axis=1)

    result = meta.copy()
    result["Prediction"] = [LABEL_MAP.get(p, str(p)) for p in preds]
    result["Probability"] = max_proba

    label_col = find_label_column(df)
    unmapped_n = 0
    if label_col is not None:
        raw_labels = df[label_col].reset_index(drop=True)
        true_labels = raw_labels.apply(normalize_label)
        unmapped_n = int(true_labels.isna().sum())
        result["TrueLabel"] = true_labels.values
        result["RawLabel"] = raw_labels.astype(str).values
    else:
        result["TrueLabel"] = None
        result["RawLabel"] = None

    result = result.reset_index(drop=True)
    return result, X.reset_index(drop=True), miss, label_col, unmapped_n


def compute_metrics(result_df: pd.DataFrame):
    """
    Compute accuracy/precision/recall/F1/confusion-matrix from a result_df
    that has both 'TrueLabel' and 'Prediction' columns. Rows where
    TrueLabel is None/unmapped are excluded from the metrics.
    """
    evaluable = result_df.dropna(subset=["TrueLabel"])
    if evaluable.empty:
        return None

    y_true = evaluable["TrueLabel"].values
    y_pred = evaluable["Prediction"].values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="DoS GoldenEye", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="DoS GoldenEye", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="DoS GoldenEye", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    tn, fp, fn, tp = cm.ravel()

    return {
        "evaluated_rows": int(len(evaluable)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
