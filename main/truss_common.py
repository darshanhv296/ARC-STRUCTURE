"""Shared schema and utilities for the fixed TRUSS ML pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Put these scripts in C:\\TRUSS\\files. The project root is then C:\\TRUSS.
ROOT = Path(os.environ.get("TRUSS_ROOT", Path(__file__).resolve().parent.parent))
DATASET_DIR = ROOT / "DATASET"
PREPROCESSED_DIR = ROOT / "preprocessed"
MODEL_DIR = ROOT / "models" / "truss_hybrid_v4"
RESULT_DIR = ROOT / "results" / "truss_hybrid_v4"
SEED = 42

# V4 hybrid schema: use available user-known design inputs. The roof configuration
# is supplied as "Unknown" if absent from the source CSV; it is never fabricated.
# Constant overhang_ft is deliberately excluded from model features.
BASE_FEATURES = [
    "site_width_ft",
    "site_length_ft",
    "roofing_type",
    "roof_sides",
    "center_support_present",
    "column_height_B_ft",
    "roof_configuration_detail",
]
FAMILY_TARGET = "truss_family"
RISE_TARGET = "rise_height_ft"
RISE_FEATURES = BASE_FEATURES + [FAMILY_TARGET]
DESIGN_FEATURES = BASE_FEATURES + [FAMILY_TARGET, RISE_TARGET]
DESIGN_TARGETS = [
    "frame_chord_pipe_size",
    "web_pipe_size",
    "web_spacing_ft",
    "web_intermediate_member_count",
    "web_total_member_count",
    "truss_spacing_ft",
    "truss_total_count",
    "column_pipe_size",
    "column_total_count",
    "column_spacing_along_length_ft",
    "column_spacing_across_span_ft",
    "beam_pipe_size",
    "beam_total_count",
    "purlin_pipe_size",
    "purlin_orientation",
    "purlin_spacing_ft",
]
CATEGORICAL_INPUTS = {
    "roofing_type", "roof_sides", "center_support_present",
    "roof_configuration_detail", FAMILY_TARGET,
}
CATEGORICAL_TARGETS = {
    FAMILY_TARGET,
    "frame_chord_pipe_size", "web_pipe_size", "column_pipe_size",
    "beam_pipe_size", "purlin_pipe_size", "purlin_orientation",
}
COUNT_TARGETS = {
    "web_intermediate_member_count", "web_total_member_count",
    "truss_total_count", "column_total_count", "beam_total_count",
}
NUMERIC_TARGETS = [c for c in DESIGN_TARGETS if c not in CATEGORICAL_TARGETS]
MODEL_TARGETS = [FAMILY_TARGET, RISE_TARGET] + DESIGN_TARGETS
METADATA_COLUMNS = ["sample_id", "design_id", "source_design_id"]
MODEL_COLUMNS = list(dict.fromkeys(BASE_FEATURES + [FAMILY_TARGET, RISE_TARGET] + DESIGN_TARGETS))
SPLIT_COLUMNS = list(dict.fromkeys(METADATA_COLUMNS + MODEL_COLUMNS + ["overhang_ft"]))


class CatBoostCompatClassifier(__import__("sklearn.base", fromlist=["ClassifierMixin"]).ClassifierMixin, __import__("sklearn.base", fromlist=["BaseEstimator"]).BaseEstimator):
    """Thin sklearn-compatible wrapper for CatBoost releases missing sklearn 1.8 tags."""
    def __init__(self, iterations=500, depth=6, learning_rate=0.05, loss_function="MultiClass", random_seed=42, auto_class_weights="Balanced", thread_count=-1):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.auto_class_weights = auto_class_weights
        self.thread_count = thread_count

    def fit(self, X, y):
        from catboost import CatBoostClassifier
        self.estimator_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            loss_function=self.loss_function, random_seed=self.random_seed, verbose=False,
            allow_writing_files=False, auto_class_weights=self.auto_class_weights,
            thread_count=self.thread_count,
        )
        self.estimator_.fit(X, y, verbose=False)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.asarray(self.estimator_.predict(X)).reshape(-1)

    def predict_proba(self, X):
        return np.asarray(self.estimator_.predict_proba(X), dtype=float)


class CatBoostCompatRegressor(__import__("sklearn.base", fromlist=["RegressorMixin"]).RegressorMixin, __import__("sklearn.base", fromlist=["BaseEstimator"]).BaseEstimator):
    """Thin sklearn-compatible wrapper for CatBoost releases missing sklearn 1.8 tags."""
    def __init__(self, iterations=500, depth=6, learning_rate=0.05, loss_function="RMSE", random_seed=42, thread_count=-1):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.thread_count = thread_count

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.estimator_ = CatBoostRegressor(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            loss_function=self.loss_function, random_seed=self.random_seed,
            verbose=False, allow_writing_files=False, thread_count=self.thread_count,
        )
        self.estimator_.fit(X, y, verbose=False)
        return self

    def predict(self, X):
        return np.asarray(self.estimator_.predict(X)).reshape(-1)


def make_preprocessor(input_columns: list[str]) -> ColumnTransformer:
    cat_cols = [c for c in input_columns if c in CATEGORICAL_INPUTS]
    num_cols = [c for c in input_columns if c not in cat_cols]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    transformers = []
    if num_cols:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols))
    if cat_cols:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]), cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def make_estimator_pipeline(input_columns: list[str], estimator: Any) -> Pipeline:
    return Pipeline([("preprocessor", make_preprocessor(input_columns)), ("model", estimator)])


def clean_features(frame: pd.DataFrame, input_columns: list[str]) -> pd.DataFrame:
    x = frame.loc[:, input_columns].copy()
    for col in input_columns:
        if col in CATEGORICAL_INPUTS:
            x[col] = x[col].astype("string").fillna("Unknown").str.strip()
            x[col] = x[col].replace({"": "Unknown", "nan": "Unknown", "<NA>": "Unknown"}).astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    if x.isna().all(axis=None):
        raise ValueError("All model input values are missing after conversion.")
    return x


def encode_target(y: pd.Series, categorical: bool):
    if not categorical:
        vals = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError("Numeric target contains missing or non-finite values.")
        return vals, None
    from sklearn.preprocessing import LabelEncoder
    labels = y.astype(str).str.strip()
    if labels.isna().any() or labels.isin(["", "nan", "None", "<NA>", "Unknown"]).any():
        raise ValueError("Categorical target contains missing/placeholder labels.")
    enc = LabelEncoder()
    return enc.fit_transform(labels), enc


def decode_predictions(bundle: dict, raw_predictions) -> np.ndarray:
    pred = np.asarray(raw_predictions).reshape(-1)
    if bundle.get("is_categorical", False):
        enc = bundle["label_encoder"]
        ids = np.rint(pred.astype(float)).astype(int)
        ids = np.clip(ids, 0, len(enc.classes_) - 1)
        return enc.inverse_transform(ids)
    pred = pred.astype(float)
    if bundle.get("target") in COUNT_TARGETS:
        pred = np.maximum(0, np.rint(pred)).astype(int)
    return pred


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    x = clean_features(frame, bundle["input_columns"])
    return decode_predictions(bundle, bundle["model"].predict(x))


def regression_metrics(actual, predicted) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if not len(a):
        return {"rows_scored": 0, "error": "No valid numeric rows"}
    return {
        "rows_scored": int(len(a)),
        "mae": float(mean_absolute_error(a, p)),
        "rmse": float(np.sqrt(mean_squared_error(a, p))),
        "r2": float(r2_score(a, p)) if len(a) > 1 else None,
        "max_absolute_error": float(np.max(np.abs(a - p))),
    }


def categorical_metrics(actual, predicted) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    a = pd.Series(actual).astype(str).to_numpy()
    p = pd.Series(predicted).astype(str).to_numpy()
    mask = np.isin(a, ["", "nan", "None", "<NA>", "Unknown"], invert=True)
    a, p = a[mask], p[mask]
    if not len(a):
        return {"rows_scored": 0, "error": "No valid categorical rows"}
    return {
        "rows_scored": int(len(a)),
        "accuracy": float(accuracy_score(a, p)),
        "macro_f1": float(f1_score(a, p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(a, p, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(a, p, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(a, p, average="macro", zero_division=0)),
    }


def is_categorical_target(target: str, values=None) -> bool:
    """Use classification for known categorical outputs and low-cardinality web spacing."""
    if target in CATEGORICAL_TARGETS:
        return True
    if target == "web_spacing_ft" and values is not None:
        vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().unique()
        # Treat spacing as a choice label only when the observed training choices are few.
        return 2 <= len(vals) <= 12
    return False


def metrics_for_target(actual, predicted, target: str, categorical_override=None) -> dict:
    categorical = target in CATEGORICAL_TARGETS if categorical_override is None else bool(categorical_override)
    if categorical:
        return categorical_metrics(actual, predicted)
    return regression_metrics(actual, predicted)


def apply_design_guardrails(predictions: dict) -> tuple[dict, np.ndarray]:
    """Apply only the confirmed single-rafter convention; return corrected predictions and mask."""
    pred = {k: np.asarray(v).copy() for k, v in predictions.items()}
    n = len(next(iter(pred.values()))) if pred else 0
    corrected = np.zeros(n, dtype=bool)
    web_pipe = pred.get("web_pipe_size")
    if web_pipe is not None:
        single = np.asarray([str(v).strip() == "None (Single Rafter)" for v in web_pipe], dtype=bool)
        corrected |= single
        for target in ("web_spacing_ft", "web_intermediate_member_count", "web_total_member_count"):
            if target in pred:
                vals = pred[target].astype(object)
                vals[single] = 0
                pred[target] = vals
    # Counts must be nonnegative integers regardless of estimator output.
    for target in COUNT_TARGETS:
        if target in pred:
            vals = pd.to_numeric(pd.Series(pred[target]), errors="coerce").fillna(0).to_numpy(float)
            pred[target] = np.maximum(0, np.rint(vals)).astype(int)
    return pred, corrected


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_split(name: str) -> pd.DataFrame:
    path = PREPROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run preprocess_truss.py first.")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in SPLIT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} has incompatible schema; missing: {missing}. Rerun preprocessing.")
    return df


def load_bundle(target: str) -> dict:
    path = MODEL_DIR / f"{target}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing model {path}. Run train_truss.py first.")
    bundle = joblib.load(path)
    if bundle.get("target") != target:
        raise ValueError(f"Model file {path.name} contains unexpected target {bundle.get('target')!r}.")
    return bundle


def sequential_predict(frame: pd.DataFrame, include_design: bool = True) -> tuple[pd.DataFrame, dict]:
    """Predict family -> rise -> design targets using only upstream predictions."""
    output = frame[METADATA_COLUMNS].copy() if all(c in frame for c in METADATA_COLUMNS) else pd.DataFrame(index=frame.index)
    family_bundle = load_bundle(FAMILY_TARGET)
    family_x = clean_features(frame, family_bundle["input_columns"])
    family_pred = decode_predictions(family_bundle, family_bundle["model"].predict(family_x))
    output[f"actual_{FAMILY_TARGET}"] = frame[FAMILY_TARGET].astype(str).to_numpy()
    output[f"predicted_{FAMILY_TARGET}"] = family_pred
    try:
        family_probabilities = np.asarray(family_bundle["model"].predict_proba(family_x), dtype=float)
        output["predicted_truss_family_confidence"] = family_probabilities.max(axis=1)
    except (AttributeError, ValueError):
        output["predicted_truss_family_confidence"] = np.nan

    rise_bundle = load_bundle(RISE_TARGET)
    rise_input = frame[BASE_FEATURES].copy()
    rise_input[FAMILY_TARGET] = family_pred
    rise_pred = predict_bundle(rise_bundle, rise_input)
    output[f"actual_{RISE_TARGET}"] = pd.to_numeric(frame[RISE_TARGET], errors="coerce").to_numpy()
    output[f"predicted_{RISE_TARGET}"] = rise_pred

    predictions = {FAMILY_TARGET: family_pred, RISE_TARGET: rise_pred}
    if include_design:
        design_input = frame[BASE_FEATURES].copy()
        design_input[FAMILY_TARGET] = family_pred
        design_input[RISE_TARGET] = rise_pred
        for target in DESIGN_TARGETS:
            bundle = load_bundle(target)
            pred = predict_bundle(bundle, design_input)
            predictions[target] = pred
        predictions, corrected_mask = apply_design_guardrails(predictions)
        output["guardrail_single_rafter_applied"] = corrected_mask
        for target in DESIGN_TARGETS:
            output[f"actual_{target}"] = frame[target].to_numpy()
            output[f"predicted_{target}"] = predictions[target]
    return output, predictions
