"""Train family, rise, and 16 design models. Select winners using validation only."""
from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None
from sklearn.preprocessing import LabelEncoder
from truss_common import *

warnings.filterwarnings("ignore", category=UserWarning)


def optional_candidates(target: str, y_train: np.ndarray, input_cols: list[str], quick: bool = False) -> dict:
    categorical = is_categorical_target(target, y_train if target == "web_spacing_ft" else None)
    n_classes = len(np.unique(y_train)) if categorical else None
    if categorical and n_classes < 2:
        # A genuinely single-class target is valid; use an explicit constant baseline.
        # If validation contains another class, fit_and_select rejects that split below.
        return {"constant_baseline": make_estimator_pipeline(
            input_cols, DummyClassifier(strategy="constant", constant=int(np.unique(y_train)[0]))
        )}
    if not categorical and len(np.unique(np.asarray(y_train, dtype=float))) < 2:
        return {"constant_baseline": make_estimator_pipeline(
            input_cols, DummyRegressor(strategy="constant", constant=float(np.asarray(y_train, dtype=float)[0]))
        )}
    rf_n = 80 if quick else 350
    xgb_n = 80 if quick else 400
    cat_n = 80 if quick else 500
    candidates = {
        "random_forest": make_estimator_pipeline(
            input_cols,
            RandomForestClassifier(n_estimators=rf_n, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
            if categorical else RandomForestRegressor(n_estimators=rf_n, random_state=SEED, n_jobs=-1),
        )
    }
    try:
        from xgboost import XGBClassifier, XGBRegressor
        if categorical:
            est = XGBClassifier(
                n_estimators=xgb_n, max_depth=6, learning_rate=0.05, subsample=0.85,
                colsample_bytree=0.85, reg_lambda=2, objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                eval_metric="mlogloss" if n_classes > 2 else "logloss", tree_method="hist", random_state=SEED, n_jobs=-1,
            )
        else:
            est = XGBRegressor(
                n_estimators=xgb_n, max_depth=6, learning_rate=0.05, subsample=0.85,
                colsample_bytree=0.85, reg_lambda=2, objective="reg:squarederror", eval_metric="rmse",
                tree_method="hist", random_state=SEED, n_jobs=-1,
            )
        candidates["xgboost"] = make_estimator_pipeline(input_cols, est)
    except ImportError:
        print("[INFO] xgboost not installed; skipping XGBoost.")
    try:
        import catboost  # noqa: F401
        if categorical:
            est = CatBoostCompatClassifier(
                iterations=cat_n, depth=6, learning_rate=0.05,
                loss_function="MultiClass" if n_classes > 2 else "Logloss",
                random_seed=SEED, auto_class_weights="Balanced", thread_count=-1,
            )
        else:
            est = CatBoostCompatRegressor(
                iterations=cat_n, depth=6, learning_rate=0.05, loss_function="RMSE",
                random_seed=SEED, thread_count=-1,
            )
        candidates["catboost"] = make_estimator_pipeline(input_cols, est)
    except ImportError:
        print("[INFO] catboost not installed; skipping CatBoost.")
    return candidates


def fit_and_select(target: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   train_x: pd.DataFrame, val_x: pd.DataFrame, input_cols: list[str], quick: bool = False):
    categorical = is_categorical_target(target, train_df[target])
    y_train, encoder = encode_target(train_df[target], categorical)
    y_val_raw = train_df[target].astype(str) if False else val_df[target]
    if categorical:
        # A validation class absent from training is a data split issue, not a label to silently encode.
        unknown = sorted(set(y_val_raw.astype(str)) - set(encoder.classes_))
        if unknown:
            raise ValueError(f"Validation contains {target} classes absent from training: {unknown}. Adjust group split or add data.")
        y_val = encoder.transform(y_val_raw.astype(str))
    else:
        y_val = pd.to_numeric(y_val_raw, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y_val).all():
            raise ValueError(f"Validation target {target} has missing/non-finite values.")

    candidates = optional_candidates(target, y_train, input_cols, quick=quick)
    scores = []
    best_name, best_model, best_score = None, None, None
    for name, model in candidates.items():
        model.fit(clean_features(train_x, input_cols), y_train)
        raw = model.predict(clean_features(val_x, input_cols))
        if categorical:
            pred = encoder.inverse_transform(np.clip(np.rint(np.asarray(raw, dtype=float)).astype(int), 0, len(encoder.classes_) - 1))
            actual = val_df[target].astype(str).to_numpy()
            metric = categorical_metrics(actual, pred)
            score = metric["macro_f1"]
            better = best_score is None or score > best_score
            selection_metric = "macro_f1"
        else:
            pred = np.asarray(raw, dtype=float)
            if target in COUNT_TARGETS:
                pred = np.maximum(0, np.rint(pred))
            metric = regression_metrics(y_val, pred)
            score = metric["mae"]
            better = best_score is None or score < best_score
            selection_metric = "mae"
        scores.append({"model": name, "selection_metric": selection_metric, "selection_score": float(score), "validation_metrics": metric})
        if better:
            best_name, best_model, best_score = name, model, score
    bundle = {
        "schema_version": "truss_hybrid_v4",
        "target": target,
        "model_name": best_name,
        "model": best_model,
        "input_columns": input_cols,
        "is_categorical": categorical,
        "label_encoder": encoder,
        "selection_metric": selection_metric,
        "selection_score": float(best_score),
        "candidate_results": scores,
        "seed": SEED,
    }
    return bundle, scores


def oof_predictions(bundle: dict, x: pd.DataFrame, y: np.ndarray, target: str, quick: bool = False, groups=None) -> np.ndarray:
    """OOF predictions keep related site geometries together where feasible."""
    model = bundle["model"]
    n_splits = 3 if quick else 5
    categorical = bool(bundle.get("is_categorical", target in CATEGORICAL_TARGETS))
    if groups is not None and pd.Series(groups).nunique() >= 2:
        n_splits = min(n_splits, int(pd.Series(groups).nunique()))
        if categorical and StratifiedGroupKFold is not None:
            try:
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
                split_iter = splitter.split(x, y, groups)
            except ValueError:
                splitter = GroupKFold(n_splits=n_splits)
                split_iter = splitter.split(x, y, groups)
        else:
            splitter = GroupKFold(n_splits=n_splits)
            split_iter = splitter.split(x, y, groups)
    elif categorical:
        min_class = int(pd.Series(y).value_counts().min())
        n_splits = min(n_splits, min_class)
        if n_splits < 2:
            raise ValueError(f"Cannot make stratified OOF predictions for {target}: a class has fewer than 2 rows.")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        split_iter = splitter.split(x, y)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        split_iter = splitter.split(x)
    oof = np.empty(len(x), dtype=float)
    for fit_idx, hold_idx in split_iter:
        fold_model = clone(model)
        fold_model.fit(clean_features(x.iloc[fit_idx], bundle["input_columns"]), y[fit_idx])
        oof[hold_idx] = np.asarray(fold_model.predict(clean_features(x.iloc[hold_idx], bundle["input_columns"]))).reshape(-1)
    return decode_predictions(bundle, oof)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use fewer trees/iterations and 3-fold OOF for a smoke run; outputs go to *_smoke folders.")
    args = parser.parse_args()
    global MODEL_DIR, RESULT_DIR
    if args.quick:
        MODEL_DIR = ROOT / "models" / "truss_hybrid_v4_smoke"
        RESULT_DIR = ROOT / "results" / "truss_hybrid_v4_smoke"
    train_df, val_df = load_split("train"), load_split("validation")
    # Test split is deliberately not loaded or consulted during training/model selection.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print("=" * 78)
    print("TRUSS HYBRID V4 TRAINING")
    print(f"Train rows={len(train_df):,} | Validation rows={len(val_df):,} | Test is untouched")
    print(f"Model output: {MODEL_DIR}")
    print("=" * 78)

    x_train_family = clean_features(train_df, BASE_FEATURES)
    x_val_family = clean_features(val_df, BASE_FEATURES)
    if "source_design_id" in train_df and train_df["source_design_id"].notna().all() and train_df["source_design_id"].nunique() < len(train_df):
        oof_groups = train_df["source_design_id"].astype(str).to_numpy()
    else:
        oof_groups = train_df[["site_width_ft", "site_length_ft"]].round(6).astype(str).agg("|".join, axis=1).to_numpy()
    family_bundle, family_candidates = fit_and_select(FAMILY_TARGET, train_df, val_df, x_train_family, x_val_family, BASE_FEATURES, args.quick)
    # OOF family predictions are used for training downstream models; validation uses model predictions.
    y_family_encoded = family_bundle["label_encoder"].transform(train_df[FAMILY_TARGET].astype(str))
    family_oof = oof_predictions(family_bundle, x_train_family, y_family_encoded, FAMILY_TARGET, args.quick, oof_groups)
    family_val_pred = predict_bundle(family_bundle, val_df)
    joblib.dump(family_bundle, MODEL_DIR / f"{FAMILY_TARGET}.joblib")

    # Rise model: family is an upstream prediction, not the ground-truth label, for model inputs.
    x_train_rise = train_df[BASE_FEATURES].copy()
    x_train_rise[FAMILY_TARGET] = family_oof
    x_val_rise = val_df[BASE_FEATURES].copy()
    x_val_rise[FAMILY_TARGET] = family_val_pred
    rise_bundle, rise_candidates = fit_and_select(RISE_TARGET, train_df, val_df, x_train_rise, x_val_rise, RISE_FEATURES, args.quick)
    rise_y = pd.to_numeric(train_df[RISE_TARGET], errors="coerce").to_numpy(dtype=float)
    rise_oof = oof_predictions(rise_bundle, x_train_rise, rise_y, RISE_TARGET, args.quick, oof_groups)
    rise_val_pred = predict_bundle(rise_bundle, x_val_rise)
    joblib.dump(rise_bundle, MODEL_DIR / f"{RISE_TARGET}.joblib")

    x_train_design = train_df[BASE_FEATURES].copy()
    x_train_design[FAMILY_TARGET] = family_oof
    x_train_design[RISE_TARGET] = rise_oof
    x_val_design = val_df[BASE_FEATURES].copy()
    x_val_design[FAMILY_TARGET] = family_val_pred
    x_val_design[RISE_TARGET] = rise_val_pred

    summaries = {
        "schema_version": "truss_hybrid_v4",
        "selection_policy": "Best candidate chosen only by validation metrics; test split is not loaded or used during training.",
        "train_rows": len(train_df), "validation_rows": len(val_df),
        "family_model": {"winner": family_bundle["model_name"], "candidates": family_candidates},
        "rise_model": {"winner": rise_bundle["model_name"], "candidates": rise_candidates},
        "design_models": {},
        "elapsed_seconds": None,
        "feature_schema": {"family": BASE_FEATURES, "rise": RISE_FEATURES, "design": DESIGN_FEATURES},
        "design_targets": DESIGN_TARGETS,
        "target_types": {t: ("classification" if is_categorical_target(t, train_df[t]) else "regression") for t in MODEL_TARGETS},
        "warnings": [],
    }
    if "overhang_ft" in train_df and train_df["overhang_ft"].nunique(dropna=True) <= 1:
        summaries["warnings"].append("overhang_ft is constant/missing and excluded from model inputs.")
    if train_df["roof_configuration_detail"].nunique(dropna=True) <= 1 or train_df["roof_configuration_detail"].astype(str).eq("Unknown").all():
        summaries["warnings"].append("roof_configuration_detail has no useful variation; family prediction cannot benefit from it until real values are supplied.")
    if float(family_bundle["selection_score"]) < 0.80:
        summaries["warnings"].append(
            f"AUTO truss-family classifier validation macro-F1 is only {family_bundle['selection_score']:.3f} (<0.800 acceptance target). Keep MANUAL family selection as the safe default until the family inputs/rules or dataset are improved and revalidated."
        )

    from tqdm import tqdm
    for target in tqdm(DESIGN_TARGETS, desc="Training design targets", unit="target", dynamic_ncols=True):
        bundle, candidate_scores = fit_and_select(target, train_df, val_df, x_train_design, x_val_design, DESIGN_FEATURES, args.quick)
        joblib.dump(bundle, MODEL_DIR / f"{target}.joblib")
        summaries["design_models"][target] = {"winner": bundle["model_name"], "selection_metric": bundle["selection_metric"], "selection_score": bundle["selection_score"], "candidates": candidate_scores}
    summaries["elapsed_seconds"] = round(time.time() - start, 2)
    write_json(RESULT_DIR / "training_summary.json", summaries)
    write_json(MODEL_DIR / "pipeline_metadata.json", {
        "schema_version": "truss_hybrid_v4", "base_features": BASE_FEATURES,
        "family_target": FAMILY_TARGET, "rise_features": RISE_FEATURES, "rise_target": RISE_TARGET,
        "design_features": DESIGN_FEATURES, "design_targets": DESIGN_TARGETS,
        "model_files": [f"{t}.joblib" for t in MODEL_TARGETS],
        "seed": SEED, "selection_source": "validation.csv", "test_used_during_training": False,
        "numeric_unit": "feet internally; UI should convert fractional feet/inches to decimal feet before inference.",
    })
    print(f"\nTraining complete in {summaries['elapsed_seconds'] / 60:.1f} minutes.")
    print(f"Models  : {MODEL_DIR}")
    print(f"Summary : {RESULT_DIR / 'training_summary.json'}")

if __name__ == "__main__":
    main()
