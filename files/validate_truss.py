"""Evaluate selected saved models on validation.csv using sequential upstream predictions."""
from __future__ import annotations
import argparse
import pandas as pd
import truss_common as tc
from truss_common import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use models/results from the --quick training run.")
    args = parser.parse_args()
    global MODEL_DIR, RESULT_DIR
    if args.smoke:
        MODEL_DIR = ROOT / "models" / "truss_hybrid_v4_smoke"
        RESULT_DIR = ROOT / "results" / "truss_hybrid_v4_smoke"
        tc.MODEL_DIR = MODEL_DIR
        tc.RESULT_DIR = RESULT_DIR
    df = load_split("validation")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    predictions, _ = sequential_predict(df, include_design=True)
    metrics = {}
    target_types = {}
    for target in MODEL_TARGETS:
        bundle = load_bundle(target)
        target_types[target] = bool(bundle.get("is_categorical", target in CATEGORICAL_TARGETS))
        metrics[target] = metrics_for_target(df[target], predictions[f"predicted_{target}"], target, target_types[target])
    predictions.to_csv(RESULT_DIR / "validation_predictions.csv", index=False)
    family_macro_f1 = metrics[FAMILY_TARGET].get("macro_f1")
    readiness_warning = None
    if family_macro_f1 is not None and family_macro_f1 < 0.80:
        readiness_warning = "AUTO family prediction is below the 0.800 macro-F1 acceptance target. Prefer MANUAL family selection until inputs/rules or dataset are improved."
    write_json(RESULT_DIR / "validation_metrics.json", {
        "split": "validation", "rows": len(df), "metrics": metrics, "target_types": target_types,
        "auto_family_readiness_warning": readiness_warning,
        "note": "Sequential predictions: predicted family feeds rise model; predicted family and rise feed design models. Family class IDs are decoded to original labels before scoring. predicted_truss_family_confidence is the model's maximum class probability, not a calibrated probability.",
    })
    print("Validation complete.")
    for target, result in metrics.items():
        if target_types.get(target, target in CATEGORICAL_TARGETS):
            print(f"{target:38s} accuracy={result.get('accuracy', float('nan')):.4f} macro-F1={result.get('macro_f1', float('nan')):.4f}")
        else:
            print(f"{target:38s} MAE={result.get('mae', float('nan')):.5f} RMSE={result.get('rmse', float('nan')):.5f}")
    print(f"Metrics: {RESULT_DIR / 'validation_metrics.json'}")

if __name__ == "__main__":
    main()
