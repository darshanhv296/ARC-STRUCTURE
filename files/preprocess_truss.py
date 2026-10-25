"""Validate the master dataset and create deterministic train/validation/test splits."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from truss_common import (
    ROOT, DATASET_DIR, PREPROCESSED_DIR, BASE_FEATURES, FAMILY_TARGET,
    RISE_TARGET, DESIGN_TARGETS, MODEL_COLUMNS, SPLIT_COLUMNS, METADATA_COLUMNS,
    SEED, write_json,
)

PREFERRED_DATASET = "truss_ai_complete_dataset_10000_purlin_remainder_fixed.csv"
NUMERIC_COLUMNS = [
    "site_width_ft", "site_length_ft", "overhang_ft", "column_height_B_ft", "rise_height_ft",
    "web_spacing_ft", "web_intermediate_member_count", "web_total_member_count", "truss_spacing_ft",
    "truss_total_count", "column_total_count", "column_spacing_along_length_ft",
    "column_spacing_across_span_ft", "beam_total_count", "purlin_spacing_ft",
]
POSITIVE_COLUMNS = [
    "site_width_ft", "site_length_ft", "column_height_B_ft", "rise_height_ft",
    "truss_spacing_ft", "truss_total_count", "column_total_count",
    "column_spacing_along_length_ft", "column_spacing_across_span_ft", "beam_total_count", "purlin_spacing_ft",
]
COUNT_COLUMNS = ["web_intermediate_member_count", "web_total_member_count", "truss_total_count", "column_total_count", "beam_total_count"]
REQUIRED_SOURCE = list(dict.fromkeys(MODEL_COLUMNS + [c for c in METADATA_COLUMNS if c in MODEL_COLUMNS]))


def find_master() -> Path:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}\nCreate it and place the master CSV there.")
    preferred = DATASET_DIR / PREFERRED_DATASET
    if preferred.exists():
        return preferred
    candidates = sorted(DATASET_DIR.glob("truss_ai_complete_dataset_10000_purlin_remainder_fixed.csv"))
    if not candidates:
        candidates = sorted(DATASET_DIR.glob("truss_ai_complete_dataset_*.csv"))
    if not candidates:
        candidates = sorted(DATASET_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV master dataset found in {DATASET_DIR}")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise RuntimeError(f"Multiple possible master CSVs found ({names}). Keep the intended source named {PREFERRED_DATASET} to prevent training on the wrong version.")
    return candidates[0]


def normalize_numeric(s: pd.Series) -> pd.Series:
    text = s.astype("string").str.strip().str.replace(r"(?i)\s*(ft|feet)\s*$", "", regex=True).str.replace(",", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def group_keys(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "source_design_id" in df and df["source_design_id"].notna().any():
        src = df["source_design_id"].astype("string").str.strip()
        src = src.mask(src.isna() | src.isin(["", "nan", "<NA>"]))
        if src.notna().sum() == len(df) and src.nunique() < len(df):
            return src.astype(str), "source_design_id"
    # If source IDs are unique, keep exact site geometries together when they repeat.
    keys = df[["site_width_ft", "site_length_ft"]].round(6).astype(str).agg("|".join, axis=1)
    return keys, "site_width_ft+site_length_ft"


def split_groups(df: pd.DataFrame, groups: pd.Series):
    # Hold out ~15% test, then ~15% of total validation from the remaining 85%.
    idx = np.arange(len(df))
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    trainval_idx, test_idx = next(first.split(idx, groups=groups))
    trainval = df.iloc[trainval_idx].reset_index(drop=True)
    groups_tv = groups.iloc[trainval_idx].reset_index(drop=True)
    second = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=SEED + 1)
    train_idx_rel, val_idx_rel = next(second.split(trainval, groups=groups_tv))
    train = trainval.iloc[train_idx_rel].reset_index(drop=True)
    val = trainval.iloc[val_idx_rel].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    # Exact group-overlap checks are mandatory.
    g_train = set(groups.iloc[trainval_idx].iloc[train_idx_rel].astype(str))
    g_val = set(groups.iloc[trainval_idx].iloc[val_idx_rel].astype(str))
    g_test = set(groups.iloc[test_idx].astype(str))
    if g_train & g_val or g_train & g_test or g_val & g_test:
        raise RuntimeError("Leakage check failed: a group appears in more than one split.")
    return train, val, test, {"train": g_train, "validation": g_val, "test": g_test}


def main() -> None:
    print("=" * 78)
    print("TRUSS PREPROCESSING — FIXED SCHEMA")
    print("=" * 78)
    master_path = find_master()
    print(f"Project root : {ROOT}")
    print(f"Master file  : {master_path}")
    master = pd.read_csv(master_path, low_memory=False)
    master.columns = master.columns.astype(str).str.strip()
    # This feature is useful when present, but older V2/V3 CSVs may not contain it.
    # Add an explicit Unknown category rather than inventing a configuration.
    if "roof_configuration_detail" not in master.columns:
        master["roof_configuration_detail"] = "Unknown"
        print("[WARNING] roof_configuration_detail absent; using Unknown. Add real values to improve family prediction.")
    if "overhang_ft" not in master.columns:
        master["overhang_ft"] = np.nan
    missing = [c for c in MODEL_COLUMNS if c not in master.columns]
    if missing:
        raise ValueError("Master dataset is missing required model columns:\n - " + "\n - ".join(missing))
    original_rows = len(master)
    data = master[[c for c in SPLIT_COLUMNS if c in master.columns]].copy()

    # Strict normalization; no target labels are silently guessed or imputed.
    for col in NUMERIC_COLUMNS:
        if col in data:
            data[col] = normalize_numeric(data[col])
    categorical_cols = [c for c in MODEL_COLUMNS if c not in NUMERIC_COLUMNS]
    for col in categorical_cols:
        data[col] = data[col].astype("string").str.strip()
        data[col] = data[col].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "<NA>": pd.NA})
    if "center_support_present" in data:
        bool_map = {True: "True", False: "False", "True": "True", "False": "False", "true": "True", "false": "False", 1: "True", 0: "False", "1": "True", "0": "False", "Yes": "True", "No": "False"}
        data["center_support_present"] = data["center_support_present"].map(lambda v: bool_map.get(v, bool_map.get(str(v), str(v))))

    # Keep only rows with valid required inputs and targets; report every removal.
    required = MODEL_COLUMNS
    missing_mask = data[required].isna().any(axis=1)
    missing_rows = int(missing_mask.sum())
    data = data.loc[~missing_mask].copy()
    if data.empty:
        raise ValueError("No rows remain after missing-value checks.")

    bad_positive = pd.Series(False, index=data.index)
    for col in POSITIVE_COLUMNS:
        bad_positive |= data[col] <= 0
    bad_counts = pd.Series(False, index=data.index)
    for col in COUNT_COLUMNS:
        vals = data[col]
        bad_counts |= vals.lt(0) | ((vals % 1).abs() > 1e-9)
    bad_numeric = pd.Series(False, index=data.index)
    for col in NUMERIC_COLUMNS:
        bad_numeric |= ~np.isfinite(data[col].astype(float))
    invalid_rows_mask = bad_positive | bad_counts | bad_numeric
    invalid_rows = int(invalid_rows_mask.sum())
    data = data.loc[~invalid_rows_mask].copy()
    if data.empty:
        raise ValueError("No rows remain after numeric/domain validation.")

    # Apply the confirmed single-rafter convention and log any corrections.
    single_rafter_mask = data["web_pipe_size"].astype(str).str.strip().eq("None (Single Rafter)")
    corrected_single_rafter_rows = int((single_rafter_mask & (
        data["web_spacing_ft"].ne(0) | data["web_intermediate_member_count"].ne(0) | data["web_total_member_count"].ne(0)
    )).sum())
    data.loc[single_rafter_mask, ["web_spacing_ft", "web_intermediate_member_count", "web_total_member_count"]] = 0

    # Do not silently deduplicate different designs. Only remove fully duplicated records.
    duplicate_metadata = [c for c in ["sample_id", "design_id"] if c in data.columns]
    if duplicate_metadata:
        dup_mask = data.duplicated(subset=duplicate_metadata, keep="first")
        duplicate_id_rows = int(dup_mask.sum())
        data = data.loc[~dup_mask].copy()
    else:
        duplicate_id_rows = 0
    data = data.reset_index(drop=True)
    if len(data) < 100:
        raise ValueError(f"Only {len(data)} valid rows remain; refusing to create unreliable splits.")

    groups, group_strategy = group_keys(data)
    if groups.nunique() < 10:
        raise ValueError(f"Only {groups.nunique()} independent groups found; need at least 10 for a useful split.")
    train, val, test, group_sets = split_groups(data, groups.reset_index(drop=True))
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in [("train", train), ("validation", val), ("test", test)]:
        out = frame.reindex(columns=[c for c in SPLIT_COLUMNS if c in frame.columns])
        out.to_csv(PREPROCESSED_DIR / f"{name}.csv", index=False)
        print(f"Saved {name:10s}: {len(out):5d} rows")

    # Exact geometry overlap audit; site dimensions may overlap individually, but pairs/groups must not.
    def geom_set(frame):
        return set(zip(frame["site_width_ft"].round(6), frame["site_length_ft"].round(6)))
    geom = {"train": geom_set(train), "validation": geom_set(val), "test": geom_set(test)}
    report = {
        "schema_version": "truss_hybrid_v4",
        "master_file": str(master_path),
        "original_rows": original_rows,
        "retained_rows": int(len(data)),
        "dropped_missing_rows": missing_rows,
        "dropped_invalid_numeric_or_domain_rows": invalid_rows,
        "dropped_duplicate_id_rows": duplicate_id_rows,
        "single_rafter_rows": int(single_rafter_mask.sum()),
        "single_rafter_rows_corrected_to_zero": corrected_single_rafter_rows,
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "group_strategy": group_strategy,
        "independent_groups": int(groups.nunique()),
        "group_overlap_counts": {
            "train_validation": len(group_sets["train"] & group_sets["validation"]),
            "train_test": len(group_sets["train"] & group_sets["test"]),
            "validation_test": len(group_sets["validation"] & group_sets["test"]),
        },
        "site_geometry_overlap_counts": {
            "train_validation": len(geom["train"] & geom["validation"]),
            "train_test": len(geom["train"] & geom["test"]),
            "validation_test": len(geom["validation"] & geom["test"]),
        },
        "overhang_ft_unique_values": sorted(float(v) for v in data["overhang_ft"].dropna().unique()) if "overhang_ft" in data else [],
        "overhang_warning": "overhang_ft is retained for audit only and excluded from model inputs. The model cannot learn overhang effects from this feature.",
        "roof_configuration_detail_available": bool("roof_configuration_detail" in master.columns and master["roof_configuration_detail"].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"}).nunique() > 1),
        "base_features": BASE_FEATURES,
        "family_target": FAMILY_TARGET,
        "rise_features": BASE_FEATURES + [FAMILY_TARGET],
        "rise_target": RISE_TARGET,
        "design_features": BASE_FEATURES + [FAMILY_TARGET, RISE_TARGET],
        "design_targets": DESIGN_TARGETS,
        "seed": SEED,
        "notes": [
            "Only model inputs and targets plus IDs are included in split CSVs; downstream derived geometry/BOM fields remain in the master dataset and are calculated by the design engine.",
            "The source dataset convention is site_width_ft == horizontal_span_ft and site_length_ft == building_length_ft. Do not silently swap these when mapping UI fields.",
            "Test split is held out and must not be used for model selection.",
        ],
    }
    if any(report["group_overlap_counts"].values()) or any(report["site_geometry_overlap_counts"].values()):
        raise RuntimeError("Post-split leakage audit failed; outputs written but must not be used.")
    write_json(PREPROCESSED_DIR / "preprocessing_report.json", report)
    write_json(PREPROCESSED_DIR / "columns.json", {
        "model_columns": MODEL_COLUMNS,
        "split_columns": [c for c in SPLIT_COLUMNS if c in data.columns],
        "metadata_columns": METADATA_COLUMNS,
        "base_features": BASE_FEATURES,
        "family_target": FAMILY_TARGET,
        "rise_target": RISE_TARGET,
        "design_targets": DESIGN_TARGETS,
    })
    print(f"Audit report : {PREPROCESSED_DIR / 'preprocessing_report.json'}")
    print(f"Rows removed : missing={missing_rows}, invalid={invalid_rows}, duplicate IDs={duplicate_id_rows}")
    print(f"Group split  : {group_strategy}; groups={groups.nunique()}; no overlap verified")
    if "overhang_ft" in data and data["overhang_ft"].nunique(dropna=True) <= 1:
        print("[INFO] overhang_ft is constant/missing and excluded from model inputs.")
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FATAL] Preprocessing failed: {exc}", file=sys.stderr)
        raise
