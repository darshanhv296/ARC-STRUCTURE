"""Load and run the saved TRUSS hybrid model bundles in sequence."""
from __future__ import annotations

import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

def _install_sklearn_pickle_compatibility() -> None:
    """Provide the removed sklearn helper name needed by some older joblib files.

    scikit-learn 1.6.1 artifacts can reference
    sklearn.compose._column_transformer._RemainderColsList, which is absent in
    some later versions. This narrowly scoped list subclass lets pickle resolve
    that historical class name. It does not guarantee cross-version estimator
    compatibility; the training scikit-learn version remains the safest choice.
    """
    try:
        import sklearn.compose._column_transformer as column_transformer_module

        if not hasattr(column_transformer_module, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            _RemainderColsList.__module__ = column_transformer_module.__name__
            setattr(column_transformer_module, "_RemainderColsList", _RemainderColsList)
            warnings.warn(
                "Installed scikit-learn is missing the historical _RemainderColsList "
                "class referenced by one or more saved TRUSS model files. A pickle "
                "compatibility shim was installed. Verify predictions against the "
                "training environment before relying on structural outputs.",
                RuntimeWarning,
                stacklevel=2,
            )
    except ImportError:
        # The normal sklearn import / model-load error will explain the missing dependency.
        return


_install_sklearn_pickle_compatibility()

_PIPELINE_DIR = Path(__file__).resolve().parents[1] / "files"
if not _PIPELINE_DIR.is_dir():
    raise ImportError(f"Required TRUSS inference module directory not found: {_PIPELINE_DIR}")
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

try:
    from truss_common import apply_design_guardrails, decode_predictions
except ImportError as exc:
    raise ImportError(
        f"Could not load the required TRUSS pipeline module from {_PIPELINE_DIR}: {exc}"
    ) from exc


COUNT_TARGETS = {
    "web_intermediate_member_count",
    "web_total_member_count",
    "truss_total_count",
    "column_total_count",
    "beam_total_count",
}

CATEGORICAL_TARGETS = {
    "truss_family",
    "frame_chord_pipe_size",
    "web_pipe_size",
    "column_pipe_size",
    "beam_pipe_size",
    "purlin_pipe_size",
    "purlin_orientation",
}


@lru_cache(maxsize=1)
def _load_model_catalog(model_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    model_path = Path(model_dir).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    metadata_file = model_path / "pipeline_metadata.json"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Missing model metadata: {metadata_file}")

    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid model metadata at {metadata_file}: {exc}") from exc

    model_files = metadata.get("model_files")
    if not isinstance(model_files, list) or not model_files:
        raise ValueError(
            f"Model metadata {metadata_file} must contain a non-empty 'model_files' list."
        )

    missing = [name for name in model_files if not (model_path / str(name)).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required model bundles: {', '.join(map(str, missing))}")

    bundles: dict[str, Any] = {}
    for artifact_name in model_files:
        artifact = model_path / str(artifact_name)
        try:
            bundle = joblib.load(artifact)
        except Exception as exc:
            message = (
                f"Could not load model bundle '{artifact.name}': {type(exc).__name__}: {exc}. "
                "This commonly indicates a scikit-learn version mismatch between model "
                "training and inference. Check the training environment's scikit-learn "
                "version and install that exact version in this project's virtual environment; "
                "if the compatibility shim does not resolve it, retrain and resave the models "
                "in the current environment."
            )
            raise RuntimeError(message) from exc

        if not isinstance(bundle, dict):
            raise TypeError(f"Model bundle '{artifact.name}' must contain a dictionary.")
        target = bundle.get("target")
        if not target:
            raise ValueError(f"Model bundle '{artifact.name}' is missing its 'target' field.")
        if str(target) in bundles:
            raise ValueError(f"Duplicate model target '{target}' found in model directory.")
        bundles[str(target)] = bundle

    return metadata, bundles


def _coerce_input_value(column: str, value: Any) -> Any:
    if value is None:
        raise ValueError(f"Missing required input '{column}'.")

    if column == "roof_sides":
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned in {"1", "2"}:
                return cleaned
            if cleaned.lower() in {"one", "1-side", "1 side"}:
                return "1"
            if cleaned.lower() in {"two", "2-side", "2 side"}:
                return "2"
        try:
            side_number = int(float(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Input 'roof_sides' must be 1 or 2.") from exc
        if side_number not in {1, 2}:
            raise ValueError("Input 'roof_sides' must be 1 or 2.")
        return str(side_number)

    if column == "center_support_present":
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in {"yes", "y", "true", "1"}:
                return "Yes"
            if cleaned in {"no", "n", "false", "0"}:
                return "No"
        if bool(value):
            return "Yes"
        return "No"

    if column in {"site_width_ft", "site_length_ft", "column_height_B_ft", "rise_height_ft"}:
        try:
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f"Input '{column}' must be a finite number; received {value!r}.")
            return number
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Input '{column}' must be numeric; received {value!r}.") from exc

    if isinstance(value, (int, float, np.integer, np.floating)):
        value = str(value)
    return str(value).strip()


def _row_from_inputs(inputs: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for column in feature_columns:
        if column not in inputs:
            raise ValueError(f"Missing required feature '{column}' for model inference.")
        row[column] = _coerce_input_value(column, inputs[column])
    return pd.DataFrame([row], columns=feature_columns)


def _decode_prediction(bundle: dict[str, Any], raw_prediction: Any) -> Any:
    decoded = decode_predictions(bundle, raw_prediction)
    value = decoded[0]
    if bundle.get("target") in COUNT_TARGETS:
        return int(value)
    if bundle.get("is_categorical"):
        return value
    return float(value)


def _predict_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> Any:
    model = bundle.get("model")
    if model is None:
        raise ValueError(f"Model bundle for '{bundle.get('target')}' is missing a fitted estimator.")
    prediction = model.predict(frame)
    return _decode_prediction(bundle, prediction)


def get_model_status(model_dir: Path):
    model_path = Path(model_dir)
    exists = model_path.exists()
    if not exists:
        return {
            "directory_exists": False,
            "artifact_count": 0,
            "artifact_names": [],
            "message": f"Model directory not found: {model_path}",
        }

    try:
        metadata_file = model_path / "pipeline_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Missing model metadata: {metadata_file}")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        model_files = metadata.get("model_files", [])
        missing = [name for name in model_files if not (model_path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing required model bundles: {', '.join(missing)}")
    except Exception as exc:  # pragma: no cover - UI reporting path
        return {
            "directory_exists": True,
            "artifact_count": len(list(model_path.glob("*.joblib"))),
            "artifact_names": [p.name for p in model_path.iterdir()],
            "message": str(exc),
        }

    return {
        "directory_exists": True,
        "artifact_count": len(model_files),
        "artifact_names": sorted(Path(name).stem for name in model_files),
        "metadata": metadata,
        "message": "Model metadata and saved artifacts are available; models load when a design is submitted.",
    }


def predict_design(inputs: dict, model_dir: Path, truss_family_override: str | None = None) -> dict:
    model_path = Path(model_dir)
    try:
        for key in ("site_width_ft", "site_length_ft", "column_height_B_ft"):
            value = _coerce_input_value(key, inputs.get(key))
            if value <= 0:
                return {
                    "connected": False,
                    "message": f"Input '{key}' must be greater than zero.",
                    "model_directory": str(model_path),
                }
        side_value = _coerce_input_value("roof_sides", inputs.get("roof_sides"))
    except (TypeError, ValueError) as exc:
        return {
            "connected": False,
            "message": f"Invalid model input: {exc}",
            "model_directory": str(model_path),
        }
    detail_value = str(inputs.get("roof_configuration_detail", "")).strip().casefold()
    if (side_value == "1" and detail_value in {"gable", "two sides / gable", "two-sided / gable"}) or (
        side_value == "2" and detail_value in {"single slope", "single-slope", "mono-pitch"}
    ):
        return {
            "connected": False,
            "message": "Roof sides conflict with the selected roof configuration.",
            "model_directory": str(model_path),
        }
    try:
        metadata, bundles = _load_model_catalog(model_path)
    except Exception as exc:
        return {
            "connected": False,
            "message": str(exc),
            "model_directory": str(model_path),
            "artifacts": [p.name for p in model_path.iterdir()] if model_path.exists() else [],
        }

    required_base = list(metadata.get("base_features", []))
    missing = [name for name in required_base if name not in inputs]
    if missing:
        return {
            "connected": False,
            "message": f"Missing required model inputs: {', '.join(missing)}",
            "model_directory": str(model_path),
            "artifacts": sorted(p.name for p in model_path.iterdir() if p.suffix == ".joblib"),
        }

    try:
        for metadata_key in ("family_target", "rise_target"):
            if metadata_key not in metadata:
                raise KeyError(f"Model metadata is missing required key '{metadata_key}'.")
        family_target = str(metadata["family_target"])
        rise_target = str(metadata["rise_target"])
        if family_target not in bundles:
            raise KeyError(f"Required family model target '{family_target}' was not loaded.")
        if rise_target not in bundles:
            raise KeyError(f"Required rise model target '{rise_target}' was not loaded.")
        family_bundle = bundles[family_target]
        rise_bundle = bundles[rise_target]
        base_frame = _row_from_inputs(inputs, family_bundle["input_columns"])
        model_family_pred = _predict_bundle(family_bundle, base_frame)
        family_pred = str(truss_family_override).strip() if truss_family_override else model_family_pred

        rise_inputs = dict(inputs)
        rise_inputs["truss_family"] = family_pred
        rise_frame = _row_from_inputs(rise_inputs, rise_bundle["input_columns"])
        rise_pred = _predict_bundle(rise_bundle, rise_frame)

        predictions: dict[str, Any] = {
            "truss_family": family_pred,
            "rise_height_ft": float(rise_pred),
        }
        if truss_family_override:
            predictions["model_predicted_truss_family"] = model_family_pred

        design_inputs = dict(inputs)
        design_inputs["truss_family"] = family_pred
        design_inputs["rise_height_ft"] = float(rise_pred)

        for target in metadata.get("design_targets", []):
            if target not in bundles:
                raise KeyError(f"Required design model target '{target}' was not loaded.")
            bundle = bundles[target]
            frame = _row_from_inputs(design_inputs, bundle["input_columns"])
            predictions[target] = _predict_bundle(bundle, frame)

        guarded, _ = apply_design_guardrails(
            {key: np.asarray([value]) for key, value in predictions.items()}
        )
        predictions.update({key: values[0] for key, values in guarded.items() if key != "guardrail_family_web_conflict"})
        predictions["guardrail_single_rafter_applied"] = bool(
            str(predictions.get("truss_family", "")).strip().casefold() == "plain pipe"
        )
        predictions["guardrail_family_web_conflict"] = bool(
            guarded.get("guardrail_family_web_conflict", np.asarray([False]))[0]
        )

        for key, value in list(predictions.items()):
            if key in COUNT_TARGETS and isinstance(value, (int, float, np.integer, np.floating)):
                predictions[key] = int(round(float(value)))
            elif isinstance(value, (float, np.floating)) and np.isfinite(value):
                predictions[key] = float(value)

        result = {
            "connected": True,
            "model_directory": str(model_path),
            "artifact_count": len(bundles),
            "design": predictions,
            "metadata": metadata,
            "message": "Sequential TRUSS model inference completed using the saved joblib bundles.",
        }
        return result
    except Exception as exc:  # pragma: no cover - validation path
        return {
            "connected": False,
            "message": f"Inference failed: {exc}",
            "model_directory": str(model_path),
            "artifacts": sorted(p.name for p in model_path.iterdir() if p.suffix == ".joblib"),
        }
