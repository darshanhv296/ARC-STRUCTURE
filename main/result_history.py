"""Fast, validated design-result history and exact-request cache for TRUSS.

The CSV is a compact searchable index. Full validated result payloads are stored
as JSON sidecars so thousands of geometry members do not make CSV scans slow.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CACHE_VERSION = "truss-result-cache-v2"
_LOCK = threading.RLock()
_COLUMNS = [
    "cache_key", "created_at_utc", "cache_version", "model_dataset_signature",
    "mode", "requested_family", "roofing_type", "roof_sides",
    "site_width_ft", "site_length_ft", "column_height_B_ft", "overhang_ft",
    "candidate_count", "selected_family", "support_configuration",
    "max_unsupported_bay_ft", "member_count", "material_weight_kg", "payload_file",
]


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict(orient="records"))
        except TypeError:
            return _json_safe(value.to_dict())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _file_signature(path: Path):
    try:
        stat = path.stat()
        return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        return f"{path.name}:missing"


def model_dataset_signature(project_root: Path, model_dir: Path) -> str:
    """Lightly scan model artifacts and source CSV metadata, not their full bytes."""
    parts = []
    for name in ("master datatset.csv", "master dataset.csv", "pipes.csv"):
        parts.append(_file_signature(project_root / name))
    try:
        model_files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    except OSError:
        model_files = []
    for path in model_files:
        try:
            stat = path.stat()
            parts.append(f"{path.relative_to(model_dir).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            continue
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def make_cache_key(inputs: dict, mode: str, requested_family: str | None,
                   project_root: Path, model_dir: Path) -> tuple[str, str]:
    signature = model_dataset_signature(Path(project_root), Path(model_dir))
    normalized = {
        "version": CACHE_VERSION,
        "signature": signature,
        "mode": str(mode).upper(),
        "requested_family": requested_family or "",
        "inputs": {k: round(float(v), 4) if isinstance(v, (int, float)) and not isinstance(v, bool)
                    else v for k, v in sorted(inputs.items())},
    }
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), signature


def _paths(project_root: Path):
    results_dir = Path(project_root) / "results" / "generated_design_history"
    payload_dir = results_dir / "payloads"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / "generated_design_results.csv", payload_dir


def load_exact(cache_key: str, project_root: Path):
    """Load only an exact-key payload that was saved after successful validation."""
    _, payload_dir = _paths(project_root)
    payload = payload_dir / f"{cache_key}.json"
    if not payload.is_file():
        return None
    try:
        data = json.loads(payload.read_text(encoding="utf-8"))
        if data.get("cache_version") != CACHE_VERSION or not data.get("state", {}).get("candidates"):
            return None
        return data["state"]
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def save_success(cache_key: str, signature: str, state: dict, project_root: Path):
    """Persist a successful validated state plus a compact row for future search."""
    candidates = state.get("candidates") or []
    if not candidates:
        return
    csv_path, payload_dir = _paths(project_root)
    payload_path = payload_dir / f"{cache_key}.json"
    safe_state = _json_safe(state)
    payload = {"cache_version": CACHE_VERSION, "model_dataset_signature": signature,
               "state": safe_state}
    temp_payload = payload_path.with_suffix(".tmp")
    temp_payload.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temp_payload.replace(payload_path)

    chosen = candidates[0]
    design = chosen.get("design", {})
    bom = chosen.get("bom", {})
    inputs = state.get("inputs", {})
    row = {
        "cache_key": cache_key,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_version": CACHE_VERSION,
        "model_dataset_signature": signature,
        "mode": state.get("mode", ""),
        "requested_family": state.get("requested_family") or "",
        "roofing_type": inputs.get("roofing_type", ""),
        "roof_sides": inputs.get("roof_sides", ""),
        "site_width_ft": inputs.get("site_width_ft", ""),
        "site_length_ft": inputs.get("site_length_ft", ""),
        "column_height_B_ft": inputs.get("column_height_B_ft", ""),
        "overhang_ft": inputs.get("overhang_ft", ""),
        "candidate_count": len(candidates),
        "selected_family": chosen.get("truss_family", design.get("truss_family", "")),
        "support_configuration": chosen.get("support_configuration", ""),
        "max_unsupported_bay_ft": chosen.get("unsupported_span_ft", ""),
        "member_count": bom.get("member_count", ""),
        "material_weight_kg": bom.get("material_weight_kg", ""),
        "payload_file": str(payload_path.relative_to(Path(project_root))),
    }
    with _LOCK:
        exists = csv_path.exists() and csv_path.stat().st_size > 0
        # Prevent duplicate index rows for the same cache key.
        if exists:
            try:
                with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
                    if any(r.get("cache_key") == cache_key for r in csv.DictReader(f)):
                        return
            except (OSError, csv.Error):
                pass
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def similar_history(inputs: dict, mode: str, requested_family: str | None,
                    project_root: Path, limit: int = 3):
    """Return similar historical summaries as suggestions only, never as final designs."""
    csv_path, _ = _paths(project_root)
    if not csv_path.is_file():
        return []
    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))[-500:]
    except (OSError, csv.Error):
        return []
    candidates = []
    for row in rows:
        if row.get("cache_version") != CACHE_VERSION:
            continue
        if row.get("mode") != str(mode).upper():
            continue
        if row.get("roofing_type") != str(inputs.get("roofing_type", "")):
            continue
        if str(row.get("roof_sides")) != str(inputs.get("roof_sides", "")):
            continue
        if requested_family and row.get("requested_family") not in (requested_family, ""):
            continue
        try:
            width_delta = abs(float(row.get("site_width_ft", 0)) - float(inputs.get("site_width_ft", 0)))
            length_delta = abs(float(row.get("site_length_ft", 0)) - float(inputs.get("site_length_ft", 0)))
            height_delta = abs(float(row.get("column_height_B_ft", 0)) - float(inputs.get("column_height_B_ft", 0)))
            overhang_delta = abs(float(row.get("overhang_ft", 0)) - float(inputs.get("overhang_ft", 0)))
            distance = width_delta + length_delta + 2.0 * height_delta + 2.0 * overhang_delta
        except (TypeError, ValueError):
            continue
        item = dict(row)
        item["similarity_distance"] = round(distance, 3)
        item["width_difference_ft"] = round(width_delta, 2)
        item["length_difference_ft"] = round(length_delta, 2)
        candidates.append(item)
    candidates.sort(key=lambda r: (r["similarity_distance"], -len(str(r.get("created_at_utc", "")))))
    return candidates[:limit]
