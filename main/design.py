"""Generate one deterministic member model for truss, building, and BOM views."""
from __future__ import annotations

import csv
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import plotly.graph_objects as go

MAX_OVERHANG_FT = 2.0
MAX_COLUMN_HEIGHT_FT = 18.0
RIDGE_CLEARANCE_FT = 0.5
REFERENCE_REQUIRED_COLUMNS = {
    "design_id",
    "roofing_type",
    "roof_sides",
    "truss_type",
    "span_min_ft",
    "span_max_ft",
    "main_pipe_alias",
    "web_pipe_alias",
    "support_pipe_alias",
    "column_pipe_alias",
    "beam_pipe_alias",
    "middle_support_required",
    "middle_support_pipe_alias",
    "support_count",
    "support_spacing_ft",
    "web_spacing_ft",
    "web_spacing_rule",
    "web_layout",
    "overhang_ft",
    "purlin_pipe_alias",
    "purlin_spacing_in",
    "notes",
}

ROOFING_RULES = {
    "Metal Sheet": {
        "spacing_source": "design value",
        "max_purlin_spacing_ft": 5.0,
        "max_purlin_spacing_at_30ft_span_ft": 3.0,
        "spacing_is_center_to_center": True,
        "purlin_orientations": [
            'Larger 1.5" face upward (flat) for broad sheet seating and screw grip',
            'Larger 2.5" face upward (flat) for broad sheet seating and screw grip',
            'Larger 3.0" face upward (flat) for broad sheet seating and screw grip',
            'Rotated 2 x 4 orientation (4.0" face upward flat, 2.0" vertical depth) for broad sheet seating',
            'Square 1.5" face upward (flat) for broad sheet seating and screw grip',
        ],
    },
    "Clay / Concrete Tile": {
        "max_purlin_spacing_ft": 0.75,
        "spacing_is_center_to_center": False,
        "purlin_orientations": ['Larger 1.5" face upward (flat) for tile nib resting and fastening'],
    },
}


def _number(data, key, default=None):
    try:
        value = float(data[key])
        return value if math.isfinite(value) else default
    except (KeyError, TypeError, ValueError):
        return default


def _count(data, key, default):
    value = _number(data, key, default)
    return max(0, int(round(value))) if value is not None else default


def _roof_name(inputs):
    detail = str(inputs.get("roof_configuration_detail", "")).strip().casefold()
    sides = str(inputs.get("roof_sides", "")).strip()
    if detail in {"gable", "two sides / gable", "two-sided / gable"}:
        return "gable"
    if detail in {"single slope", "single-slope", "mono-pitch"}:
        return "single_slope"
    if sides == "2":
        return "gable"
    if sides == "1":
        return "single_slope"
    raise ValueError("Choose a supported roof configuration: gable or single slope.")


def family_roof_compatibility_error(family, roof):
    """Return the deterministic geometry rule that rejects a family, if any."""
    family_kind = _family_kind(family)
    if family_kind in {"full_a", "half_a"} and roof != "gable":
        return f"{family} is supported only for a two-sided / gable roof in the current reference set."
    if family_kind == "semi" and roof != "single_slope":
        return f"{family} is supported only for a single-slope roof in the current reference set."
    if family_kind not in {"plain", "box", "full_a", "half_a", "semi"}:
        return f"No deterministic geometry rules are defined for truss family '{family}'."
    return None


def compatible_truss_families(inputs):
    """List named families accepted by the current roof's geometry rules."""
    roof = _roof_name(inputs)
    families = ["Full-A Truss", "Half-A Truss", "Box Truss", "Plain Pipe", "Semi-Truss"]
    if roof == "single_slope":
        families = ["Semi-Truss", "Plain Pipe", "Box Truss", "Full-A Truss", "Half-A Truss"]
    return [family for family in families if family_roof_compatibility_error(family, roof) is None]


def reconcile_truss_family(current_family, family_options):
    """Retain a compatible family; require an explicit choice after incompatibility."""
    return current_family if current_family in family_options else None


def _family_kind(family):
    label = str(family or "").strip().casefold()
    if not label:
        raise ValueError("A truss family is required to generate member geometry.")
    if "plain pipe" in label or "single rafter" in label:
        return "plain"
    if "box" in label:
        return "box"
    if "full-a" in label or "full a" in label:
        return "full_a"
    if "half-a" in label or "half a" in label:
        return "half_a"
    if "semi" in label:
        return "semi"
    if "a-frame" in label or "a frame" in label or label == "a-truss":
        return "a_frame"
    raise ValueError(f"No deterministic geometry rules are defined for truss family '{family}'.")


def _profile(x, span, rise, roof):
    if roof == "single_slope":
        return rise * x / span
    if x <= span / 2.0:
        return rise * 2.0 * x / span
    return rise * (2.0 - 2.0 * x / span)


@lru_cache(maxsize=1)
def _reference_rows():
    reference_path = Path(__file__).resolve().parents[1] / "master datatset.csv"
    try:
        with reference_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            missing_columns = REFERENCE_REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(
                    "TRUSS reference dataset is missing required columns: "
                    + ", ".join(sorted(missing_columns))
                )
            rows = tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ValueError(f"TRUSS design reference dataset is unavailable: {reference_path}") from exc
    if not rows:
        raise ValueError(f"TRUSS design reference dataset is empty: {reference_path}")

    design_ids = [str(row.get("design_id", "")).strip() for row in rows]
    if any(not design_id for design_id in design_ids):
        raise ValueError("TRUSS reference dataset contains a row without a design_id.")
    duplicate_ids = sorted(
        design_id for design_id, count in Counter(design_ids).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            "TRUSS reference dataset contains duplicate design_id values: "
            + ", ".join(duplicate_ids)
        )
    for row in rows:
        design_id = str(row["design_id"]).strip()
        span_min = _number(row, "span_min_ft")
        span_max = _number(row, "span_max_ft")
        if span_min is None or span_max is None or span_min < 0 or span_min > span_max:
            raise ValueError(
                f"TRUSS reference row {design_id} has an invalid supported span range."
            )
    return rows


def _reference_roofing_match(reference_label, selected_label):
    source = str(reference_label).strip().casefold()
    selected = str(selected_label).strip().casefold()
    if selected in {"metal sheet", "sheet"}:
        return source in {"sheet", "metal sheet"}
    if selected in {"clay / concrete tile", "tile", "tiles", "roofing tiles"}:
        return source in {"tile", "tiles", "roofing tiles", "clay / concrete tile"}
    return source == selected


def _reference_family_match(reference_label, family_kind):
    try:
        return _family_kind(reference_label) == family_kind
    except ValueError:
        return False


def _reference_candidates(inputs, family_kind, roof):
    """Find reference rows for the requested roof, including Box Truss span parity.

    For Box Truss only, a two-side/gable roof inherits the same supported span
    envelope as its one-side counterpart when the exact gable span band is absent.
    The source row's member aliases and span limits are retained; the requested
    input roof configuration still drives the generated gable geometry and ridge
    support layout. Exact roof-side references always take precedence when they
    cover the requested bay span.
    """
    side = "2-side" if roof == "gable" else "1-side"
    candidates = []
    rows = _reference_rows()
    for row in rows:
        row_side = str(row.get("roof_sides", "")).strip().casefold()
        if row_side != side:
            # Do not silently transfer span rules between roof configurations for
            # other families. Box Truss is the explicitly configured parity case.
            if not (family_kind == "box" and roof == "gable" and row_side == "1-side"):
                continue
            is_span_parity_fallback = True
        else:
            is_span_parity_fallback = False

        if not _reference_roofing_match(row.get("roofing_type", ""), inputs.get("roofing_type", "")):
            continue
        if not _reference_family_match(row.get("truss_type", ""), family_kind):
            continue
        try:
            span_min = float(row["span_min_ft"])
            span_max = float(row["span_max_ft"])
        except (KeyError, TypeError, ValueError):
            continue

        candidate_row = dict(row)
        if is_span_parity_fallback:
            candidate_row["_span_parity_fallback"] = True
            candidate_row["_source_roof_sides"] = row_side
            # Downstream geometry uses the user's requested roof configuration,
            # while the row supplies the shared span/member reference values.
            candidate_row["roof_sides"] = "2-side"
            candidate_row["notes"] = (
                str(candidate_row.get("notes", "")).strip()
                + " Gable span-parity fallback: shared one-side span/member reference; "
                  "generate two-side roof geometry and ridge/peak support arrangement."
            ).strip()
        else:
            candidate_row["_span_parity_fallback"] = False
        candidates.append((span_max - span_min, span_max, span_min, candidate_row))

    return sorted(
        candidates,
        key=lambda item: (
            bool(item[3].get("_span_parity_fallback", False)),
            item[0],
            -item[2],
            str(item[3].get("design_id", "")).strip().casefold(),
        ),
    )


@lru_cache(maxsize=1)
def _pipe_catalog_index():
    catalog_path = Path(__file__).resolve().parents[1] / "pipes.csv"
    try:
        with catalog_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required = {"pipe_id", "pipe_size", "thickness_mm", "standard_length_ft", "weight_kg"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    "TRUSS pipe catalog is missing required columns: "
                    + ", ".join(sorted(missing))
                )
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise ValueError(f"TRUSS pipe catalog is unavailable: {catalog_path}") from exc
    if not rows:
        raise ValueError(f"TRUSS pipe catalog is empty: {catalog_path}")

    by_id = {}
    by_size = {}
    for row in rows:
        pipe_id = str(row.get("pipe_id", "")).strip().casefold()
        pipe_size = str(row.get("pipe_size", "")).strip()
        if not pipe_id or not pipe_size:
            raise ValueError("TRUSS pipe catalog contains a row without pipe_id or pipe_size.")
        if pipe_id in by_id:
            raise ValueError(f"TRUSS pipe catalog contains duplicate pipe_id '{pipe_id.upper()}'.")
        for field in ("thickness_mm", "standard_length_ft", "weight_kg"):
            number = _number(row, field)
            if number is None or number <= 0:
                raise ValueError(
                    f"TRUSS pipe catalog section '{pipe_id.upper()}' has an invalid {field}."
                )
        by_id[pipe_id] = row
        by_size.setdefault(_normalize_section_label(pipe_size), []).append(row)
    return by_id, by_size


def _pipe_alias_condition(alias, match, next_match):
    segment = alias[match.end():next_match.start() if next_match else len(alias)]
    conditions = re.findall(r"\(([^)]*)\)", segment)
    return next(
        (
            condition.strip().casefold()
            for condition in conditions
            if re.search(r"(?:<=|>=|<|>)\s*\d", condition)
        ),
        None,
    )


def _pipe_alias_condition_matches(condition, overall_span, unsupported_span, alias, column_height_ft=None):
    """Evaluate pipe aliases against the actual unsupported truss bay.

    Span bands such as P001/P002/P005 are per truss bay, not the whole building
    width after intermediate supports are introduced. Column-height clauses use
    the actual column height and never the unsupported bay span.
    """
    if condition is None:
        return True
    remainder = condition
    column_match = re.search(r"column\s*>\s*(\d+(?:\.\d+)?)\s*ft", remainder)
    if column_match:
        if column_height_ft is None or column_height_ft <= float(column_match.group(1)):
            return False
        remainder = remainder[:column_match.start()] + remainder[column_match.end():]

    span_value = unsupported_span if unsupported_span is not None else overall_span
    interval = re.search(
        r">\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*ft", remainder
    )
    if interval:
        lower, upper = (float(interval.group(i)) for i in (1, 2))
        return span_value > lower and span_value <= upper
    upper = re.search(r"<=\s*(\d+(?:\.\d+)?)\s*ft", remainder)
    if upper:
        return span_value <= float(upper.group(1))
    lower_inclusive = re.search(r">=\s*(\d+(?:\.\d+)?)\s*ft", remainder)
    if lower_inclusive:
        return span_value >= float(lower_inclusive.group(1))
    lower = re.search(r">\s*(\d+(?:\.\d+)?)\s*ft", remainder)
    if lower:
        return span_value > float(lower.group(1))
    strict_upper = re.search(r"<\s*(\d+(?:\.\d+)?)\s*ft", remainder)
    if strict_upper:
        return span_value < float(strict_upper.group(1))
    raise ValueError(
        f"Reference pipe alias '{alias}' has an unsupported selection condition '{condition}'."
    )


def _prediction_matches_pipe(prediction, allowed_rows, by_id):
    normalized = _normalize_section_label(prediction)
    if not normalized:
        return None
    id_match = re.fullmatch(r"p\d{3}", normalized)
    if id_match:
        row = by_id.get(normalized)
        if row in allowed_rows:
            return row
        return None
    return next(
        (
            row for row in allowed_rows
            if _normalize_section_label(row["pipe_size"]) == normalized
            or _normalize_section_label(row["pipe_id"]) == normalized
        ),
        None,
    )


def _resolve_pipe_reference_alias(
    alias, prediction="", overall_span=0.0, unsupported_span=0.0, column_height_ft=None
):
    alias = str(alias or "").strip()
    if not alias:
        raise ValueError("A pipe section is not configured in the matched reference rule.")
    by_id, by_size = _pipe_catalog_index()
    matches = list(re.finditer(r"P\d{3}", alias, flags=re.IGNORECASE))
    allowed_rows = []
    selection_candidates = []
    selection_reason = ""

    if matches:
        reference_ids = [match.group(0).casefold() for match in matches]
        missing_ids = [pipe_id.upper() for pipe_id in reference_ids if pipe_id not in by_id]
        if missing_ids:
            raise ValueError(
                f"Reference pipe alias '{alias}' contains catalog ID(s) absent from "
                f"pipes.csv: {', '.join(missing_ids)}."
            )
        conditioned = []
        unconditioned = []
        for index, match in enumerate(matches):
            condition = _pipe_alias_condition(
                alias, match, matches[index + 1] if index + 1 < len(matches) else None
            )
            row = by_id[match.group(0).casefold()]
            if condition is None:
                unconditioned.append(row)
            elif _pipe_alias_condition_matches(
                condition, overall_span, unsupported_span, alias, column_height_ft
            ):
                conditioned.append(row)

        if conditioned:
            if len(conditioned) != 1:
                ids = ", ".join(str(row["pipe_id"]) for row in conditioned)
                raise ValueError(
                    f"Reference pipe alias '{alias}' matches multiple catalog options "
                    f"for this span/support configuration: {ids}."
                )
            allowed_rows = conditioned
            selection_candidates = conditioned
            selection_reason = "The matched reference span/support condition selects this catalog section."
        elif not any(
            _pipe_alias_condition(alias, match, matches[index + 1] if index + 1 < len(matches) else None)
            is not None
            for index, match in enumerate(matches)
        ):
            selection_candidates = unconditioned
            allowed_rows = list(unconditioned)
            if len(selection_candidates) > 1:
                prediction_match = _prediction_matches_pipe(prediction, selection_candidates, by_id)
                if prediction_match is not None:
                    allowed_rows = [prediction_match]
                    selection_reason = "The model prediction matches an option explicitly allowed by the reference."
                else:
                    allowed_rows = [allowed_rows[0]]
                    selection_reason = (
                        "No model prediction matched the reference-authorized alternatives; "
                        "the first listed catalog option was selected deterministically."
                    )
            else:
                selection_reason = "The reference alias names one catalog section."
        elif unconditioned and not conditioned:
            if len(unconditioned) == 1:
                allowed_rows = unconditioned
                selection_candidates = unconditioned
                selection_reason = "The unconditioned reference catalog option applies."
            else:
                raise ValueError(
                    f"Reference pipe alias '{alias}' has multiple unconditioned catalog options."
                )
        else:
            raise ValueError(
                f"No catalog option in reference pipe alias '{alias}' matches the "
                f"{overall_span:.2f} ft span and {unsupported_span:.2f} ft unsupported span."
            )
    else:
        normalized_alias = _normalize_section_label(alias)
        allowed_rows = by_size.get(normalized_alias, [])
        if not allowed_rows:
            raise ValueError(
                f"Reference pipe alias '{alias}' does not match any section in pipes.csv."
            )
        if len(allowed_rows) != 1:
            ids = ", ".join(str(row["pipe_id"]) for row in allowed_rows)
            raise ValueError(
                f"Reference pipe alias '{alias}' resolves ambiguously in pipes.csv: {ids}."
            )
        selection_candidates = allowed_rows
        selection_reason = "The matched reference section resolves to one catalog size."

    selected_row = allowed_rows[0]
    prediction = str(prediction or "").strip()
    prediction_match = _prediction_matches_pipe(prediction, selection_candidates, by_id)
    adjustment_reason = None
    if prediction and prediction_match is None:
        adjustment_reason = (
            f"Model prediction '{prediction}' is not authorized by reference alias "
            f"'{alias}'; catalog section '{selected_row['pipe_size']}' was selected."
        )
    return {
        "reference_alias": alias,
        "selected_pipe_id": str(selected_row["pipe_id"]),
        "selected_section": str(selected_row["pipe_size"]),
        "predicted_section": prediction or None,
        "allowed_sections": [
            str(row["pipe_size"]) for row in selection_candidates
        ],
        "selection_reason": selection_reason,
        "adjustment_reason": adjustment_reason,
    }


def _pipe_size_from_reference_alias(
    alias, fallback="", overall_span=0.0, unsupported_span=0.0, column_height_ft=None
):
    return _resolve_pipe_reference_alias(
        alias, fallback, overall_span, unsupported_span, column_height_ft
    )["selected_section"]


def _normalize_section_label(value):
    label = re.sub(
        r"\s*\(\s*P\d{3}\s*\)\s*$",
        "",
        str(value or "").strip(),
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", "", label.casefold()).replace("×", "x")


def _section_dimensions_in(section):
    match = re.search(
        r"(\d+(?:\.\d+)?|\d+\s*/\s*\d+)\s*[x×]\s*(\d+(?:\.\d+)?|\d+\s*/\s*\d+)",
        str(section),
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            f"Pipe section '{section}' has no usable dimensions for support geometry."
        )
    def inches(value):
        value = value.replace(" ", "")
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return float(numerator) / float(denominator)
        return float(value)
    return inches(match.group(1)), inches(match.group(2))


def _section_depth_ft(section, orientation=None):
    """Resolve vertical depth from an explicit orientation or the catalog dimension order."""
    dimensions = _section_dimensions_in(section)
    orientation = str(orientation or "").casefold()
    vertical_depth = re.search(
        r"(\d+(?:\.\d+)?)\s*[\"”]?\s*vertical\s+depth", orientation
    )
    if vertical_depth:
        depth_in = float(vertical_depth.group(1))
    else:
        face_up = re.search(
            r"(\d+(?:\.\d+)?)\s*[\"”]?\s*face\s+upward", orientation
        )
        if face_up:
            face_width = float(face_up.group(1))
            remaining = [dimension for dimension in dimensions if not math.isclose(
                dimension, face_width, rel_tol=0.0, abs_tol=1e-7
            )]
            depth_in = remaining[0] if remaining else dimensions[0]
        else:
            # Catalog labels are ordered depth x width for the standard upright
            # frame orientation; rotated members must provide their orientation.
            depth_in = dimensions[0]
    if depth_in <= 0:
        raise ValueError(f"Pipe section '{section}' has a non-positive vertical depth.")
    return depth_in / 12.0


def _box_depth_from_reference(reference_rule, inputs, family_kind):
    """Resolve box depth only from an explicit, nearby reference configuration.

    Some gable Box Truss rows say they use the same design as a single-slope row
    but omit the depth text themselves. Search both roof configurations for an
    explicit box-height note and choose the closest span band; never invent a
    proportional box depth.
    """
    note_pattern = re.compile(
        r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*-?\s*ft\s+box\s+height",
        flags=re.IGNORECASE,
    )
    direct_note = str(reference_rule.get("reference_notes", ""))
    direct_match = note_pattern.search(direct_note)
    if direct_match:
        return float(direct_match.group(1)), direct_note

    current_roof = _roof_name(inputs)
    other_roof = "single_slope" if current_roof == "gable" else "gable"
    target_span = _number(
        inputs, "unsupported_span_ft",
        _number(inputs, "site_width_ft", 0.0),
    )
    matched = []
    for roof_rank, roof in enumerate((current_roof, other_roof)):
        for _, span_max, span_min, row in _reference_candidates(inputs, family_kind, roof):
            note = str(row.get("notes", ""))
            match = note_pattern.search(note)
            if not match:
                continue
            if span_min <= target_span <= span_max:
                distance = 0.0
            else:
                distance = min(abs(target_span - span_min), abs(target_span - span_max))
            matched.append((roof_rank, distance, span_max - span_min, -span_max, row, match, note))
    if matched:
        _, _, _, _, row, match, note = min(matched, key=lambda item: item[:4])
        return float(match.group(1)), f"{row.get('design_id', 'reference')} · {note}"
    raise ValueError(
        "No nearby Box Truss reference specifies a box depth for this roofing type; "
        "the design cannot be validated without inventing a depth."
    )


def _reference_web_spacing_ft(reference_rule, inputs, family_kind, values):
    source_texts = [
        str(reference_rule.get("reference_notes", "")),
        str(reference_rule.get("web_spacing_ft", "")),
        str(reference_rule.get("web_spacing_rule", "")),
    ]
    for text in source_texts:
        box_spacing = re.search(
            r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*-?\s*ft\s+(?:web\s+)?spacing",
            text,
            flags=re.IGNORECASE,
        )
        if box_spacing:
            return float(box_spacing.group(2) or box_spacing.group(1)), text
        from_middle = re.search(
            r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*ft\s+from\s+middle",
            text,
            flags=re.IGNORECASE,
        )
        if from_middle:
            return float(from_middle.group(2)), text
        a_truss = re.search(
            r"A-truss\s+(\d+(?:\.\d+)?)\s*ft", text, flags=re.IGNORECASE
        )
        if a_truss and family_kind in {"full_a", "half_a", "semi"}:
            return float(a_truss.group(1)), text
    predicted = _number(values, "web_spacing_ft")
    if predicted is not None and predicted > 0:
        return predicted, "ML prediction; no numeric web spacing is configured in the reference"
    raise ValueError(
        "No numeric web spacing is configured for the matched reference and no valid "
        "web-spacing prediction is available."
    )


def _select_reference_candidate(inputs, family_kind, roof, requested_span):
    applicable = [
        item for item in _reference_candidates(inputs, family_kind, roof)
        if item[2] <= requested_span <= item[1]
    ]
    if not applicable:
        return None, []
    applicable.sort(
        key=lambda item: (
            bool(item[3].get("_span_parity_fallback", False)),
            item[0],
            -item[2],
            str(item[3].get("design_id", "")).strip().casefold(),
        )
    )
    return applicable[0], applicable


def reference_span_limit(inputs, family):
    """Return the largest recorded span for this roof/material/family combination."""
    roof = _roof_name(inputs)
    family_kind = _family_kind(family)
    rows = _reference_candidates(inputs, family_kind, roof)
    return max((item[1] for item in rows), default=None)


def reference_span_rule_for_bay(inputs, family, bay_span_ft):
    """Return the actual selected reference row's span and support eligibility for a bay."""
    bay_span = _number({"bay_span_ft": bay_span_ft}, "bay_span_ft")
    if bay_span is None or bay_span <= 0:
        return None
    roof = _roof_name(inputs)
    family_kind = _family_kind(family)
    selected, _ = _select_reference_candidate(inputs, family_kind, roof, bay_span)
    if selected is None:
        return None
    _, span_max, span_min, row = selected
    return {
        "design_id": str(row.get("design_id", "")).strip(),
        "span_min_ft": span_min,
        "span_max_ft": span_max,
        "middle_support_available": _row_supports_middle_line(row),
        "support_count_rule": str(row.get("support_count", "")).strip(),
        "pipe_aliases": {
            "main_pipe_alias": str(row.get("main_pipe_alias", "")).strip(),
            "web_pipe_alias": str(row.get("web_pipe_alias", "")).strip(),
        },
    }


def reference_middle_support_available(inputs, family):
    """Whether matching reference records describe a three-line support option."""
    roof = _roof_name(inputs)
    family_kind = _family_kind(family)
    for _, _, _, row in _reference_candidates(inputs, family_kind, roof):
        if _row_supports_middle_line(row):
            return True
    return False


def _row_supports_middle_line(row):
    middle_support = str(row.get("middle_support_required", "")).casefold()
    support_count = str(row.get("support_count", "")).casefold()
    return "yes" in middle_support and re.search(r"3\s+(?:with|if)\s+middle", support_count) is not None


def _reference_geometry_rule(inputs, values, family, family_kind, roof, span):
    overall_span = _number(inputs, "site_width_ft", span)
    support_span = _number(inputs, "unsupported_span_ft", span)
    rule_span = overall_span if family_kind == "half_a" else support_span
    roofing = inputs.get("roofing_type", "")
    center_support_requested = str(inputs.get("center_support_present", "No")).strip().casefold() in {
        "yes", "true", "1"
    }
    selected, candidates = _select_reference_candidate(
        inputs, family_kind, roof, rule_span
    )
    if selected is None:
        all_matching = _reference_candidates(inputs, family_kind, roof)
        supported_max = max(
            (item[1] for item in all_matching),
            default=None,
        )
        limit_detail = (
            f" The largest matching reference span is {supported_max:.2f} ft."
            if supported_max is not None else " No records exist for this roof, roofing material, and family."
        )
        middle_support_detail = ""
        alternate_bay = overall_span / 2.0
        if (
            not center_support_requested
            and family_kind != "half_a"
            and any(
                _row_supports_middle_line(item[3])
                and item[2] <= alternate_bay <= item[1]
                for item in all_matching
            )
        ):
            middle_support_detail = (
                f" The UI has center support OFF; the reference data allows a middle-support "
                f"configuration for {alternate_bay:.2f} ft bays when that UI option is enabled."
            )
        raise ValueError(
            f"No reference geometry rule matches {family}, {roof.replace('_', ' ')}, "
            f"{roofing}, and {rule_span:.2f} ft {('overall truss span' if family_kind == 'half_a' else 'unsupported bay span')} "
            f"(overall width {overall_span:.2f} ft).{limit_detail}{middle_support_detail}"
        )
    _, span_max, span_min, rule = selected
    generated_layout = bool(inputs.get("generated_support_layout", False))
    if center_support_requested and not _row_supports_middle_line(rule):
        raise ValueError(
            f"Reference rule {rule.get('design_id', '')} does not authorize an intermediate support configuration."
        )
    if generated_layout and not _row_supports_middle_line(rule):
        raise ValueError(
            f"Reference rule {rule.get('design_id', '')} does not authorize the supported-bay design required by this generated layout."
        )
    overhang_text = str(rule.get("overhang_ft", "")).strip()
    overhang_values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", overhang_text)]
    if not overhang_values:
        raise ValueError(
            f"The reference rule for {family} at {span:.2f} ft has no numeric overhang guidance."
        )
    overhang_default = overhang_values[0]
    configured_overhang_max = overhang_values[-1]
    # The application requirement permits 0–4 ft at each eave. Keep the
    # dataset value as diagnostic context, but do not let an old 2 ft row
    # silently override the user-configured application limit.
    overhang_max = MAX_OVERHANG_FT
    purlin_spacing_text = str(rule.get("purlin_spacing_in", "")).strip()
    purlin_spacing_numbers = [
        float(value) for value in re.findall(r"\d+(?:\.\d+)?", purlin_spacing_text)
    ]
    if "in" in purlin_spacing_text.casefold() and purlin_spacing_numbers:
        if "average" in purlin_spacing_text.casefold():
            purlin_spacing_target_ft = purlin_spacing_numbers[0] / 12.0
            purlin_spacing_limit_ft = min(purlin_spacing_numbers) / 12.0
        else:
            purlin_spacing_target_ft = min(purlin_spacing_numbers) / 12.0
            purlin_spacing_limit_ft = purlin_spacing_target_ft
    elif "ft" in purlin_spacing_text.casefold() and purlin_spacing_numbers:
        purlin_spacing_limit_ft = (
            3.0 if overall_span >= 30.0 and len(purlin_spacing_numbers) >= 3
            else purlin_spacing_numbers[0]
        )
        purlin_spacing_target_ft = purlin_spacing_limit_ft
    else:
        purlin_spacing_target_ft = None
        purlin_spacing_limit_ft = None
    truss_spacing_text = str(rule.get("truss_spacing_ft", "")).strip()
    truss_spacing_numbers = [
        float(value) for value in re.findall(r"\d+(?:\.\d+)?", truss_spacing_text)
    ]
    truss_spacing_limit_ft = max(truss_spacing_numbers) if truss_spacing_numbers else None
    truss_spacing_min_ft = min(truss_spacing_numbers) if truss_spacing_numbers else None
    if truss_spacing_limit_ft is None or truss_spacing_limit_ft <= 0:
        raise ValueError(
            f"Reference rule {rule.get('design_id', '')} has no usable longitudinal truss spacing."
        )
    if family_kind == "plain":
        web_spacing_limit_ft, web_spacing_source = None, "Not applicable"
    else:
        web_spacing_limit_ft, web_spacing_source = _reference_web_spacing_ft(
            {
                "reference_notes": str(rule.get("notes", "")).strip(),
                "web_spacing_ft": str(rule.get("web_spacing_ft", "")).strip(),
                "web_spacing_rule": str(rule.get("web_spacing_rule", "")).strip(),
            },
            inputs,
            family_kind,
            values,
        )
    purlin_alias = str(rule.get("purlin_pipe_alias", "")).strip()
    # Purlins span between adjacent trusses along the building length. Their
    # catalog span band must therefore use the resolved longitudinal truss pitch,
    # not the building's roof width / unsupported truss bay. Using roof width here
    # incorrectly rejects otherwise valid 30+ ft roofs for short purlin spans.
    predicted_purlin_span = _number(values, "truss_spacing_ft")
    if (
        predicted_purlin_span is None
        or predicted_purlin_span < truss_spacing_min_ft - 1e-9
        or predicted_purlin_span > truss_spacing_limit_ft + 1e-9
    ):
        purlin_span_for_alias = truss_spacing_limit_ft
    else:
        purlin_span_for_alias = predicted_purlin_span
    purlin_resolution = _resolve_pipe_reference_alias(
        purlin_alias,
        values.get("purlin_pipe_size", ""),
        overall_span=purlin_span_for_alias,
        unsupported_span=purlin_span_for_alias,
        column_height_ft=_number(inputs, "column_height_B_ft"),
    )
    resolved_purlin_alias = purlin_resolution["selected_section"]
    requested_overhang = _number(inputs, "overhang_ft")
    if requested_overhang is None:
        requested_overhang = _number(inputs, "overhang_length_ft")
    if requested_overhang is None:
        requested_overhang = _number(values, "overhang_ft", overhang_default)
    if requested_overhang < 0 or requested_overhang > overhang_max:
        raise ValueError(
            f"Requested roof overhang ({requested_overhang:.2f} ft) is outside the "
            f"supported range 0 to {overhang_max:.2f} ft for {family}."
        )
    return {
        "span_min_ft": span_min,
        "span_max_ft": span_max,
        "web_layout": str(rule.get("web_layout", "")).strip(),
        "web_spacing_limit_ft": web_spacing_limit_ft,
        "web_spacing_source": web_spacing_source,
        "web_spacing_rule": str(rule.get("web_spacing_rule", "")).strip(),
        "web_spacing_reference": str(rule.get("web_spacing_ft", "")).strip(),
        "roof_sides": str(rule.get("roof_sides", "")).strip(),
        "overall_span_ft": overall_span,
        "unsupported_span_ft": support_span,
        "overhang_default_ft": overhang_default,
        "overhang_max_ft": overhang_max,
        "configured_overhang_max_ft": configured_overhang_max,
        "overhang_ft": requested_overhang,
        "purlin_spacing_rule": purlin_spacing_text,
        "purlin_spacing_target_ft": purlin_spacing_target_ft,
        "purlin_spacing_limit_ft": purlin_spacing_limit_ft,
        "ridge_clearance_rule_ft": None,
        "ridge_clearance_note": "Not configured in the reference dataset.",
        "tile_weight_kg": str(rule.get("tile_weight_kg", "")).strip(),
        "middle_support_required": str(rule.get("middle_support_required", "")).strip(),
        "support_count": str(rule.get("support_count", "")).strip(),
        "support_spacing_ft": str(rule.get("support_spacing_ft", "")).strip(),
        "truss_spacing_rule": truss_spacing_text,
        "truss_spacing_min_ft": truss_spacing_min_ft,
        "truss_spacing_limit_ft": truss_spacing_limit_ft,
        "middle_support_pipe_alias": str(rule.get("middle_support_pipe_alias", "")).strip(),
        "pipe_aliases": {
            "frame_chord_pipe_size": str(rule.get("main_pipe_alias", "")).strip(),
            "web_pipe_size": str(rule.get("web_pipe_alias", "")).strip(),
            "support_pipe_size": str(rule.get("support_pipe_alias", "")).strip(),
            "column_pipe_size": str(rule.get("column_pipe_alias", "")).strip(),
            "beam_pipe_size": str(rule.get("beam_pipe_alias", "")).strip(),
            "middle_support_pipe_size": str(rule.get("middle_support_pipe_alias", "")).strip(),
            "purlin_pipe_size": resolved_purlin_alias,
        },
        "reference_design_id": str(rule.get("design_id", "")).strip(),
        "reference_notes": str(rule.get("notes", "")).strip(),
        "reference_selection": {
            "design_id": str(rule.get("design_id", "")).strip(),
            "requested_span_ft": rule_span,
            "span_basis": (
                "overall truss span" if family_kind == "half_a"
                else "unsupported bay span"
            ),
            "selected_span_range_ft": [span_min, span_max],
            "overlapping_design_ids": [
                str(item[3].get("design_id", "")).strip() for item in candidates
            ],
            "selection_rule": (
                "Narrowest containing span interval; ties use the greatest lower bound, "
                "then ascending design_id."
            ),
        },
        "pipe_resolution_diagnostics": {
            "purlin_pipe_size": purlin_resolution,
            "purlin_pipe_span_basis": "resolved longitudinal truss spacing",
            "purlin_pipe_span_ft": purlin_span_for_alias,
        },
    }



def _sheet_purlin_section_for_span(span_ft):
    """User-configured sheet-roof purlin tiers based on total truss span.

    These are project selection rules, not a substitute for load/capacity checks.
    """
    by_id, _ = _pipe_catalog_index()
    if span_ft < 9.0:
        pipe_id = "p001"
    elif span_ft <= 16.0:
        pipe_id = "p002"
    elif span_ft <= 80.0:
        pipe_id = "p005"
    else:
        pipe_id = "p006"
    row = by_id.get(pipe_id)
    if not row:
        raise ValueError(f"Required sheet purlin catalog section {pipe_id.upper()} is missing from pipes.csv.")
    return row["pipe_size"]


def _section_fits_within(inner_section, outer_section):
    """True when each ordered dimension of inner is no larger than outer."""
    inner = sorted(_section_dimensions_in(inner_section))
    outer = sorted(_section_dimensions_in(outer_section))
    return inner[0] <= outer[0] + 1e-9 and inner[1] <= outer[1] + 1e-9


def _cap_web_section_to_chord(web_section, chord_section):
    """Prevent web member section dimensions from exceeding the chord envelope."""
    if _section_fits_within(web_section, chord_section):
        return web_section
    by_id, _ = _pipe_catalog_index()
    candidates = []
    for row in by_id.values():
        section = row["pipe_size"]
        if _section_fits_within(section, chord_section):
            dims = _section_dimensions_in(section)
            area_proxy = dims[0] * dims[1]
            candidates.append((area_proxy, float(row.get("weight_kg") or 0.0), section))
    if not candidates:
        raise ValueError(
            f"No catalog web pipe fits within chord section '{chord_section}'. "
            "Update pipes.csv or the reference rule before generating this design."
        )
    return max(candidates)[2]


def _equalized_slope_positions(start_x, end_x, slope_length_ft, target_pitch_ft):
    """Equally distribute rows between boundaries using a reference target pitch."""
    if slope_length_ft <= 0 or target_pitch_ft <= 0:
        raise ValueError("Equalized purlin layout requires positive slope length and pitch.")
    intervals = max(1, int(round(slope_length_ft / target_pitch_ft)))
    positions = [
        round(start_x + (end_x - start_x) * i / intervals, 7)
        for i in range(intervals + 1)
    ]
    return positions, 0.0

def _inclusive_positions(extent, maximum_spacing=None, requested_positions=None):
    spacing = _number({"spacing": maximum_spacing}, "spacing")
    if spacing is not None and spacing > 0:
        intervals = max(1, int(math.ceil(extent / spacing)))
    else:
        intervals = max(1, int(requested_positions or 2) - 1)
    positions = [extent * index / intervals for index in range(intervals + 1)]
    return positions, extent / intervals


def _column_positions_from_prediction(
    length, predicted_spacing, predicted_total, support_line_count
):
    """Resolve equal, end-inclusive column stations from the spacing target.

    A positive predicted spacing is treated as the maximum permitted station
    interval: choose enough intervals that the actual equal spacing cannot
    exceed that target, then distribute those intervals evenly over the full
    building length so both ends are supported. The predicted total count is
    only a compatibility fallback when no usable spacing prediction exists.
    """
    if length is None or not math.isfinite(length) or length <= 0:
        raise ValueError("Building length must be a finite positive value.")
    if not isinstance(support_line_count, int) or support_line_count < 2:
        raise ValueError("Column layout requires at least two support lines.")

    spacing = _number({"spacing": predicted_spacing}, "spacing")
    if spacing is not None and spacing > 0:
        ratio = length / spacing
        nearest_integer = round(ratio)
        # ML targets may be rounded (e.g. 25 / 7 reported as 3.571).
        # Snap only tiny ratio-rounding noise; otherwise spacing is a hard maximum.
        if nearest_integer >= 1 and abs(ratio - nearest_integer) <= 1e-3:
            intervals = int(nearest_integer)
        else:
            intervals = max(1, int(math.ceil(ratio)))
        stations_per_line = intervals + 1
    else:
        if predicted_total is None or predicted_total <= 0:
            raise ValueError(
                "A positive column_spacing_along_length_ft prediction is required; "
                "column_total_count can only be used as a fallback when it is valid."
            )
        if support_line_count == 2:
            final_total = int(round(predicted_total))
            if final_total % 2:
                final_total += 1
            stations_per_line = max(2, final_total // 2)
        else:
            stations_per_line = max(
                2, int(math.floor(predicted_total / support_line_count + 0.5))
            )

    positions = [length * index / (stations_per_line - 1) for index in range(stations_per_line)]
    final_total = stations_per_line * support_line_count
    return positions, final_total


def _purlin_rule(inputs, values, mode, reference_rule):
    label = str(inputs.get("roofing_type", "")).strip().casefold()
    config = next((rule for name, rule in ROOFING_RULES.items() if name.casefold() == label), None)
    if config is None:
        raise ValueError(f"No purlin spacing rule is configured for roofing type '{inputs.get('roofing_type', '')}'.")

    # Tile roofs use a fixed nominal 9-inch pitch. The end remainder is handled
    # by the purlin-coordinate generator and is reported separately.
    roofing_hard_cap_ft = config["max_purlin_spacing_ft"]
    if "tile" in label:
        return 0.75, 0.75, "Fixed tile pitch: 9 inches (end remainder reported separately)"

    if "sheet" in label:
        # 3.0-3.5 ft is a reference range, not a hard validation limit.
        # Prefer a model prediction inside that range; otherwise use the midpoint.
        predicted_gap = _number(values, "purlin_spacing_ft")
        target_gap = 3.25 if predicted_gap is None or predicted_gap <= 0 else min(3.5, max(3.0, predicted_gap))
        return target_gap, 3.5, "Metal-sheet reference range 3.0-3.5 ft; equalized layout, not a hard cap"

    if reference_rule.get("purlin_spacing_limit_ft") is not None:
        permitted_spacing = min(
            reference_rule["purlin_spacing_limit_ft"],
            roofing_hard_cap_ft,
        )
        default_spacing = reference_rule.get("purlin_spacing_target_ft") or permitted_spacing
        spacing = _number(values, "purlin_spacing_ft", default_spacing)
        if spacing is None or spacing <= 0:
            spacing = default_spacing
        return (
            spacing,
            permitted_spacing,
            f"Reference dataset rule capped by roofing type max ({roofing_hard_cap_ft:.2f} ft): {reference_rule['purlin_spacing_rule']}",
        )

    # No reference rule, use the hard cap directly
    spacing = _number(values, "purlin_spacing_ft")
    if spacing is None or spacing <= 0:
        spacing = roofing_hard_cap_ft
    source = "ML prediction" if str(mode).upper() == "AUTO" else "User specified"
    return spacing, roofing_hard_cap_ft, source


def _fixed_pitch_slope_positions(start_x, end_x, usable_slope_length_ft, pitch_ft):
    """Return fixed-pitch row coordinates and the unfilled far-end remainder in inches."""
    horizontal_run_ft = abs(end_x - start_x)
    if horizontal_run_ft <= 0.0 or usable_slope_length_ft <= 0.0 or pitch_ft <= 0.0:
        raise ValueError("Purlin layout requires positive roof run, slope length, and pitch.")
    interval_count = int(math.floor((usable_slope_length_ft + 1e-9) / pitch_ft))
    positions = [
        round(start_x + (end_x - start_x) * (index * pitch_ft / usable_slope_length_ft), 7)
        for index in range(interval_count + 1)
    ]
    remainder_ft = max(0.0, usable_slope_length_ft - interval_count * pitch_ft)
    return positions, remainder_ft * 12.0


def _resolve_sheet_center_pitch(predicted_clear_gap_ft, maximum_clear_gap_ft, face_width_in, minimum_clear_gap_ft=3.0):
    """Resolve centre pitch from a clear-gap prediction and clear-gap limits."""
    if maximum_clear_gap_ft + 1e-9 < minimum_clear_gap_ft:
        raise ValueError(
            "Metal Sheet purlin rules conflict: the required minimum "
            f"{minimum_clear_gap_ft:.1f} ft clear gap exceeds the current "
            f"reference/structural maximum clear gap of {maximum_clear_gap_ft:.3f} ft. "
            "Review the reference spacing rule; no unsafe layout was generated."
        )
    if predicted_clear_gap_ft is None or predicted_clear_gap_ft <= 0:
        predicted_clear_gap_ft = minimum_clear_gap_ft
    clear_gap_ft = min(maximum_clear_gap_ft, max(minimum_clear_gap_ft, predicted_clear_gap_ft))
    return clear_gap_ft + face_width_in / 12.0


def _purlin_face_width_in(values):
    """Return the profile width along the roof slope from the upward-facing orientation."""
    orientation = str(values.get("purlin_orientation", "")).strip()
    # Examples: 'Larger 1.5" face upward', '4.0" face upward flat'.
    match = re.search(r"(\d+(?:\.\d+)?)\s*[\"”]?\s*face\s+upward", orientation, flags=re.IGNORECASE)
    if match:
        width = float(match.group(1))
        if width > 0:
            return width, "orientation label"
    raise ValueError(
        "The selected purlin orientation does not specify an upward face width; "
        "purlin spacing cannot be calculated without that configured dimension."
    )


def generate_design(inputs, values, mode="AUTO"):
    """Build the sole source of member coordinates used by all visual/material views."""
    span = _number(inputs, "site_width_ft")
    length = _number(inputs, "site_length_ft")
    column_height = _number(inputs, "column_height_B_ft")
    rise = _number(values, "rise_height_ft")
    if span is None or length is None or column_height is None or rise is None:
        raise ValueError("Span, building length, column height, and a real rise value are required.")
    if min(span, length, column_height, rise) <= 0:
        raise ValueError("Span, building length, column height, and rise must be greater than zero.")
    if column_height > MAX_COLUMN_HEIGHT_FT + 1e-9:
        raise ValueError(
            f"Column height is limited to {MAX_COLUMN_HEIGHT_FT:.0f} ft; "
            f"received {column_height:.2f} ft."
        )

    roof = _roof_name(inputs)
    gable_end_overhang = _number(inputs, "gable_end_overhang_ft", 0.0)
    if gable_end_overhang is None or not 0.0 <= gable_end_overhang <= MAX_OVERHANG_FT:
        raise ValueError(
            f"Gable-end overhang must be between 0 and {MAX_OVERHANG_FT:.1f} ft."
        )
    if gable_end_overhang > 0:
        raise ValueError(
            "Gable-end overhang framing is unconfigured in the project reference dataset; "
            "only eave overhang is generated."
        )
    family = str(values.get("truss_family", ""))
    family_kind = _family_kind(family)
    predicted_spacing = _number(values, "truss_spacing_ft")
    predicted_truss_count = max(1, _count(values, "truss_total_count", 1))
    compatibility_error = family_roof_compatibility_error(family, roof)
    if compatibility_error:
        raise ValueError(compatibility_error)
    has_center_support = str(inputs.get("center_support_present", "No")).strip().casefold() in {"yes", "true", "1"}
    explicit_support_positions = inputs.get("support_line_positions_ft")
    if isinstance(explicit_support_positions, (list, tuple)) and len(explicit_support_positions) >= 2:
        try:
            requested_lanes = sorted({float(x) for x in explicit_support_positions})
        except (TypeError, ValueError):
            raise ValueError("support_line_positions_ft must contain numeric positions.")
        if requested_lanes[0] < -1e-7 or requested_lanes[-1] > span + 1e-7:
            raise ValueError("Support line positions must lie within the overall span.")
        if not math.isclose(requested_lanes[0], 0.0, abs_tol=1e-6) or not math.isclose(requested_lanes[-1], span, abs_tol=1e-6):
            raise ValueError("Support line positions must include both outer supports at 0 and the overall span.")
        if any(b - a <= 1e-6 for a, b in zip(requested_lanes, requested_lanes[1:])):
            raise ValueError("Support line positions must be strictly increasing.")
    else:
        requested_lanes = [0.0, span / 2.0, span] if has_center_support else [0.0, span]
    overhang_hint = _number(inputs, "overhang_ft")
    if overhang_hint is None:
        overhang_hint = _number(inputs, "overhang_length_ft")
    if overhang_hint is None:
        overhang_hint = _number(values, "overhang_ft", MAX_OVERHANG_FT)
    bearing_inset_hint = min(MAX_OVERHANG_FT, max(0.0, overhang_hint))
    support_left_hint = 0.0
    support_right_hint = span
    column_lanes = [support_left_hint, *[x for x in requested_lanes[1:-1]], support_right_hint]
    column_lanes = sorted(set(column_lanes))
    supported_bay_spans = [
        second - first for first, second in zip(column_lanes, column_lanes[1:])
    ]
    unsupported_span = max(supported_bay_spans)
    reference_inputs = {**inputs, "unsupported_span_ft": unsupported_span}
    reference_rule = _reference_geometry_rule(
        reference_inputs, values, family, family_kind, roof, span
    )
    overhang = reference_rule["overhang_ft"]
    support_left = 0.0
    support_right = span
    column_lanes = [support_left, *[x for x in requested_lanes[1:-1]], support_right]
    column_lanes = sorted(set(column_lanes))
    supported_bay_spans = [
        second - first for first, second in zip(column_lanes, column_lanes[1:])
    ]
    unsupported_span = max(supported_bay_spans)
    reference_rule["unsupported_span_ft"] = unsupported_span
    resolved_truss_spacing = reference_rule["truss_spacing_limit_ft"]
    truss_spacing_min = reference_rule["truss_spacing_min_ft"]
    if predicted_spacing is not None and truss_spacing_min <= predicted_spacing <= resolved_truss_spacing:
        truss_spacing_target = predicted_spacing
        truss_spacing_source = "ML prediction within the matched reference range"
    else:
        truss_spacing_target = resolved_truss_spacing
        truss_spacing_source = "Matched reference maximum"
    truss_positions, effective_spacing = _inclusive_positions(
        length, truss_spacing_target, None
    )
    truss_count = len(truss_positions)
    if family_kind == "plain":
        web_spacing_target = None
        web_spacing_source = "Not applicable"
        web_panels = 2 if roof == "gable" else 1
    else:
        web_spacing_target = reference_rule["web_spacing_limit_ft"]
        web_spacing_source = reference_rule["web_spacing_source"]
        predicted_web_spacing = _number(values, "web_spacing_ft")
        if predicted_web_spacing is not None and predicted_web_spacing > 0:
            web_spacing_target = min(predicted_web_spacing, web_spacing_target)
            if predicted_web_spacing > reference_rule["web_spacing_limit_ft"]:
                web_spacing_source = (
                    "Reference-constrained ML prediction; prediction exceeded "
                    "the maximum spacing recorded in the matched design"
                )
            else:
                web_spacing_source = "ML prediction within the matched reference limit"
        web_panels = max(1, int(math.ceil(span / web_spacing_target - 1e-12)))
        if roof == "gable" and web_panels % 2:
            web_panels += 1
    if web_panels > 128:
        raise ValueError(f"Predicted web layout requires {web_panels} panels; the supported limit is 128.")
    pipe_aliases = reference_rule["pipe_aliases"]
    chord_section = _pipe_size_from_reference_alias(
        pipe_aliases["frame_chord_pipe_size"], values.get("frame_chord_pipe_size", ""),
        overall_span=span, unsupported_span=unsupported_span, column_height_ft=column_height,
    )
    web_section = "" if family_kind == "plain" else _pipe_size_from_reference_alias(
        pipe_aliases["web_pipe_size"], values.get("web_pipe_size", ""),
        overall_span=span, unsupported_span=unsupported_span, column_height_ft=column_height,
    )
    requested_web_section = web_section
    if web_section:
        web_section = _cap_web_section_to_chord(web_section, chord_section)
    column_section = _pipe_size_from_reference_alias(
        pipe_aliases["column_pipe_size"], values.get("column_pipe_size", chord_section),
        overall_span=span, unsupported_span=unsupported_span, column_height_ft=column_height,
    )
    beam_section = _pipe_size_from_reference_alias(
        pipe_aliases["beam_pipe_size"], values.get("beam_pipe_size", chord_section),
        overall_span=span, unsupported_span=unsupported_span, column_height_ft=column_height,
    )
    middle_support_section = _pipe_size_from_reference_alias(
        pipe_aliases["middle_support_pipe_size"] or pipe_aliases["support_pipe_size"],
        column_section, overall_span=span, unsupported_span=unsupported_span,
        column_height_ft=column_height,
    )
    purlin_section = _pipe_size_from_reference_alias(
        pipe_aliases["purlin_pipe_size"],
        values.get("purlin_pipe_size", ""),
        overall_span=span,
        unsupported_span=unsupported_span,
        column_height_ft=column_height,
    )
    roofing_label_for_size = str(inputs.get("roofing_type", "")).strip().casefold()
    resolved_purlin_orientation = str(values.get("purlin_orientation", "")).strip()
    if "metal sheet" in roofing_label_for_size:
        purlin_section = _sheet_purlin_section_for_span(span)
        purlin_dims = _section_dimensions_in(purlin_section)
        upward_face = max(purlin_dims)
        resolved_purlin_orientation = (
            f'Larger {upward_face:g}" face upward (flat) for broad sheet seating and screw grip'
        )
    if not chord_section:
        raise ValueError("A frame chord pipe section is required to generate material quantities.")
    if family_kind != "plain" and not web_section:
        raise ValueError(f"A web pipe section is required for the {family} family.")
    chord_depth = _section_depth_ft(chord_section)
    beam_depth = _section_depth_ft(beam_section)
    purlin_depth = _section_depth_ft(
        purlin_section, resolved_purlin_orientation
    )
    bearing_chord_base_z = column_height + beam_depth + chord_depth / 2.0

    members = []
    member_keys = set()
    zero_length_count = 0
    duplicate_member_count = 0

    def add(member_type, section, start, end, truss_index=None, attributes=None):
        nonlocal zero_length_count, duplicate_member_count
        start = tuple(float(v) for v in start)
        end = tuple(float(v) for v in end)
        length_ft = math.dist(start, end)
        if length_ft <= 1e-8:
            zero_length_count += 1
            return
        endpoint_a = tuple(round(v, 7) for v in start)
        endpoint_b = tuple(round(v, 7) for v in end)
        endpoints = tuple(sorted((endpoint_a, endpoint_b)))
        member_key = (member_type, str(section), endpoints, truss_index)
        if member_key in member_keys:
            duplicate_member_count += 1
            return
        member_keys.add(member_key)
        member = {
            "member_id": len(members) + 1,
            "member_type": member_type,
            "pipe_section": str(section),
            "start": start,
            "end": end,
            "length_ft": length_ft,
            "truss_index": truss_index,
            **(attributes or {}),
        }
        members.append(member)
        return member

    # Family-specific front-elevation geometry based on the supplied Solid Edge
    # reference sketches. The predicted web count is diagnostic only; it never
    # creates arbitrary members. Every member below is a real coordinate pair.
    overhang = reference_rule["overhang_ft"]
    support_left = 0.0
    support_right = span
    overhang_left = -overhang if overhang > 0.0 else 0.0
    overhang_right = span + overhang
    roof_edge_overhang = overhang
    if family_kind == "box":
        box_depth_ft, box_depth_source = _box_depth_from_reference(
            reference_rule, reference_inputs, family_kind
        )
        box_depth = box_depth_ft
    else:
        box_depth = 0.0
        box_depth_source = "Not applicable"

    def top_z(x):
        roof_base = bearing_chord_base_z + box_depth
        return roof_base + _profile(x, span, rise, roof)

    def support_z(x):
        if family_kind == "plain":
            return top_z(x)
        if family_kind == "box":
            return top_z(x) - box_depth
        return bearing_chord_base_z

    support_joint_xs = [x for x in column_lanes[1:-1]]
    for truss_index, y in enumerate(truss_positions):
        if family_kind == "plain":
            xs = (
                sorted(set([overhang_left, 0.0, *support_joint_xs, span, overhang_right]))
                if roof == "single_slope"
                else sorted(set([overhang_left, 0.0, span / 2.0, *support_joint_xs, span, overhang_right]))
            )
            top = [(x, top_z(x)) for x in xs]
            for a, b in zip(top, top[1:]):
                add("Top chord", chord_section, (a[0], y, a[1]), (b[0], y, b[1]), truss_index)
            continue

        # Use one consistent roof profile for the truss family, and reuse the same
        # eave and ridge points for adjoining members. This keeps the lower chord,
        # top chord, and webs connected at their intended structural joints.
        panel_xs = [span * i / web_panels for i in range(web_panels + 1)]
        # Half-A uses a deliberately shortened, centered bottom chord. The
        # supplied Solid Edge sketch is approximately 2/3 of the full span;
        # this scales proportionally (within the requested 50–75% range).
        half_a_bottom_left = span / 6.0
        half_a_bottom_right = span - half_a_bottom_left
        extra_top_joints = (
            [half_a_bottom_left, half_a_bottom_right]
            if family_kind == "half_a" else []
        )
        reliable_top_xs = sorted(set(
            [overhang_left, 0.0, *panel_xs, *extra_top_joints, *support_joint_xs,
             *([span / 2.0] if roof == "gable" else []), span, overhang_right]
            if roof == "gable"
            else [overhang_left, 0.0, *panel_xs, *support_joint_xs, span, overhang_right]
        ))
        top = [(x, top_z(x)) for x in reliable_top_xs]

        if family_kind == "box":
            lower = [(x, z - box_depth) for x, z in top]
            for a, b in zip(top, top[1:]):
                add("Top chord", chord_section, (a[0], y, a[1]), (b[0], y, b[1]), truss_index)
            for a, b in zip(lower, lower[1:]):
                add("Bottom chord", chord_section, (a[0], y, a[1]), (b[0], y, b[1]), truss_index)
            zigzag = "zig" in reference_rule["web_layout"].casefold()
            # Keep a vertical at every panel joint; diagonals supplement rather
            # than replace vertical web members in the resolved frame. 
            vertical_indices = list(range(len(top)))
            for i in vertical_indices:
                add("Web vertical", web_section,
                    (lower[i][0], y, lower[i][1]), (top[i][0], y, top[i][1]), truss_index)
            if zigzag:
                for i in range(len(top) - 1):
                    a, b = (lower[i], top[i + 1]) if i % 2 == 0 else (top[i], lower[i + 1])
                    add("Web diagonal", web_section, (a[0], y, a[1]), (b[0], y, b[1]), truss_index)
            continue

        for a, b in zip(top, top[1:]):
            is_half_a_eave_extension = family_kind == "half_a" and (
                (a[0] < support_left and b[0] <= support_left + 1e-7)
                or (a[0] >= support_right - 1e-7 and b[0] > support_right)
            )
            add(
                "Top chord", chord_section, (a[0], y, a[1]), (b[0], y, b[1]),
                truss_index,
                {"overhang_frame": True} if is_half_a_eave_extension else None,
            )

        if family_kind == "semi":
            lower_xs = sorted(set([0.0, *panel_xs[1:-1], *support_joint_xs, span]))
            lower = [(x, support_z(x)) for x in lower_xs]
            for a, b in zip(lower, lower[1:]):
                add("Bottom chord", chord_section, (a[0], y, a[1]), (b[0], y, b[1]), truss_index)
            if "zig" in reference_rule["web_layout"].casefold():
                # Vertical web at each lower-chord joint, with diagonal bracing
                # between adjacent panels.
                for x in lower_xs[1:]:
                    add("Web vertical", web_section,
                        (x, y, support_z(x)), (x, y, top_z(x)), truss_index)
                for i in range(len(lower_xs) - 1):
                    a_x, b_x = lower_xs[i], lower_xs[i + 1]
                    if i % 2 == 0:
                        add("Web diagonal", web_section,
                            (a_x, y, support_z(a_x)), (b_x, y, top_z(b_x)), truss_index)
                    else:
                        add("Web diagonal", web_section,
                            (a_x, y, top_z(a_x)), (b_x, y, support_z(b_x)), truss_index)
            else:
                # The low-end top and bottom joints coincide; the high-end joint
                # needs a vertical to close the roof profile to the lower chord.
                for x in lower_xs[1:]:
                    add("Web vertical", web_section,
                        (x, y, support_z(x)), (x, y, top_z(x)), truss_index)
            continue

        if family_kind == "full_a":
            lower_xs = sorted(set([0.0, *[x for x in panel_xs[1:-1] if 0.0 < x < span], *support_joint_xs, span]))
            lower = [(x, bearing_chord_base_z) for x in lower_xs]
            for a, b in zip(lower, lower[1:]):
                add("Bottom chord", chord_section, (a[0], y, a[1]), (b[0], y, b[1]), truss_index)
            zigzag = "zig" in reference_rule["web_layout"].casefold()
            for x in lower_xs:
                if x in {0.0, span}:
                    continue
                if math.isclose(x, span / 2.0, abs_tol=1e-7):
                    add("Web king post", web_section,
                        (x, y, bearing_chord_base_z), (x, y, top_z(x)), truss_index)
                else:
                    add("Web vertical", web_section,
                        (x, y, bearing_chord_base_z), (x, y, top_z(x)), truss_index)
            if zigzag:
                for i, (a, b) in enumerate(zip(lower_xs, lower_xs[1:])):
                    if i % 2 == 0:
                        add("Web diagonal", web_section,
                            (a, y, bearing_chord_base_z), (b, y, top_z(b)), truss_index)
                    else:
                        add("Web diagonal", web_section,
                            (a, y, top_z(a)), (b, y, bearing_chord_base_z), truss_index)
            continue

        if family_kind == "half_a":
            # The bottom chord is inset from both eaves; columns/bearings stay
            # at x=0 and x=span. For a 20 ft span this produces ~13.33 ft of
            # bottom chord, matching the supplied reference sketch.
            lower_left, lower_right = half_a_bottom_left, half_a_bottom_right
            lower_xs = sorted(set([
                lower_left,
                *[x for x in panel_xs[1:-1] if lower_left < x < lower_right],
                *[x for x in support_joint_xs if lower_left < x < lower_right],
                span / 2.0,
                lower_right,
            ]))
            lower = [(x, bearing_chord_base_z) for x in lower_xs]
            for a, b in zip(lower, lower[1:]):
                add("Bottom chord", chord_section, (a[0], y, a[1]), (b[0], y, b[1]), truss_index)
            # Vertical webs are required at every eligible lower-chord joint,
            # including both inset chord ends. Their upper endpoints are real
            # top-chord joints because the inset positions were added above.
            for x in lower_xs:
                member_type = "Web king post" if math.isclose(x, span / 2.0, abs_tol=1e-7) else "Web vertical"
                add(member_type, web_section,
                    (x, y, bearing_chord_base_z), (x, y, top_z(x)), truss_index)
            zigzag = "zig" in reference_rule["web_layout"].casefold()
            if zigzag:
                for i, (a, b) in enumerate(zip(lower_xs, lower_xs[1:])):
                    if i % 2 == 0:
                        add("Web diagonal", web_section,
                            (a, y, bearing_chord_base_z), (b, y, top_z(b)), truss_index)
                    else:
                        add("Web diagonal", web_section,
                            (b, y, bearing_chord_base_z), (a, y, top_z(a)), truss_index)
            continue

    column_lanes = [support_left, *[x for x in requested_lanes[1:-1]], support_right]
    column_lanes = sorted(set(column_lanes))
    predicted_column_spacing = _number(values, "column_spacing_along_length_ft")
    raw_predicted_column_count = _number(values, "column_total_count")
    predicted_column_count = raw_predicted_column_count

    # Project-requested roofing-specific longitudinal column spacing policy:
    # Metal Sheet: use 10 ft as the maximum station interval.
    # Clay / Concrete Tile: use 6 ft maximum (within the requested 5-6 ft range).
    # The spacing model remains visible as a diagnostic, but its count is never
    # used to force the physical column layout. Other roofing types continue to
    # use the model-predicted spacing.
    roofing_label_for_columns = str(inputs.get("roofing_type", "")).strip().casefold()
    if roofing_label_for_columns == "metal sheet" or "sheet" in roofing_label_for_columns:
        column_spacing_target = 10.0
        column_spacing_source = "Roofing rule: Metal Sheet, 10 ft maximum interval"
    elif "tile" in roofing_label_for_columns:
        column_spacing_target = 6.0
        column_spacing_source = "Roofing rule: Tile, 5-6 ft range (6 ft maximum interval)"
    elif predicted_column_spacing is not None and predicted_column_spacing > 0:
        column_spacing_target = predicted_column_spacing
        column_spacing_source = "Model-predicted spacing, used as maximum interval"
    else:
        column_spacing_target = None
        column_spacing_source = "Model spacing unavailable; validated count fallback"

    column_y_positions, final_column_count = _column_positions_from_prediction(
        length,
        column_spacing_target,
        raw_predicted_column_count,
        len(column_lanes),
    )
    resolved_column_spacing = length / (len(column_y_positions) - 1)
    column_locations = []

    def beam_axis_z(x):
        if family_kind in {"plain", "box"}:
            return support_z(x) - (chord_depth + beam_depth) / 2.0
        return column_height + beam_depth / 2.0

    def column_top_z(x):
        return beam_axis_z(x) - beam_depth / 2.0

    if column_section:
        for x in column_lanes:
            support_top_z = column_top_z(x)
            section = (
                middle_support_section
                if x not in (support_left, support_right) and middle_support_section
                else column_section
            )
            for y in column_y_positions:
                column_location = (x, y, support_top_z)
                column_locations.append(column_location)
                add(
                    "Column", section, (x, y, 0.0), (x, y, support_top_z),
                    attributes={
                        "support_line": "outer" if math.isclose(x, support_left, abs_tol=1e-7) or math.isclose(x, support_right, abs_tol=1e-7) else "intermediate",
                        "top_bearing_surface_z_ft": support_top_z,
                    },
                )

    beam_lanes = list(column_lanes) if column_locations else []
    predicted_beam_count = _count(values, "beam_total_count", len(beam_lanes))
    # Beams run along the building length at each support line, providing bearing
    # for both columns and trusses. Include all unique Y positions to ensure
    # every truss has beam support while minimizing unneeded segments.
    beam_stations = sorted(set(truss_positions + column_y_positions))
    if beam_section:
        for run_index, x in enumerate(beam_lanes):
            z = beam_axis_z(x)
            segments = list(zip(beam_stations, beam_stations[1:]))
            for segment_index, (y0, y1) in enumerate(segments):
                add(
                    "Beam", beam_section, (x, y0, z), (x, y1, z),
                    attributes={
                        "beam_run_index": run_index,
                        "beam_segment_index": segment_index,
                        "section_depth_ft": beam_depth,
                        "bottom_surface_z_ft": z - beam_depth / 2.0,
                        "top_surface_z_ft": z + beam_depth / 2.0,
                    },
                )

    # Treat the ML purlin spacing as the CLEAR inner-to-inner gap. The model's
    # predicted count is diagnostic only; the actual row count comes from slope
    # length + clear gap + upward-facing profile width.
    predicted_purlin_spacing = _number(values, "purlin_spacing_ft")
    predicted_purlin_count = _count(values, "purlin_total_count", _count(values, "purlin_total_lines", 0))
    purlin_spacing_target, purlin_spacing_rule, purlin_spacing_source = _purlin_rule(
        inputs, values, mode, reference_rule
    )
    if not purlin_section:
        raise ValueError("A purlin pipe section is required to generate material quantities.")

    purlin_face_width_in, purlin_face_width_source = _purlin_face_width_in({**values, "purlin_orientation": resolved_purlin_orientation})
    roofing_label = str(inputs.get("roofing_type", "")).strip().casefold()
    roofing_config = next(
        (cfg for name, cfg in ROOFING_RULES.items() if name.casefold() == roofing_label),
        {},
    )
    spacing_is_center_to_center = roofing_config.get("spacing_is_center_to_center", False)

    if spacing_is_center_to_center:
        # Metal-sheet spacing is a reference target (3.0-3.5 ft clear gap),
        # not a hard cap. Rows are equally distributed between eave and ridge limits.
        target_clear_gap_ft = min(3.5, max(3.0, purlin_spacing_target))
        center_pitch_ft = target_clear_gap_ft + purlin_face_width_in / 12.0
        center_pitch_in = center_pitch_ft * 12.0
        effective_clear_gap_in = target_clear_gap_ft * 12.0
    else:
        # Tile rule: preserve exactly 9 in clear between purlin faces. Do not
        # squeeze every interval to force a row onto the ridge/end boundary;
        # any leftover is reported as a separate end remainder.
        effective_clear_gap_in = 9.0
        center_pitch_in = purlin_face_width_in + effective_clear_gap_in
        center_pitch_ft = center_pitch_in / 12.0
    horizontal_projection = (
        span / 2.0 + roof_edge_overhang
        if roof == "gable"
        else span + 2.0 * roof_edge_overhang
    )
    slope_rise = (
        rise * (1.0 + 2.0 * roof_edge_overhang / span)
        if roof == "gable"
        else rise * (span + 2.0 * roof_edge_overhang) / span
    )
    roof_slope_length = math.hypot(horizontal_projection, slope_rise)
    ridge_clearance = RIDGE_CLEARANCE_FT if roof == "gable" else 0.0
    purlin_usable_slope_length = roof_slope_length - ridge_clearance
    roof_slope_length_in = purlin_usable_slope_length * 12.0
    if roof_slope_length_in < purlin_face_width_in:
        raise ValueError("Roof slope is shorter than the selected purlin face width.")

    if spacing_is_center_to_center:
        warnings_prefix = None
    elif abs(purlin_spacing_target - 0.75) > 1e-9:
        warnings_prefix = (
            f"Tile-roofing model/reference spacing ({purlin_spacing_target:.3f} ft) was overridden "
            "by the required 0.750 ft (9 in) clear-gap rule."
        )
    else:
        warnings_prefix = None

    # Place rows at the exact permitted pitch from the eave/end datum. Do not
    # evenly redistribute intervals: doing so silently changes the clear gap.
    # The unused distance at the far end is returned and reported explicitly.
    center_pitch_ft = center_pitch_in / 12.0

    if roof == "gable":
        half_span = span / 2.0
        roof_half_run = half_span + roof_edge_overhang
        ridge_clearance_projection = ridge_clearance * roof_half_run / roof_slope_length
        left_ridge_boundary = half_span - ridge_clearance_projection
        right_ridge_boundary = half_span + ridge_clearance_projection
        left_run = left_ridge_boundary - overhang_left
        usable_slope_length_ft = purlin_usable_slope_length
        if spacing_is_center_to_center:
            left_positions, left_remainder_in = _equalized_slope_positions(
                overhang_left, left_ridge_boundary, usable_slope_length_ft, center_pitch_ft
            )
        else:
            left_positions, left_remainder_in = _fixed_pitch_slope_positions(
                overhang_left, left_ridge_boundary, usable_slope_length_ft, center_pitch_ft
            )
        right_positions = [round(span - x, 7) for x in left_positions]
        right_positions = sorted(set(right_positions))
        purlin_x_positions = sorted(set(left_positions + right_positions))
        purlin_sides = {round(x, 7): "left" for x in left_positions}
        purlin_sides.update({round(x, 7): "right" for x in right_positions})
        left_row_count = len(left_positions)
        right_row_count = len(right_positions)
        ridge_row_count = 0
        actual_slope_pitch_ft = center_pitch_ft
        top_remainder_in = left_remainder_in
    else:
        total_run = overhang_right - overhang_left
        if spacing_is_center_to_center:
            purlin_x_positions, top_remainder_in = _equalized_slope_positions(
                overhang_left, overhang_right, roof_slope_length, center_pitch_ft
            )
        else:
            purlin_x_positions, top_remainder_in = _fixed_pitch_slope_positions(
                overhang_left, overhang_right, roof_slope_length, center_pitch_ft
            )
        purlin_x_positions = [round(x, 7) for x in purlin_x_positions]
        purlin_sides = {round(x, 7): "single" for x in purlin_x_positions}
        left_row_count = len(purlin_x_positions)
        right_row_count = 0
        ridge_row_count = 0
        actual_slope_pitch_ft = center_pitch_ft

    purlin_longitudinal_segments = max(0, len(truss_positions) - 1)
    purlin_bearing_contacts = []

    def _top_chord_contact(x, y):
        for member in members:
            if member["member_type"] != "Top chord" or member["truss_index"] is None:
                continue
            if not math.isclose(member["start"][1], y, abs_tol=1e-7) or not math.isclose(
                member["end"][1], y, abs_tol=1e-7
            ):
                continue
            x0, x1 = member["start"][0], member["end"][0]
            if min(x0, x1) - 1e-7 <= x <= max(x0, x1) + 1e-7:
                fraction = (x - x0) / (x1 - x0)
                z = member["start"][2] + fraction * (member["end"][2] - member["start"][2])
                return member, z
        return None, None

    def _roof_angle_at_x(x):
        if roof == "gable":
            run = max(span / 2.0, 1e-9)
            return math.atan2(rise, run) * (1.0 if x < span / 2.0 else -1.0)
        return math.atan2(rise, max(span, 1e-9))

    for row_index, x in enumerate(purlin_x_positions, start=1):
        roof_angle_rad = _roof_angle_at_x(x)
        # Purlin thickness is measured normal to the sloped roof face, so its
        # vertical offset is the normal depth multiplied by cos(slope angle).
        z = top_z(x) + (chord_depth + purlin_depth) * math.cos(roof_angle_rad) / 2.0
        for segment_index, (y0, y1) in enumerate(zip(truss_positions, truss_positions[1:]), start=1):
            purlin_member = add(
                "Purlin", purlin_section, (x, y0, z), (x, y1, z),
                attributes={
                    "profile_orientation": values.get("purlin_orientation", "Not specified"),
                    "roof_slope": purlin_sides[round(x, 7)],
                    "cross_section_rotation_deg": math.degrees(roof_angle_rad),
                    "purlin_row_index": row_index,
                    "longitudinal_run_index": row_index,
                    "longitudinal_segment_index": segment_index,
                    "section_depth_ft": purlin_depth,
                },
            )
            if purlin_member:
                for y in (y0, y1):
                    chord_member, chord_contact_z = _top_chord_contact(x, y)
                    chord_surface_z = (
                        chord_contact_z + chord_depth * math.cos(roof_angle_rad) / 2.0
                        if chord_contact_z is not None else None
                    )
                    purlin_bottom_z = z - purlin_depth * math.cos(roof_angle_rad) / 2.0
                    purlin_bearing_contacts.append({
                        "purlin_member_id": purlin_member["member_id"],
                        "truss_member_id": (
                            chord_member["member_id"] if chord_member else None
                        ),
                        "truss_station_y_ft": y,
                        "roof_chord_surface_z_ft": chord_surface_z,
                        "purlin_bottom_surface_z_ft": purlin_bottom_z,
                        "aligned": chord_member is not None and math.isclose(
                            chord_surface_z, purlin_bottom_z,
                            rel_tol=0.0, abs_tol=1e-7,
                        ),
                    })
    # Actual spacing is the clear inner-to-inner gap. Center-to-center pitch is
    # also reported separately so the two meanings cannot be confused in the UI.
    # Compute actual clear gaps from generated purlin coordinates and profile width.
    actual_clear_gaps_ft = []
    if roof == "single_slope":
        slope_position_groups = [(purlin_x_positions, roof_slope_length, overhang_right - overhang_left)]
    else:
        slope_position_groups = [
            (left_positions, purlin_usable_slope_length, left_ridge_boundary - overhang_left),
            (right_positions, purlin_usable_slope_length, overhang_right - right_ridge_boundary),
        ]
    for slope_positions_list, usable_slope_ft, horizontal_run_ft in slope_position_groups:
        if horizontal_run_ft <= 0.0:
            continue
        slope_scale = usable_slope_ft / horizontal_run_ft
        for a, b in zip(slope_positions_list, slope_positions_list[1:]):
            center_distance_along_slope = abs(b - a) * slope_scale
            gap = center_distance_along_slope - purlin_face_width_in / 12.0
            actual_clear_gaps_ft.append(max(0.0, gap))
    actual_purlin_max_spacing = round(max(actual_clear_gaps_ft), 6) if actual_clear_gaps_ft else 0.0

    purlin_rows_per_slope = max(left_row_count, right_row_count) if roof == "gable" else left_row_count
    purlin_actual_center_pitch_ft = actual_slope_pitch_ft
    purlin_top_remainder_in = top_remainder_in

    purlin_layout = {
        "left_slope_row_count": left_row_count if roof == "gable" else 0,
        "right_slope_row_count": right_row_count if roof == "gable" else 0,
        "single_slope_row_count": left_row_count if roof == "single_slope" else 0,
        "total_slope_rows": len(purlin_x_positions),
        "ridge_row_count": ridge_row_count,
        "ridge_clearance_ft": ridge_clearance,
        "ridge_clearance_source": "Task-specified 6-inch clearance; not defined in the reference dataset",
        "horizontal_projection_ft": horizontal_projection,
        "sloped_roof_length_ft": roof_slope_length,
        "roof_boundary_x_ft": [overhang_left, overhang_right],
        "overhang_ft": overhang,
        "predicted_spacing_ft": predicted_purlin_spacing,
        "predicted_purlin_count_diagnostic_only": predicted_purlin_count,
        "target_clear_gap_ft": purlin_spacing_target if spacing_is_center_to_center else 0.75,
        "actual_clear_gap_ft": actual_purlin_max_spacing,
        "target_spacing_ft": purlin_spacing_target if spacing_is_center_to_center else 0.75,
        "actual_spacing_ft": actual_purlin_max_spacing,
        "max_actual_spacing_ft": actual_purlin_max_spacing,
        "profile_face_width_in": purlin_face_width_in,
        "profile_face_width_source": purlin_face_width_source,
        "center_to_center_pitch_ft": center_pitch_ft,
        "top_remainder_in": round(top_remainder_in, 4),
        "rows_per_slope": left_row_count if roof == "gable" else left_row_count,
        "longitudinal_run_count": len(purlin_x_positions),
        "longitudinal_segment_count": len(purlin_x_positions) * purlin_longitudinal_segments,
        "physical_member_count": len(purlin_x_positions) * purlin_longitudinal_segments,
        "purlin_positions_x_ft": purlin_x_positions,
    }
    support_connections = []
    column_members = [member for member in members if member["member_type"] == "Column"]
    beam_members = [member for member in members if member["member_type"] == "Beam"]
    truss_members_all = [
        member for member in members
        if member["member_type"] in {"Top chord", "Bottom chord"}
    ]
    bearing_chord_type = "Top chord" if family_kind in {"plain", "half_a"} else "Bottom chord"
    bearing_chord_members = [
        member for member in truss_members_all
        if member["member_type"] == bearing_chord_type
    ]

    def _same_point(first, second):
        return all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-7) for a, b in zip(first, second))

    for x in column_lanes:
        beam_z = beam_axis_z(x)
        for y in truss_positions:
            col_top = (x, y, column_top_z(x))
            column_member = next(
                (member for member in column_members if _same_point(member["end"], col_top)),
                None,
            )
            beam_member = next(
                (
                    member for member in beam_members
                    if math.isclose(member["start"][0], x, abs_tol=1e-7)
                    and member["start"][1] - 1e-7 <= y <= member["end"][1] + 1e-7
                ),
                None,
            )
            chord_node = (x, y, support_z(x))
            chord_member = next(
                (
                    member for member in bearing_chord_members
                    if member["truss_index"] == truss_positions.index(y)
                    and any(_same_point(endpoint, chord_node) for endpoint in (member["start"], member["end"]))
                ),
                None,
            )
            beam_bottom = beam_z - beam_depth / 2.0
            beam_top = beam_z + beam_depth / 2.0
            support_connections.append({
                "x_ft": x,
                "y_ft": y,
                "column_member_id": column_member["member_id"] if column_member else None,
                "beam_member_id": beam_member["member_id"] if beam_member else None,
                "truss_member_id": chord_member["member_id"] if chord_member else None,
                "bearing_chord_type": bearing_chord_type,
                "column_top_surface_z_ft": column_top_z(x),
                "beam_bottom_surface_z_ft": beam_bottom,
                "beam_top_surface_z_ft": beam_top,
                "truss_chord_bottom_surface_z_ft": support_z(x) - chord_depth / 2.0,
                "column_bearing_here": column_member is not None,
                "column_to_beam_aligned": (
                    math.isclose(column_top_z(x), beam_bottom, rel_tol=0.0, abs_tol=1e-7)
                    if column_member and beam_member else None
                ),
                "truss_to_beam_aligned": bool(beam_member and chord_member) and math.isclose(
                    beam_top, support_z(x) - chord_depth / 2.0,
                    rel_tol=0.0, abs_tol=1e-7,
                ),
                "chord_support_node_present": chord_member is not None,
            })
    column_beam_connections = []
    column_support_connections = []
    for x, y, z in column_locations:
        column_member = next(
            (member for member in column_members if _same_point(member["end"], (x, y, z))),
            None,
        )
        beam_member = next(
            (
                member for member in beam_members
                if math.isclose(member["start"][0], x, abs_tol=1e-7)
                and member["start"][1] - 1e-7 <= y <= member["end"][1] + 1e-7
            ),
            None,
        )
        beam_bottom = beam_axis_z(x) - beam_depth / 2.0
        aligned = bool(column_member and beam_member) and math.isclose(
            z, beam_bottom, rel_tol=0.0, abs_tol=1e-7
        )
        column_beam_connections.append(aligned)
        column_support_connections.append({
            "x_ft": x,
            "y_ft": y,
            "column_member_id": column_member["member_id"] if column_member else None,
            "beam_member_id": beam_member["member_id"] if beam_member else None,
            "column_top_surface_z_ft": z,
            "beam_bottom_surface_z_ft": beam_bottom,
            "aligned": aligned,
        })
    truss_beam_connections = [
        connection["truss_to_beam_aligned"] for connection in support_connections
    ]
    chord_support_connections = [
        connection["chord_support_node_present"] for connection in support_connections
    ]
    beam_members = [member for member in members if member["member_type"] == "Beam"]
    actual_beam_member_count = len(beam_members)
    actual_beam_run_count = len({
        member["beam_run_index"]
        for member in beam_members
        if "beam_run_index" in member
    })

    actual_web_count = sum(
        m["member_type"].startswith("Web") and m["truss_index"] == 0 for m in members
    )
    warnings = [warnings_prefix] if warnings_prefix else []
    if requested_web_section and web_section and requested_web_section != web_section:
        warnings.append(
            f"Web pipe section '{requested_web_section}' exceeded the chord section "
            f"'{chord_section}' envelope; resolved to '{web_section}'. "
            "This dimensional guardrail is not a structural capacity check."
        )
    if any(
        len(set(_section_dimensions_in(section))) > 1
        for section in (chord_section, beam_section)
    ):
        warnings.append(
            "The pipe catalogue does not store a frame-member orientation field; "
            "the first listed catalog dimension is used as vertical depth for chords "
            "and beams. Confirm the fabricated section orientation."
        )
    predicted_web_spacing = _number(values, "web_spacing_ft") if family_kind != "plain" else None
    actual_web_spacing = span / web_panels if family_kind != "plain" else None
    if family_kind != "plain" and predicted_web_spacing and predicted_web_spacing > 0 and (
        predicted_web_spacing > reference_rule["web_spacing_limit_ft"] + 1e-7
    ):
        warnings.append(
            f"Predicted web spacing ({predicted_web_spacing:.2f} ft) exceeds the matched "
            f"reference limit ({reference_rule['web_spacing_limit_ft']:.2f} ft); the reference limit was used."
        )
    elif family_kind != "plain" and predicted_web_spacing and predicted_web_spacing > 0 and not math.isclose(
        predicted_web_spacing, actual_web_spacing, rel_tol=0.1, abs_tol=0.25
    ):
        warnings.append(
            f"Predicted web spacing ({predicted_web_spacing:.2f} ft) differs from the "
            f"reference-constrained panel pitch ({actual_web_spacing:.2f} ft)."
        )
    predicted_web_count = _count(values, "web_total_member_count", actual_web_count) if family_kind != "plain" else 0
    if family_kind == "plain" and predicted_web_count:
        warnings.append("Predicted web count conflicts with the Plain Pipe no-web family guardrail; no web members were generated.")
    elif family_kind != "plain" and predicted_web_count != actual_web_count:
        warnings.append(f"Predicted web count ({predicted_web_count}) differs from generated family geometry ({actual_web_count}); members were not discarded.")
    actual_column_count = len(column_locations)
    if predicted_column_count is not None and int(round(predicted_column_count)) != final_column_count:
        if predicted_column_spacing is not None and predicted_column_spacing > 0:
            adjustment_reason = (
                "the end-inclusive column layout was derived from predicted longitudinal "
                "spacing; the predicted count is diagnostic only"
            )
        else:
            adjustment_reason = (
                "the count fallback was adjusted to provide equal end-inclusive stations "
                "on every selected support line"
            )
        warnings.append(
            f"Predicted total column count ({predicted_column_count}) differs from the "
            f"resolved total ({final_column_count}); {adjustment_reason}."
        )
    warnings.append(
        "Geometry and BOM validation only; engineering verification is required. This result is not a structural approval."
    )
    if reference_rule["tile_weight_kg"]:
        warnings.append(
            f"Tile reference load note: {reference_rule['tile_weight_kg']}. Tile variant and project-specific load details are not modeled."
        )
    # Validate against the spacing target that actually generated the layout.
    # Roofing-specific overrides (10 ft sheet / 6 ft tile) intentionally take
    # precedence over the model's raw spacing prediction, so comparing against
    # predicted_column_spacing here would incorrectly reject valid layouts.
    if column_spacing_target is not None and column_spacing_target > 0:
        if resolved_column_spacing > column_spacing_target + 0.01:
            raise ValueError(
                f"Resolved column spacing ({resolved_column_spacing:.3f} ft) exceeds "
                f"the active maximum spacing rule ({column_spacing_target:.3f} ft) "
                "beyond the 0.01 ft rounding tolerance; refusing an inconsistent layout."
            )
        if predicted_column_spacing is not None and predicted_column_spacing > 0 and not math.isclose(
            predicted_column_spacing, resolved_column_spacing, rel_tol=0.1, abs_tol=0.5
        ):
            warnings.append(
                f"Model-predicted column spacing ({predicted_column_spacing:.2f} ft) differs "
                f"from the active roofing/layout rule ({column_spacing_target:.2f} ft); "
                f"the generated layout uses {resolved_column_spacing:.2f} ft over "
                f"{len(column_y_positions) - 1} equal intervals. The model value is diagnostic only."
            )
    else:
        warnings.append(
            "No positive longitudinal column-spacing target or model prediction was available; "
            "the validated predicted total count was used only as a fallback, with "
            "end-inclusive equal stations."
        )
    column_spacing_across = _number(values, "column_spacing_across_span_ft")
    actual_column_spacing_across = max((b - a for a, b in zip(column_lanes, column_lanes[1:])), default=span)
    if column_spacing_across and not math.isclose(column_spacing_across, actual_column_spacing_across, rel_tol=0.1, abs_tol=0.5):
        warnings.append(f"Predicted column spacing across span ({column_spacing_across:.2f} ft) differs from generated support spacing ({actual_column_spacing_across:.2f} ft).")
    if predicted_beam_count != len(beam_lanes):
        warnings.append(f"Predicted beam count ({predicted_beam_count}) differs from resolved column support lines ({len(beam_lanes)}); beam layout follows column connectivity.")
    actual_truss_spacing_source = truss_spacing_source
    if predicted_spacing is not None and not math.isclose(
        predicted_spacing, effective_spacing, rel_tol=0.1, abs_tol=0.5
    ):
        if truss_spacing_source == "Matched reference maximum":
            warnings.append(
                f"Predicted truss spacing ({predicted_spacing:.2f} ft) is outside the matched "
                f"reference range ({truss_spacing_min:.2f}-{resolved_truss_spacing:.2f} ft); "
                f"the reference maximum was used."
            )
        else:
            warnings.append(
                f"Predicted truss spacing ({predicted_spacing:.2f} ft) was adjusted to "
                f"{effective_spacing:.2f} ft for equal end-inclusive frame positions."
            )
    if predicted_truss_count != truss_count:
        warnings.append(f"Predicted truss count ({predicted_truss_count}) differs from inclusive spacing layout ({truss_count}); end frames are retained.")
    if zero_length_count:
        warnings.append(f"Excluded {zero_length_count} zero-length generated members.")
    if duplicate_member_count:
        warnings.append(f"Removed {duplicate_member_count} duplicate generated members.")

    node_coordinates = sorted({
        tuple(round(value, 7) for value in endpoint)
        for member in members
        for endpoint in (member["start"], member["end"])
    })
    node_ids = {point: index + 1 for index, point in enumerate(node_coordinates)}
    nodes = [{"node_id": node_ids[point], "x_ft": point[0], "y_ft": point[1], "z_ft": point[2]}
             for point in node_coordinates]
    for member in members:
        member["start_node_id"] = node_ids[tuple(round(value, 7) for value in member["start"])]
        member["end_node_id"] = node_ids[tuple(round(value, 7) for value in member["end"])]

    member_ids = {member["member_id"] for member in members}
    parents = {member_id: member_id for member_id in member_ids}

    def find_member(member_id):
        while parents[member_id] != member_id:
            parents[member_id] = parents[parents[member_id]]
            member_id = parents[member_id]
        return member_id

    def connect_members(first_id, second_id):
        if first_id not in parents or second_id not in parents:
            return False
        first_root, second_root = find_member(first_id), find_member(second_id)
        if first_root != second_root:
            parents[second_root] = first_root
        return True

    endpoint_members = {}
    for member in members:
        for endpoint_name in ("start_node_id", "end_node_id"):
            endpoint_members.setdefault(member[endpoint_name], []).append(member["member_id"])
    for touching_members in endpoint_members.values():
        for member_id in touching_members[1:]:
            connect_members(touching_members[0], member_id)

    connectivity_edges = []
    for connection in column_support_connections:
        if connection["aligned"] and connect_members(
            connection["column_member_id"], connection["beam_member_id"]
        ):
            connectivity_edges.append({
                "member_ids": [
                    connection["column_member_id"], connection["beam_member_id"]
                ],
                "connection_type": "column-to-beam bearing",
            })
    for connection in support_connections:
        if connection["truss_to_beam_aligned"] and connect_members(
            connection["beam_member_id"], connection["truss_member_id"]
        ):
            connectivity_edges.append({
                "member_ids": [
                    connection["beam_member_id"], connection["truss_member_id"]
                ],
                "connection_type": "truss-to-beam bearing",
            })
    for connection in purlin_bearing_contacts:
        if connection["aligned"] and connect_members(
            connection["purlin_member_id"], connection["truss_member_id"]
        ):
            connectivity_edges.append({
                "member_ids": [
                    connection["purlin_member_id"], connection["truss_member_id"]
                ],
                "connection_type": "purlin-to-truss bearing",
            })
    global_component_count = len({find_member(member_id) for member_id in member_ids})

    truss_component_counts = {}
    for truss_index in range(truss_count):
        truss_group = [
            member for member in members
            if member.get("truss_index") == truss_index
            and member["member_type"] != "Purlin"
        ]
        local_parent = {member["member_id"]: member["member_id"] for member in truss_group}

        def find_local(member_id):
            while local_parent[member_id] != member_id:
                local_parent[member_id] = local_parent[local_parent[member_id]]
                member_id = local_parent[member_id]
            return member_id

        local_nodes = {}
        for member in truss_group:
            for endpoint_name in ("start_node_id", "end_node_id"):
                local_nodes.setdefault(member[endpoint_name], []).append(member["member_id"])
        for touching_members in local_nodes.values():
            for member_id in touching_members[1:]:
                local_parent[find_local(member_id)] = find_local(touching_members[0])
        truss_component_counts[truss_index] = len({
            find_local(member_id) for member_id in local_parent
        })

    truss_members = [member for member in members if member["truss_index"] == 0]
    truss_node_coordinates = sorted({
        tuple(round(value, 7) for value in endpoint)
        for member in truss_members
        for endpoint in (member["start"], member["end"])
    })
    truss_nodes = [{"node_id": node_ids[point], "x_ft": point[0], "y_ft": point[1], "z_ft": point[2]}
                   for point in truss_node_coordinates]
    top_coordinates = {
        (round(endpoint[0], 7), round(endpoint[2], 7))
        for member in truss_members if member["member_type"] == "Top chord"
        for endpoint in (member["start"], member["end"])
    }
    roof_max_height = max((point[1] for point in top_coordinates), default=None)
    ridge_nodes = [point for point in top_coordinates if abs(point[1] - roof_max_height) <= 1e-7]
    ridge_node = ridge_nodes[0] if roof == "gable" and len(ridge_nodes) == 1 else None
    if roof == "gable" and ridge_node is None:
        warnings.append(f"Gable geometry must have one ridge node; generated {len(ridge_nodes)} maximum-height nodes.")
    top_chord_ridge_connections = sum(
        1 for member in truss_members if member["member_type"] == "Top chord"
        and any(abs(point[0] - ridge_node[0]) <= 1e-7 and abs(point[2] - ridge_node[1]) <= 1e-7
                for point in (member["start"], member["end"]))
    ) if ridge_node else 0
    ridge_node_id = node_ids.get((ridge_node[0], round(truss_positions[0], 7), ridge_node[1])) if ridge_node else None
    web_members = [member for member in truss_members if member["member_type"].startswith("Web")]
    web_pitch = span / web_panels if family_kind != "plain" else None
    actual_web_count = len(web_members)
    resolved_parameters = {
        "site_width_ft": span,
        "overall_building_width_ft": span,
        "unsupported_span_ft": unsupported_span,
        "supported_bay_spans_ft": supported_bay_spans,
        "support_configuration": (f"{len(column_lanes)} support lines ({len(column_lanes)-2} intermediate)" if len(column_lanes) > 2 else "two outer support lines"),
        "support_line_count": len(column_lanes),
        "support_line_locations_ft": column_lanes,
        "intermediate_support_positions_ft": column_lanes[1:-1],
        "generated_support_layout": bool(inputs.get("generated_support_layout", False)),
        "truss_continuity_at_middle_support": ("continuous chords with generated intermediate bearing joints" if len(column_lanes) > 2 else "not applicable"),
        "roofing_type": inputs.get("roofing_type", ""),
        "truss_family": family,
        "rise_height_ft": rise,
        "truss_total_count": truss_count,
        "truss_spacing_ft": effective_spacing,
        "web_intermediate_member_count": max(0, web_panels - 1) if family_kind != "plain" else 0,
        "web_total_member_count": actual_web_count,
        "web_spacing_ft": web_pitch,
        "overhang_ft": overhang,
        "overhang_default_ft": reference_rule["overhang_default_ft"],
        "overhang_max_ft": reference_rule["overhang_max_ft"],
        "reference_design_id": reference_rule["reference_design_id"],
        "reference_span_limit_ft": reference_rule["span_max_ft"],
        "reference_pipe_aliases": reference_rule["pipe_aliases"],
        "reference_web_layout": reference_rule["web_layout"],
        "web_spacing_source": web_spacing_source,
        "box_depth_ft": box_depth,
        "box_depth_source": box_depth_source,
        "frame_section_depth_convention": (
            "First catalog dimension used as vertical depth; verify actual orientation."
        ),
        "reference_support_count_rule": reference_rule["support_count"],
        "reference_middle_support_rule": reference_rule["middle_support_required"],
        "tile_weight_reference": reference_rule["tile_weight_kg"],
        "frame_chord_pipe_size": chord_section,
        "web_pipe_size": web_section or "Not applicable",
        "column_pipe_size": column_section,
        "middle_support_pipe_size": middle_support_section if has_center_support else "Not applicable",
        "beam_pipe_size": beam_section,
        "purlin_pipe_size": purlin_section,
        "reference_notes": reference_rule["reference_notes"],
        "column_total_count": actual_column_count,
        "predicted_column_total_count": predicted_column_count,
        "final_column_total_count": final_column_count,
        "support_line_count": len(column_lanes),
        "columns_per_support_line": len(column_y_positions),
        "column_positions_along_length_ft": column_y_positions,
        "column_spacing_along_length_ft": resolved_column_spacing,
        "predicted_column_spacing_along_length_ft": predicted_column_spacing,
        "column_spacing_target_ft": column_spacing_target,
        "column_spacing_source": column_spacing_source,
        "minimum_column_stations_for_spacing": (
            int(math.ceil(length / column_spacing_target - 1e-12)) + 1
            if column_spacing_target is not None and column_spacing_target > 0
            else None
        ),
        "column_spacing_across_span_ft": actual_column_spacing_across,
        "column_support_top_elevations_ft": [
            {"x_ft": x, "z_ft": column_top_z(x), "height_ft": column_top_z(x)}
            for x in column_lanes
        ],
        "beam_total_count": actual_beam_run_count,
        "beam_run_count": actual_beam_run_count,
        "beam_physical_segment_count": actual_beam_member_count,
        "beam_longitudinal_line_count": actual_beam_run_count,
        "purlin_spacing_ft": actual_purlin_max_spacing,
        "purlin_spacing_target_ft": 3.0 if spacing_is_center_to_center else 0.75,
        "purlin_center_to_center_pitch_ft": center_pitch_ft,
        "purlin_face_width_in": purlin_face_width_in,
        "purlin_top_remainder_in": top_remainder_in,
        "purlin_spacing_rule_ft": purlin_spacing_rule,
        "purlin_orientation": resolved_purlin_orientation or "Not specified",
        "roof_slope_angle_deg": math.degrees(math.atan2(rise, span if roof == "single_slope" else span / 2.0)),
        "purlin_total_lines": len(purlin_x_positions),
        "predicted_purlin_count_diagnostic_only": predicted_purlin_count,
        "purlin_rows_per_slope_calculated": purlin_rows_per_slope,
        "purlin_longitudinal_runs": len(purlin_x_positions),
        "purlin_physical_member_count": purlin_layout["physical_member_count"],
        "member_total_count": len(members),
    }
    parameter_sources = {
        "truss_family": "Geometry family",
        "rise_height_ft": "Model prediction" if str(mode).upper() == "AUTO" else "User specified",
        "truss_total_count": "Resolved spacing layout",
        "truss_spacing_ft": "Resolved end-frame coordinates",
        "web_intermediate_member_count": "Generated chord panel coordinates",
        "web_total_member_count": "Generated member geometry",
        "web_spacing_ft": "Generated panel coordinates",
        "column_total_count": "Generated physical columns across selected support lines",
        "predicted_column_total_count": "Original ML/manual prediction before feasibility adjustment",
        "final_column_total_count": "Adjusted physical total across all selected support lines",
        "columns_per_support_line": "Equal longitudinal stations generated on each selected support line",
        "column_positions_along_length_ft": "Roofing-specific/model spacing-derived equal intervals with both end stations included",
        "column_spacing_along_length_ft": "Calculated equal spacing from roofing-specific/model maximum interval and end stations",
        "column_spacing_target_ft": column_spacing_source,
        "predicted_column_spacing_along_length_ft": "Original model spacing prediction; diagnostic only",
        "column_spacing_across_span_ft": "Selected support lanes",
        "beam_total_count": "Generated logical beam runs",
        "beam_run_count": "Generated logical beam runs",
        "beam_physical_segment_count": "Generated physical beam segments",
        "beam_longitudinal_line_count": "Connected column support lines",
        "purlin_spacing_ft": "Calculated clear inner-to-inner gap",
        "purlin_spacing_target_ft": "Roofing-specific required clear gap (tile: 0.75 ft; metal sheet: minimum 3.0 ft)",
        "overhang_ft": "Selected from reference dataset or explicit design input",
        "reference_web_layout": "Matched span-specific reference dataset",
        "purlin_center_to_center_pitch_ft": "Clear gap plus upward-facing profile width",
        "purlin_face_width_in": purlin_face_width_source,
        "purlin_top_remainder_in": "Unfilled roof-slope remainder after final complete purlin row",
        "purlin_spacing_rule_ft": purlin_spacing_source,
        "purlin_orientation": "Model/manual profile-orientation label",
        "purlin_total_lines": "Generated roof-surface coordinates",
        "predicted_purlin_count_diagnostic_only": "ML prediction (diagnostic only; ignored for final count)",
        "purlin_rows_per_slope_calculated": "Calculated from slope length, purlin face width, and clear gap",
        "purlin_longitudinal_runs": "Generated roof-surface coordinates",
        "purlin_physical_member_count": "Segmented between generated trusses",
        "member_total_count": "Generated member geometry",
    }
    member_bounds_violations = [member["member_id"] for member in members if any(
        point[0] < overhang_left - 1e-7 or point[0] > overhang_right + 1e-7 or point[1] < -1e-7 or point[1] > length + 1e-7
        for point in (member["start"], member["end"])
    )]
    if member_bounds_violations:
        warnings.append(f"{len(member_bounds_violations)} members extend outside the generated roof/building bounds.")
    if roof == "gable":
        ridge_clearance_projection = ridge_clearance * (
            span / 2.0 + roof_edge_overhang
        ) / roof_slope_length
        required_purlin_boundaries = [
            overhang_left,
            span / 2.0 - ridge_clearance_projection,
            span / 2.0 + ridge_clearance_projection,
            overhang_right,
        ]
    else:
        required_purlin_boundaries = [overhang_left, overhang_right]
    purlin_boundary_rows_present = all(
        any(math.isclose(x, boundary, abs_tol=1e-7) for x in purlin_x_positions)
        for boundary in required_purlin_boundaries
    )
    # Equal distribution is checked in slope distance, not horizontal projection.
    purlin_intervals_equal = True
    for slope_positions_list, usable_slope_ft, horizontal_run_ft in slope_position_groups:
        if len(slope_positions_list) < 2 or horizontal_run_ft <= 0.0:
            purlin_intervals_equal = False
            continue
        slope_scale = usable_slope_ft / horizontal_run_ft
        intervals = [abs(b - a) * slope_scale for a, b in zip(slope_positions_list, slope_positions_list[1:])]
        if not intervals:
            purlin_intervals_equal = False
        if (not spacing_is_center_to_center
                and any(value > center_pitch_ft + 1e-6 for value in intervals)):
            purlin_intervals_equal = False
        if intervals and not all(
            math.isclose(value, intervals[0], rel_tol=0.0, abs_tol=1e-5)
            for value in intervals[1:]
        ):
            purlin_intervals_equal = False
    purlin_end_coverage = bool(
        purlin_x_positions
        and purlin_rows_per_slope >= 1
        and purlin_intervals_equal
        and purlin_top_remainder_in >= -1e-7
        and purlin_top_remainder_in < center_pitch_in + 1e-7
    )
    if spacing_is_center_to_center:
        # The metal-sheet target range guides row count but does not invalidate
        # an otherwise equalized layout solely for falling outside 3.0-3.5 ft.
        purlin_spacing_limit_met = bool(actual_clear_gaps_ft) and purlin_intervals_equal
    else:
        purlin_spacing_limit_met = math.isclose(actual_purlin_max_spacing, 0.75, rel_tol=0.0, abs_tol=1e-6)
    ridge_clearance_within_spacing_rule = (
        roof != "gable"
        or ridge_clearance <= purlin_actual_center_pitch_ft + 1e-7
    )
    if not ridge_clearance_within_spacing_rule:
        warnings.append(
            "The task-specified 6-inch ridge clearance exceeds the resolved purlin pitch; "
            "review the conflicting reference rule."
        )
    if not purlin_end_coverage:
        warnings.append("Purlin layout could not be resolved within the available roof-slope length.")
    if predicted_purlin_spacing is not None and not math.isclose(
        predicted_purlin_spacing, actual_purlin_max_spacing, rel_tol=0.02, abs_tol=0.02
    ):
        warnings.append(
            f"Model purlin spacing ({predicted_purlin_spacing:.2f} ft) differs from calculated clear gap "
            f"({actual_purlin_max_spacing:.2f} ft); final row count and positions use the calculated layout."
        )
    if spacing_is_center_to_center and actual_purlin_max_spacing > 0 and not (3.0 - 1e-6 <= actual_purlin_max_spacing <= 3.5 + 1e-6):
        warnings.append(
            f"Metal-sheet actual clear gap is {actual_purlin_max_spacing:.3f} ft; "
            "the 3.0-3.5 ft range is a reference target, not a hard validation limit. "
            "Rows remain equally distributed between the eave and ridge-clearance boundaries."
        )
    if not purlin_spacing_limit_met:
        warnings.append(
            "Purlin spacing validation failed: tile clear gaps must be exactly 0.750 ft, "
            "and generated purlin rows must be evenly distributed."
        )
    column_top_points = {
        tuple(round(value, 7) for value in member["end"])
        for member in members if member["member_type"] == "Column"
    }
    expected_column_tops = {
        tuple(round(value, 7) for value in location)
        for location in column_locations
    }
    column_coordinates_match = (
        column_top_points == expected_column_tops
        and len(column_top_points) == len(column_locations)
        and len(column_locations) == final_column_count
    )
    truss_support_points = {
        tuple(round(value, 7) for value in (x, y, support_z(x)))
        for x in column_lanes for y in truss_positions
    }
    chord_node_coordinates = {
        tuple(round(value, 7) for value in endpoint)
        for member in bearing_chord_members
        for endpoint in (member["start"], member["end"])
    }
    support_nodes_match_chords = (
        truss_support_points.issubset(chord_node_coordinates)
        and bool(chord_support_connections)
        and all(chord_support_connections)
    )
    beam_connected_to_columns = (
        len(column_beam_connections) == len(column_locations)
        and all(column_beam_connections)
    )
    truss_bearings_match_beams = (
        len(truss_beam_connections) == len(column_lanes) * len(truss_positions)
        and all(truss_beam_connections)
    )
    return {
        "complete": bool(members),
        "mode": str(mode).upper(),
        "roof_configuration": roof,
        "truss_family": family,
        "family_geometry_pattern": {
            "plain": "single-rafter profile; no bottom chord or web members",
            "semi": "single-slope top chord, horizontal bottom chord, repeated vertical posts",
            "box": "parallel roof-profile chords, cross-members, alternating diagonal braces",
            "full_a": "full-span gable top chord with eave overhang, inset bottom chord, vertical posts and ridge king post",
            "half_a": "full-span gable top chord, shortened bottom chord, central king post, vertical posts and paired diagonals",
        }[family_kind],
        "span_ft": span,
        "overall_building_width_ft": span,
        "unsupported_span_ft": unsupported_span,
        "supported_bay_spans_ft": supported_bay_spans,
        "support_configuration": (f"{len(column_lanes)} support lines ({len(column_lanes)-2} intermediate)" if len(column_lanes) > 2 else "two outer support lines"),
        "support_line_count": len(column_lanes),
        "support_line_locations_ft": column_lanes,
        "intermediate_support_positions_ft": column_lanes[1:-1],
        "generated_support_layout": bool(inputs.get("generated_support_layout", False)),
        "truss_continuity_at_middle_support": ("continuous chords with generated intermediate bearing joints" if len(column_lanes) > 2 else "not applicable"),
        "length_ft": length,
        "column_height_ft": column_height,
        "beam_depth_ft": beam_depth,
        "truss_chord_depth_ft": chord_depth,
        "purlin_depth_ft": purlin_depth,
        "beam_axis_z_ft": {str(x): beam_axis_z(x) for x in column_lanes},
        "truss_roof_base_z_ft": bearing_chord_base_z + box_depth,
        "box_depth_ft": box_depth,
        "box_depth_source": box_depth_source,
        "chord_vertical_depth_ft": chord_depth,
        "beam_vertical_depth_ft": beam_depth,
        "purlin_vertical_depth_ft": purlin_depth,
        "frame_section_depth_convention": (
            "Pipe labels have no orientation metadata; first dimension is used vertically "
            "for frame members. Purlin depth follows its configured orientation label."
        ),
        "roof_surface_base_z_ft": (
            bearing_chord_base_z + box_depth + chord_depth / 2.0 + purlin_depth
        ),
        "rise_height_ft": rise,
        "overhang_ft": overhang,
        "bearing_inset_ft": overhang if family_kind == "half_a" else 0.0,
        "roof_profile_x_bounds_ft": [overhang_left, overhang_right],
        "roof_profile_boundary_elevations_ft": [
            {"x_ft": x, "z_ft": top_z(x)}
            for x in (overhang_left, overhang_right)
        ],
        "reference_geometry_rule": reference_rule,
        "engineering_verification_required": True,
        "geometry_bom_checks_only": True,
        "roof_slope_angle_deg": math.degrees(math.atan2(rise, span if roof == "single_slope" else span / 2.0)),
        "purlin_profile_orientation": values.get("purlin_orientation", "Not specified"),
        "purlin_spacing_ft": actual_purlin_max_spacing,
        "purlin_spacing_target_ft": 3.0 if spacing_is_center_to_center else 0.75,
        "purlin_spacing_rule_ft": purlin_spacing_rule,
        "purlin_max_actual_spacing_ft": actual_purlin_max_spacing,
        "purlin_center_to_center_pitch_ft": purlin_actual_center_pitch_ft,
        "purlin_face_width_in": purlin_face_width_in,
        "purlin_top_remainder_in": purlin_top_remainder_in,
        "purlin_spacing_source": purlin_spacing_source,
        "purlin_x_positions_ft": purlin_x_positions,
        "purlin_layout": purlin_layout,
        "roofing_layer": {
            "roofing_type": inputs.get("roofing_type", ""),
            "supported_profile": True,
            "coverage_quantity_supported": False,
            "surface_elevation_base_z_ft": (
                bearing_chord_base_z + box_depth + chord_depth / 2.0 + purlin_depth
            ),
            "ridge_clearance_rule_ft": ridge_clearance if roof == "gable" else None,
            "warning": "The roof surface follows the generated eave boundaries. Sheet/tile coverage, laps, and installation quantities are unavailable because the project has no structured coverage rules.",
        },
        "web_panel_pitch_ft": web_pitch,
        "truss_positions_ft": truss_positions,
        "column_lanes_ft": column_lanes,
        "column_y_positions_ft": column_y_positions,
        "column_positions_along_length_ft": column_y_positions,
        "column_spacing_target_ft": column_spacing_target,
        "column_spacing_actual_ft": resolved_column_spacing,
        "column_spacing_source": column_spacing_source,
        "predicted_column_spacing_along_length_ft": predicted_column_spacing,
        "predicted_column_total_count": predicted_column_count,
        "final_column_total_count": final_column_count,
        "support_line_count": len(column_lanes),
        "columns_per_support_line": len(column_y_positions),
        "column_support_top_elevations_ft": [
            {"x_ft": x, "z_ft": column_top_z(x), "height_ft": column_top_z(x)}
            for x in column_lanes
        ],
        "beam_lanes_ft": beam_lanes,
        "resolved_parameters": resolved_parameters,
        "parameter_sources": parameter_sources,
        "members": members,
        "support_connections": support_connections,
        "column_support_connections": column_support_connections,
        "connectivity_edges": connectivity_edges,
        "purlin_bearing_connections": purlin_bearing_contacts,
        "truss_members": truss_members,
        "nodes": nodes,
        "truss_nodes": truss_nodes,
        "ridge_node": ridge_node,
        "ridge_node_id": ridge_node_id,
        "ridge_node_count": len(ridge_nodes) if roof == "gable" else 0,
        "ridge_top_chord_connections": top_chord_ridge_connections,
        "validation": {
            "positive_dimensions": True,
            "roof_profile": roof,
            "zero_length_members_excluded": zero_length_count,
            "duplicate_members_removed": duplicate_member_count,
            "unique_truss_nodes": len(truss_nodes),
            "truss_count": len(truss_positions),
            "member_count": len(members),
            "all_endpoints_have_node_ids": all("start_node_id" in member and "end_node_id" in member for member in members),
            "truss_components_connected": all(
                component_count == 1 for component_count in truss_component_counts.values()
            ),
            "truss_component_counts": truss_component_counts,
            "all_members_connected": bool(member_ids) and global_component_count == 1,
            "member_component_count": global_component_count,
            "member_bounds_violations": member_bounds_violations,
            "end_supports_present": all(
                any(abs(column[0] - lane) <= 1e-7 and abs(column[1] - end_y) <= 1e-7 for column in column_locations)
                for lane in column_lanes for end_y in (0.0, length)
            ),
            "beam_to_column_connectivity": beam_connected_to_columns,
            "truss_bearings_match_beams": truss_bearings_match_beams,
            "support_nodes_match_chords": support_nodes_match_chords,
            "columns_match_support_elevations": column_coordinates_match,
            "overhang_within_reference_limit": 0.0 <= overhang <= reference_rule["overhang_max_ft"],
            "all_support_surfaces_aligned": (
                beam_connected_to_columns
                and truss_bearings_match_beams
                and support_nodes_match_chords
                and all(connection["aligned"] for connection in column_support_connections)
                and bool(purlin_bearing_contacts)
                and all(connection["aligned"] for connection in purlin_bearing_contacts)
            ),
            "purlins_bear_on_trusses": bool(purlin_bearing_contacts)
            and all(connection["aligned"] for connection in purlin_bearing_contacts),
            "gable_purlins_clear_ridge": roof != "gable" or ridge_row_count == 0,
            "ridge_clearance_within_spacing_rule": ridge_clearance_within_spacing_rule,
            "beam_longitudinal_line_count": actual_beam_run_count,
            "beam_run_count": actual_beam_run_count,
            "beam_physical_member_count": actual_beam_member_count,
            "column_count_matches_coordinates": column_coordinates_match,
            "column_count_matches_support_distribution": (
                actual_column_count == len(column_lanes) * len(column_y_positions)
                and actual_column_count == final_column_count
            ),
            "purlin_max_spacing_ft": actual_purlin_max_spacing,
            "purlin_center_to_center_pitch_ft": purlin_actual_center_pitch_ft,
            "purlin_face_width_in": purlin_face_width_in,
            "purlin_top_remainder_in": purlin_top_remainder_in,
            "purlin_spacing_rule_ft": purlin_spacing_rule,
            "purlin_end_coverage": purlin_end_coverage,
            "purlin_boundary_rows_present": purlin_boundary_rows_present,
            "purlin_intervals_equal": purlin_intervals_equal,
            "purlin_spacing_limit_met": purlin_spacing_limit_met,
        },
        "warnings": warnings,
    }



def _section_dimensions_ft(section):
    """Read rectangular/square tube dimensions from labels such as '2 x 2'."""
    import re
    text = str(section or "")
    tokens = re.findall(r'(?<![A-Za-z])(\d+(?:\.\d+)?|\d+\s*/\s*\d+)', text)
    values = []
    for token in tokens[:2]:
        token = token.replace(" ", "")
        try:
            value = (float(token.split("/")[0]) / float(token.split("/")[1])) if "/" in token else float(token)
            if value > 0:
                values.append(value / 12.0)
        except (ValueError, ZeroDivisionError):
            continue
    if not values:
        return 0.10, 0.10
    if len(values) == 1:
        values.append(values[0])
    # Prevent a malformed catalogue label from making an unreadable model.
    return max(0.025, min(values[0], 1.0)), max(0.025, min(values[1], 1.0))


def _append_box_member_mesh(mesh, start, end, section, member=None):
    """Append a full-size rectangular member, with roof-aligned purlin sections."""
    import math
    try:
        p1 = [float(v) for v in start]
        p2 = [float(v) for v in end]
    except (TypeError, ValueError):
        return
    if len(p1) != 3 or len(p2) != 3:
        return
    axis = [p2[i] - p1[i] for i in range(3)]
    length = math.sqrt(sum(v * v for v in axis))
    if not math.isfinite(length) or length <= 1e-8:
        return
    axis = [v / length for v in axis]
    # Choose a reference direction that is not parallel to the member.
    ref = [0.0, 0.0, 1.0] if abs(axis[2]) < 0.88 else [0.0, 1.0, 0.0]
    u = [axis[1]*ref[2] - axis[2]*ref[1],
         axis[2]*ref[0] - axis[0]*ref[2],
         axis[0]*ref[1] - axis[1]*ref[0]]
    u_len = math.sqrt(sum(v*v for v in u))
    if u_len <= 1e-10:
        return
    u = [v/u_len for v in u]
    v = [axis[1]*u[2] - axis[2]*u[1],
         axis[2]*u[0] - axis[0]*u[2],
         axis[0]*u[1] - axis[1]*u[0]]
    dims = list(_section_dimensions_ft(section))
    member = member or {}
    kind = str(member.get("member_type", ""))
    depth = member.get("section_depth_ft")
    if depth is not None and kind in {"Beam", "Purlin"}:
        height = float(depth)
        width = max(0.025, sum(dims) - height)
    elif kind == "Beam":
        # Frame sections use the first catalogue dimension as vertical depth.
        width, height = dims[1], dims[0]
    else:
        width, height = dims[0], dims[1]

    # Rotate the purlin cross-section about its longitudinal axis so its broad
    # face follows the roof slope and its bottom face seats flush on the chord.
    angle = math.radians(float(member.get("cross_section_rotation_deg", 0.0) or 0.0))
    if kind == "Purlin" and abs(angle) > 1e-10:
        c, sn = math.cos(angle), math.sin(angle)
        u0, v0 = u[:], v[:]
        u = [c*u0[i] - sn*v0[i] for i in range(3)]
        v = [sn*u0[i] + c*v0[i] for i in range(3)]

    base_index = len(mesh["x"])
    for point in (p1, p2):
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            mesh["x"].append(point[0] + su * width * 0.5 * u[0] + sv * height * 0.5 * v[0])
            mesh["y"].append(point[1] + su * width * 0.5 * u[1] + sv * height * 0.5 * v[1])
            mesh["z"].append(point[2] + su * width * 0.5 * u[2] + sv * height * 0.5 * v[2])
    faces = [
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
    ]
    for i, j, k in faces:
        mesh["i"].append(base_index + i)
        mesh["j"].append(base_index + j)
        mesh["k"].append(base_index + k)


def _mesh_trace(grouped_members, color_for, name_prefix=""):
    """Build one Mesh3d trace per member-type/section group."""
    traces = []
    for key, members in grouped_members.items():
        mesh = {"x": [], "y": [], "z": [], "i": [], "j": [], "k": []}
        for member in members:
            _append_box_member_mesh(mesh, member.get("start"), member.get("end"), member.get("pipe_section"), member)
        if mesh["i"]:
            kind, section = key[-2], key[-1]
            traces.append(go.Mesh3d(
                x=mesh["x"], y=mesh["y"], z=mesh["z"], i=mesh["i"], j=mesh["j"], k=mesh["k"],
                color=color_for(kind), opacity=1.0, flatshading=True,
                name=f"{name_prefix}{kind} · {section}", hovertemplate=f"{kind}<br>Section: {section}<extra></extra>",
                lighting={"ambient": 0.55, "diffuse": 0.75, "specular": 0.18, "roughness": 0.72},
                showscale=False,
            ))
    return traces


def draw_truss_figure(geometry):
    """Render the generated individual truss as solid 3D rectangular tube members."""
    fig = go.Figure()
    palette = {
        "Top chord": "#17627a", "Bottom chord": "#c85b3d",
        "Web vertical": "#648477", "Web diagonal": "#9a7b50", "Web king post": "#9a7b50",
    }
    grouped = {}
    for member in geometry.get("truss_members", []):
        kind = str(member.get("member_type", "Member"))
        key = (kind, member.get("pipe_section", "N/A"), kind, str(member.get("pipe_section", "N/A")))
        grouped.setdefault(key, []).append(member)
    for trace in _mesh_trace(grouped, lambda kind: palette.get(kind, "#688078")):
        fig.add_trace(trace)

    nodes = geometry.get("truss_nodes", [])
    if nodes:
        # The individual truss is planar, but the pipe sections are true 3D solids.
        fig.add_trace(go.Scatter3d(
            x=[node.get("x_ft", 0.0) for node in nodes],
            y=[node.get("y_ft", 0.0) for node in nodes],
            z=[node.get("z_ft", 0.0) for node in nodes],
            mode="markers+text", text=[f"N{node.get('node_id', '')}" for node in nodes],
            textposition="top center", marker={"size": 3, "color": "#17303a"},
            name="Joints", hovertemplate="%{text}<extra></extra>",
        ))
    span = float(geometry.get("span_ft", 0) or 0)
    rise = float(geometry.get("rise_height_ft", 0) or 0)
    base = float(geometry.get("truss_roof_base_z_ft", 0) or 0)
    fig.update_layout(
        title=f"{geometry.get('truss_family', 'Generated')} · 3D individual truss",
        height=560, margin={"l": 0, "r": 0, "t": 48, "b": 0}, paper_bgcolor="white",
        scene={
            "xaxis_title": "Span (ft)", "yaxis_title": "Section depth (ft)", "zaxis_title": "Height (ft)",
            "aspectmode": "data", "bgcolor": "#f7f8f2",
            "camera": {"eye": {"x": 1.65, "y": 1.8, "z": 1.2}},
            "annotations": [
                {"x": span / 2, "y": 0, "z": base + rise,
                 "text": f"Rise {rise:.2f} ft", "showarrow": False, "font": {"size": 10}}
            ],
        },
        legend={"orientation": "h", "y": 1.02},
    )
    return fig


def draw_building_figure(geometry, visible_categories=None, camera="Isometric", wireframe=False):
    """Render the complete generated building with solid 3D members and roof surfaces."""
    fig = go.Figure()
    colors = {"Column": "#536b63", "Beam": "#bd654a", "Purlin": "#c4a367"}
    grouped = {}
    wire_groups = {}
    for member in geometry.get("members", []):
        kind = str(member.get("member_type", "Member"))
        category = (
            "webs" if kind.startswith("Web")
            else "trusses" if "chord" in kind.casefold()
            else f"{kind.casefold()}s" if kind in {"Column", "Beam", "Purlin"}
            else kind.casefold()
        )
        if visible_categories is not None and category not in visible_categories:
            continue
        key = (category, kind, str(member.get("pipe_section", "N/A")), kind, str(member.get("pipe_section", "N/A")))
        grouped.setdefault(key, []).append(member)
        if wireframe:
            trace_key = (category, kind, str(member.get("pipe_section", "N/A")))
            trace = wire_groups.setdefault(trace_key, {"x": [], "y": [], "z": []})
            for endpoint in (member.get("start"), member.get("end")):
                if not isinstance(endpoint, (list, tuple)) or len(endpoint) != 3:
                    continue
                trace["x"].append(endpoint[0]); trace["y"].append(endpoint[1]); trace["z"].append(endpoint[2])
            trace["x"].append(None); trace["y"].append(None); trace["z"].append(None)

    if wireframe:
        for (category, kind, section), trace in wire_groups.items():
            fig.add_trace(go.Scatter3d(
                x=trace["x"], y=trace["y"], z=trace["z"], mode="lines",
                line={"color": colors.get(kind, "#27657a" if "chord" in kind.casefold() else "#8c7654"), "width": 2},
                name=f"{kind} · {section}", legendgroup=category,
                showlegend=category not in [t.legendgroup for t in fig.data],
            ))
    else:
        for trace in _mesh_trace(
            grouped,
            lambda kind: colors.get(kind, "#27657a" if "chord" in kind.casefold() else "#8c7654"),
        ):
            fig.add_trace(trace)

    if visible_categories is None or "roofing" in visible_categories:
        span = float(geometry.get("span_ft", 0) or 0)
        length = float(geometry.get("length_ft", 0) or 0)
        base = float(geometry.get("roof_surface_base_z_ft", 0) or 0)
        rise = float(geometry.get("rise_height_ft", 0) or 0)
        roof_left, roof_right = geometry.get("roof_profile_x_bounds_ft", [0.0, span])
        roof_kind = geometry.get("roof_configuration", "single_slope")
        if span > 0 and length > 0:
            if roof_kind == "gable":
                half_span = span / 2.0
                faces = [
                    [(roof_left, 0.0, base + _profile(roof_left, span, rise, "gable")),
                     (half_span, 0.0, base + rise), (half_span, length, base + rise),
                     (roof_left, length, base + _profile(roof_left, span, rise, "gable"))],
                    [(half_span, 0.0, base + rise),
                     (roof_right, 0.0, base + _profile(roof_right, span, rise, "gable")),
                     (roof_right, length, base + _profile(roof_right, span, rise, "gable")),
                     (half_span, length, base + rise)],
                ]
            else:
                faces = [[(roof_left, 0.0, base + _profile(roof_left, span, rise, "single_slope")),
                          (roof_right, 0.0, base + _profile(roof_right, span, rise, "single_slope")),
                          (roof_right, length, base + _profile(roof_right, span, rise, "single_slope")),
                          (roof_left, length, base + _profile(roof_left, span, rise, "single_slope"))]]
            for face_index, corners in enumerate(faces, start=1):
                fig.add_trace(go.Mesh3d(
                    x=[point[0] for point in corners], y=[point[1] for point in corners],
                    z=[point[2] for point in corners], i=[0, 0], j=[1, 2], k=[2, 3],
                    color="#b8a16d", opacity=0.22, name="Roofing layer",
                    legendgroup="roofing", showlegend=face_index == 1,
                    hovertemplate=f"Roofing surface · {geometry.get('roofing_layer', {}).get('roofing_type', 'Roof')}<extra></extra>",
                ))
    eye = {
        "Isometric": {"x": 1.5, "y": 1.5, "z": 1.2},
        "Front elevation": {"x": 2.2, "y": 0.01, "z": 0.45},
        "End elevation": {"x": 0.01, "y": -2.2, "z": 0.45},
        "Plan": {"x": 0.01, "y": 0.01, "z": 2.5},
    }.get(camera, {"x": 1.5, "y": 1.5, "z": 1.2})
    fig.update_layout(
        title=f"{geometry.get('truss_family', 'Generated')} · 3D building model",
        height=600, margin={"l": 0, "r": 0, "t": 45, "b": 0}, paper_bgcolor="white",
        scene={"xaxis_title": "Width (ft)", "yaxis_title": "Length (ft)", "zaxis_title": "Height (ft)",
               "bgcolor": "#f7f8f2", "aspectmode": "data", "camera": {"eye": eye}},
        legend={"orientation": "h", "y": 1.0},
    )
    return fig
