"""Evaluate the requested truss family and support arrangement."""
from __future__ import annotations

from calculator import calculate_bom
from design import (
    compatible_truss_families,
    family_roof_compatibility_error,
    generate_design,
    reference_middle_support_available,
    reference_span_limit,
    reference_span_rule_for_bay,
)


def _enabled(value):
    return str(value).strip().casefold() in {"yes", "true", "1"}


def _candidate_label(candidate):
    support = "3 support lines" if candidate["middle_support_present"] else "2 outer support lines"
    return (
        f"{candidate['truss_family']} · {support} · "
        f"{candidate['unsupported_span_ft']:.2f} ft bays"
    )


def _manual_family_priority(options, preferred_family, mode):
    """Prefer the requested family; use a valid compatible fallback only if needed."""
    if str(mode).upper() == "AUTO" or not preferred_family:
        return options
    preferred_options = [
        option for option in options if option.get("truss_family") == preferred_family
    ]
    return preferred_options if preferred_options else options


def _families_to_evaluate(inputs, families, mode, preferred_family=None):
    """Resolve which families may be evaluated for this request.

    MANUAL mode strictly limits evaluation to the selected family. AUTO mode
    evaluates all roof-compatible families (or the supplied list if compatibility
    filtering returns none).
    """
    compatible = compatible_truss_families(inputs) or list(families)
    if str(mode).upper() != "AUTO" and preferred_family:
        # Keep the user-selected family first, but evaluate compatible alternatives
        # in the background so the UI can show the nearest validated design if all
        # layouts of the requested family fail geometry/BOM checks.
        return [preferred_family, *[family for family in compatible if family != preferred_family]]
    return compatible


def _rank_and_limit_options(options, preferred_family=None, prefer_unsupported=True, limit=3):
    """Return the best distinct candidates in strict support-strategy order.

    AUTO priority is: every feasible unsupported design first (one per family
    where possible), then one-middle-support designs, then multi-support designs.
    A supported design must never displace a feasible unsupported alternative.
    Support count is the primary priority: an eligible unsupported design is shown
    before any supported layout, then one-middle-support, then multi-support layouts
    in increasing support count. For Box Truss, a reference zig-zag web pattern is
    preferred over a vertical-only pattern when both cover the same bay span. Within
    otherwise equivalent choices, the closest adequate reference span is preferred;
    BOM weight is only a tie-breaker, not a safety/capacity metric.
    """
    def strategy(option):
        return option.get(
            "support_strategy",
            "single_middle" if option.get("middle_support_present") else "unsupported",
        )

    def rank(option):
        actual_bay = float(option.get("unsupported_span_ft", 0.0) or 0.0)
        reference_max = float(option.get("reference_span_limit_ft", 0.0) or 0.0)
        # A candidate should already have passed reference validation. This
        # margin is only a tie-breaker between validated options, not a capacity check.
        span_margin = max(0.0, reference_max - actual_bay) if reference_max > 0 else float("inf")
        family_name = str(option.get("truss_family", "")).casefold()
        web_layout = str(option.get("web_layout", "")).casefold()
        is_box = "box" in family_name
        zigzag_rank = 0 if (is_box and "zig" in web_layout) else (1 if is_box else 0)
        support_count = int(option.get("support_line_count", 0) or 0)
        return (
            ({"unsupported": 0, "single_middle": 1, "multi_support": 2}.get(strategy(option), 3)
             if prefer_unsupported else 0),
            support_count if strategy(option) == "multi_support" else 0,
            zigzag_rank,
            span_margin,
            option.get("truss_family") != preferred_family if preferred_family else False,
            option.get("bom", {}).get("material_weight_kg", float("inf")),
            option.get("truss_family", ""),
        )

    ordered = sorted(options, key=rank)
    selected = []
    selected_ids = set()
    selected_families = set()

    # Fill the requested number of results from unsupported options first. Only
    # if there are too few distinct validated unsupported families do we fall
    # back to one-middle support, and then reference-driven multi-support.
    strategies = ("unsupported", "single_middle", "multi_support") if prefer_unsupported else tuple(
        dict.fromkeys(strategy(option) for option in ordered)
    )
    for wanted in strategies:
        for option in ordered:
            if strategy(option) != wanted:
                continue
            identity = option.get("option_id", id(option))
            family = option.get("truss_family")
            if identity in selected_ids or family in selected_families:
                continue
            selected.append(option)
            selected_ids.add(identity)
            selected_families.add(family)
            if len(selected) >= limit:
                return selected

    # If only one/two families are valid, use a second support strategy for an
    # already represented family rather than inventing a design.
    for option in ordered:
        identity = option.get("option_id", id(option))
        if identity in selected_ids:
            continue
        selected.append(option)
        selected_ids.add(identity)
        if len(selected) >= limit:
            break
    return selected



def calculate_multi_support_layout(overall_span_ft, reference_max_span_ft, safety_factor=0.8):
    """Return equal support positions for a candidate span.

    This is a layout-generation heuristic only. A returned layout still needs
    reference compatibility and structural verification.
    """
    import math
    span = float(overall_span_ft)
    maximum = float(reference_max_span_ft)
    factor = float(safety_factor)
    if not math.isfinite(span) or span <= 0:
        raise ValueError("Overall span must be a finite positive number.")
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Reference maximum span must be a finite positive number.")
    if not math.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError("Safety factor must be greater than 0 and no more than 1.")
    target_bay = factor * maximum
    bay_count = max(3, int(math.ceil(span / target_bay)))
    positions = [span * i / bay_count for i in range(bay_count + 1)]
    actual_bay = span / bay_count
    if actual_bay > maximum + 1e-9:
        raise ValueError("Generated bay exceeds the reference maximum span.")
    return {"bay_count": bay_count, "support_line_count": bay_count + 1,
            "target_bay_spacing_ft": target_bay, "actual_bay_spacing_ft": actual_bay,
            "support_positions_ft": positions}


def _reference_driven_multi_support_layouts(
    inputs, family, overall_span_ft, family_max_span_ft, max_layouts=12
):
    """Generate several equal-bay, reference-matched support layouts.

    Do not stop at the first candidate: a layout can satisfy the reference-span
    lookup but still fail later geometry/BOM checks. The caller must try layouts
    with progressively more intermediate supports until a complete candidate
    validates. These are geometry candidates, not structural certification.
    """
    # Half-A reference rules define their reference span using the full roof span;
    # dividing it into bays would misapply those rules.
    if "half-a" in str(family).casefold() or not family_max_span_ft:
        return []

    span = float(overall_span_ft)
    layouts = []
    seen_designs = set()

    # Start with the minimum number of equal bays needed to keep each bay within
    # the reference maximum, then add supports only if validation needs alternatives.
    # This avoids over-supporting: a 200 ft span with a 100 ft reference maximum
    # starts with two 100 ft bays (three support lines), not three bays.
    import math
    try:
        initial_bay_count = max(2, int(math.ceil(span / float(family_max_span_ft))))
    except (TypeError, ValueError, ZeroDivisionError):
        return []
    for bay_count in range(initial_bay_count, 1001):
        bay_span = span / bay_count
        row = reference_span_rule_for_bay(inputs, family, bay_span)
        if not row or not row.get("middle_support_available"):
            continue
        # Use the matched row's actual span bounds. The helper only returns a row
        # that covers bay_span, but keep this guard explicit for clarity.
        if bay_span < float(row.get("span_min_ft", 0.0)) - 1e-9:
            continue
        if bay_span > float(row.get("span_max_ft", 0.0)) + 1e-9:
            continue

        key = (bay_count, row.get("design_id"))
        if key in seen_designs:
            continue
        seen_designs.add(key)
        positions = [span * i / bay_count for i in range(bay_count + 1)]
        layouts.append({
            "bay_count": bay_count,
            "support_line_count": bay_count + 1,
            "target_bay_spacing_ft": float(row["span_max_ft"]),
            "actual_bay_spacing_ft": bay_span,
            "support_positions_ft": positions,
            "reference_design_id": row["design_id"],
            "reference_max_span_ft": row["span_max_ft"],
        })
        if len(layouts) >= max_layouts:
            break

    return layouts


def evaluate_design_options(inputs, families, prediction_for, mode, preferred_family=None):
    """Generate the requested family and support layout with detailed rejection data.
    
    AUTO compares all compatible families. MANUAL evaluates only the selected
    family and automatically resolves support layout within that family. A
    reference maximum is a per-supported-bay matching constraint, not a hard
    overall-building-span cap: the system may add support lines to reduce bay
    spans, but every candidate must pass reference, geometry, and BOM validation.
    """
    options = []
    rejections = []
    overall_span = float(inputs["site_width_ft"])
    requested_middle_support = _enabled(inputs.get("center_support_present", "No"))
    is_auto = str(mode).upper() == "AUTO"

    # Evaluate the preferred family first in MANUAL mode, but still evaluate
    # alternatives so a valid fallback can be offered if the selected family fails.
    requested_families = _families_to_evaluate(
        inputs, families, mode, preferred_family=preferred_family
    )

    for family in requested_families:
        compatibility_error = family_roof_compatibility_error(
            family,
            "gable" if str(inputs.get("roof_sides", "")).strip() == "2" else "single_slope",
        )
        if compatibility_error:
            rejections.append({
                "truss_family": family,
                "support_configuration": "not evaluated",
                "failure_category": "Geometry rules",
                "reason": compatibility_error,
            })
            continue

        reference_limit = reference_span_limit(inputs, family)
        middle_rule_available = reference_middle_support_available(inputs, family)

        # Generate the fewest-support layouts first. Reference maximums apply to
        # each unsupported bay, not the entire building span. If a candidate fails
        # validation, progressively closer equal-bay layouts are tried.
        long_gable_plain_pipe = (
            "plain" in str(family).casefold()
            and str(inputs.get("roof_sides", "")).strip() == "2"
            and overall_span > 10.0
        )
        support_variants = []

        # Never offer an unsupported candidate when the complete bearing-to-bearing
        # span is outside this family's matched reference range. For ordinary
        # families, the reference span is checked again against each proposed bay
        # before we create a supported candidate. This avoids treating a family-wide
        # maximum as proof that every support layout is reference-compatible.
        unsupported_within_reference = (
            reference_limit is not None
            and overall_span <= float(reference_limit) + 1e-9
        )
        if not long_gable_plain_pipe and unsupported_within_reference:
            support_variants.append({
                "kind": "unsupported",
                "positions": [0.0, overall_span],
                "label": "2 outer support lines",
            })

        # A single middle bearing is a valid fallback only when the actual half-span
        # matches a reference row that explicitly permits the three-line support
        # configuration. Half-A reference rows use the overall roof span as their
        # reference basis, so adding a middle bearing must not falsely reduce their
        # reference span.
        if "half-a" not in str(family).casefold():
            half_bay_rule = reference_span_rule_for_bay(
                inputs, family, overall_span / 2.0
            )
            if half_bay_rule and half_bay_rule.get("middle_support_available"):
                support_variants.append({
                    "kind": "single_middle",
                    "positions": [0.0, overall_span / 2.0, overall_span],
                    "label": "3 support lines with one middle bearing",
                })
        # Keep several reference-matched layouts, ordered from fewer to more
        # support lines. If the first layout fails geometry/BOM validation, the
        # next layout is still attempted instead of abandoning the selected family.
        layouts = _reference_driven_multi_support_layouts(
            inputs, family, overall_span, reference_limit, max_layouts=12
        )
        for layout in layouts:
            positions = layout["support_positions_ft"]
            support_variants.append({
                "kind": "multi_support",
                "positions": positions,
                "label": (
                    f"{len(positions)} support lines with "
                    f"{len(positions)-2} intermediate supports"
                ),
            })

        if not support_variants:
            if reference_limit is not None and overall_span > float(reference_limit) + 1e-9:
                reason = (
                    f"Overall column-to-column span {overall_span:.2f} ft exceeds the "
                    f"{family} reference limit of {float(reference_limit):.2f} ft, and no "
                    "matching reference rule authorizes a middle/multi-support bay layout. "
                    "No geometry was generated rather than inventing an unsupported design."
                )
            elif long_gable_plain_pipe:
                reason = (
                    "Plain Pipe gable designs above 10 ft require a reference-supported "
                    "support layout; no valid supported layout was found."
                )
            else:
                reason = "No reference-compatible support configuration is available for this family and span."
            rejections.append({
                "truss_family": family,
                "support_configuration": "no feasible support layout",
                "overall_span_ft": overall_span,
                "reference_span_limit_ft": reference_limit,
                "middle_support_evaluated": False,
                "middle_support_rule_available": middle_rule_available,
                "failure_category": "Reference rule",
                "reason": reason,
            })
            continue

        for support_variant in support_variants:
            strategy = support_variant["kind"]
            positions = support_variant["positions"]
            middle_support = len(positions) > 2
            unsupported_span = max(b - a for a, b in zip(positions, positions[1:]))
            candidate_inputs = {
                **inputs,
                "center_support_present": "Yes" if middle_support else "No",
                "support_line_positions_ft": positions,
                "generated_support_layout": strategy == "multi_support",
                "support_strategy": strategy,
                "unsupported_span_ft": unsupported_span,
            }
            support_label = support_variant["label"]
            try:
                prediction = prediction_for(candidate_inputs, family)
                if not prediction.get("connected"):
                    raise RuntimeError(prediction.get("message", "Model inference failed."))
                design_values = prediction["design"]
                predicted_family = str(design_values.get("truss_family", "")).strip()
                if predicted_family != family:
                    raise ValueError(
                        f"Selected family is '{family}', but inference returned "
                        f"'{predicted_family or 'no family'}'; refusing to substitute geometry."
                    )
                geometry = generate_design(candidate_inputs, design_values, mode)
                bom = calculate_bom(geometry)

                validation = geometry["validation"]
                checks = {
                    "connected member endpoints": validation["all_endpoints_have_node_ids"],
                    "each truss is connected": validation["truss_components_connected"],
                    "complete building connectivity": validation["all_members_connected"],
                    "no duplicate or zero-length members": (
                        validation["zero_length_members_excluded"] == 0
                        and validation["duplicate_members_removed"] == 0
                    ),
                    "end supports": validation["end_supports_present"],
                    "beam/column connectivity": validation["beam_to_column_connectivity"],
                    "truss/beam bearings": validation["truss_bearings_match_beams"],
                    "truss/bottom-chord support joints": validation["support_nodes_match_chords"],
                    "support faces aligned": validation["all_support_surfaces_aligned"],
                    "purlins bear on trusses": validation["purlins_bear_on_trusses"],
                    "column elevations": validation["columns_match_support_elevations"],
                    "overhang limit": validation["overhang_within_reference_limit"],
                    "column coordinates": validation["column_count_matches_coordinates"],
                    "column/beam support grid": validation["beam_to_column_connectivity"],
                    "column count matches support distribution": validation["column_count_matches_support_distribution"],
                    "purlin end coverage": validation["purlin_end_coverage"],
                    "purlin spacing": validation["purlin_spacing_limit_met"],
                    "gable ridge clearance": validation["ridge_clearance_within_spacing_rule"],
                    "member bounds": not validation["member_bounds_violations"],
                    "BOM": bom.get("connected", False)
                    and bom.get("member_count") == len(geometry["members"]),
                    "gable ridge": geometry["roof_configuration"] != "gable"
                    or geometry["ridge_top_chord_connections"] == 2,
                }
                failures = [name for name, passed in checks.items() if not passed]
                if failures:
                    detail_parts = []
                    if not bom.get("connected", False):
                        detail_parts.append(
                            f"BOM rejected the generated geometry: {bom.get('message', 'unspecified BOM failure')}"
                        )
                    elif bom.get("member_count") != len(geometry.get("members", [])):
                        detail_parts.append(
                            "BOM member count does not match geometry "
                            f"({bom.get('member_count')} vs {len(geometry.get('members', []))})."
                        )
                    detail = (" Details: " + " ".join(detail_parts)) if detail_parts else ""
                    raise ValueError(
                        f"Geometry/BOM validation failed: {', '.join(failures)}.{detail}"
                    )

                option = {
                    "option_id": f"{family}|{strategy}|{len(positions)}lines",
                    "support_strategy": strategy,
                    "truss_family": family,
                    "roofing_type": candidate_inputs.get("roofing_type", ""),
                    "overall_span_ft": overall_span,
                    "unsupported_span_ft": geometry["unsupported_span_ft"],
                    "bay_spans_ft": geometry["supported_bay_spans_ft"],
                    "middle_support_present": middle_support,
                    "support_strategy": strategy,
                    "support_line_positions_ft": positions,
                    "support_target_spacing_ft": (0.8 * geometry["reference_geometry_rule"]["span_max_ft"] if strategy == "multi_support" else None),
                    "support_configuration": geometry["support_configuration"],
                    "support_line_count": geometry["support_line_count"],
                    "support_line_locations_ft": geometry["support_line_locations_ft"],
                    "column_station_count": len(geometry["column_y_positions_ft"]),
                    "columns_per_support_line": geometry["columns_per_support_line"],
                    "column_stations_along_length_ft": geometry["column_y_positions_ft"],
                    "column_positions_along_length_ft": geometry["column_positions_along_length_ft"],
                    "support_line_count": geometry["support_line_count"],
                    "predicted_column_total_count": geometry["predicted_column_total_count"],
                    "final_column_total_count": geometry["final_column_total_count"],
                    "column_spacing_ft": geometry["column_spacing_actual_ft"],
                    "predicted_column_spacing_ft": geometry["predicted_column_spacing_along_length_ft"],
                    "column_spacing_source": geometry["column_spacing_source"],
                    "support_layout_reason": (
                        f"{geometry['support_line_count']} support lines at "
                        f"{', '.join(f'{x:.2f}' for x in geometry['support_line_locations_ft'])} ft; "
                        f"maximum bay {geometry['unsupported_span_ft']:.2f} ft against the matched "
                        f"{geometry['reference_geometry_rule']['span_max_ft']:.2f} ft reference limit. "
                        f"Longitudinal column stations are spaced {geometry['column_spacing_actual_ft']:.2f} ft apart."
                    ),
                    "web_layout": geometry["reference_geometry_rule"]["web_layout"],
                    "purlin_spacing_ft": geometry["purlin_spacing_ft"],
                    "overhang_ft": geometry["overhang_ft"],
                    "reference_design_id": geometry["reference_geometry_rule"]["reference_design_id"],
                    "reference_span_limit_ft": geometry["reference_geometry_rule"]["span_max_ft"],
                    "reference_pipe_aliases": geometry["reference_geometry_rule"]["pipe_aliases"],
                    "rule_adjustments": (
                        (
                            [{
                                "parameter": "column_total_count",
                                "requested": geometry["predicted_column_total_count"],
                                "resolved": geometry["final_column_total_count"],
                                "reason": (
                                    (
                                        "The odd predicted total is rounded up once to the next even total "
                                        "and divided equally across the two exterior support lines."
                                    )
                                    if geometry["support_line_count"] == 2
                                    else (
                                        "The nearest feasible physical total is distributed equally across "
                                        "the three explicitly selected support lines, with at least two "
                                        "end-inclusive stations per line."
                                    )
                                ),
                            }]
                            if round(geometry["predicted_column_total_count"])
                            != geometry["final_column_total_count"] else []
                        )
                        + (
                            [{
                                "parameter": "column_spacing_along_length_ft",
                                "requested": geometry["predicted_column_spacing_along_length_ft"],
                                "resolved": geometry["column_spacing_actual_ft"],
                                "reason": (
                                    "The final spacing is calculated from the resolved column count and "
                                    "equal end-inclusive stations; the model value is retained as a diagnostic."
                                ),
                            }]
                            if geometry["predicted_column_spacing_along_length_ft"] is not None
                            and not abs(
                                geometry["predicted_column_spacing_along_length_ft"]
                                - geometry["column_spacing_actual_ft"]
                            ) <= 0.5 else []
                        )
                    ),
                    "engineering_verification_required": True,
                    "checks": checks,
                    "warnings": list(geometry["warnings"]) + ([
                        "Multi-support layout is generated from the 80% reference-span heuristic; the reference row describes a middle-support configuration, not a structural approval for arbitrary support counts. Engineer verification is required."
                    ] if strategy == "multi_support" else []),
                    "design": design_values,
                    "inputs": candidate_inputs,
                    "geometry": geometry,
                    "bom": bom,
                }
                option["label"] = _candidate_label(option)
                options.append(option)
            except (RuntimeError, TypeError, ValueError) as exc:
                category = (
                    "Reference rule"
                    if "reference" in str(exc).casefold()
                    else "Model inference"
                    if isinstance(exc, RuntimeError)
                    else "Geometry/BOM validation"
                )
                rejections.append({
                    "truss_family": family,
                    "support_configuration": support_label,
                    "overall_span_ft": overall_span,
                    "unsupported_span_ft": unsupported_span,
                    "reference_span_limit_ft": reference_limit,
                    "middle_support_evaluated": middle_support,
                    "middle_support_rule_available": middle_rule_available,
                    "failure_category": category,
                    "reason": str(exc),
                })

            # The UI displays at most three options. Stop expensive geometry/BOM
            # work once this family already has three fully validated candidates.
            # Support variants are generated in minimum-support-first order.
            if sum(1 for option in options if option.get("truss_family") == family) >= 3:
                break

        # In MANUAL mode, don't spend time evaluating fallback families if the
        # requested family already has a validated candidate. Fallback families
        # are evaluated only when every attempted layout of the requested family fails.
        if (
            not is_auto
            and preferred_family
            and family == preferred_family
            and any(option.get("truss_family") == preferred_family for option in options)
        ):
            break

    # In MANUAL mode, show the selected family if it has any fully validated
    # candidates. If all its layouts fail, keep the nearest compatible validated
    # alternatives so the app can offer a usable fallback instead of a rejected
    # geometry/BOM result.
    options = _manual_family_priority(options, preferred_family, mode)

    options = _rank_and_limit_options(
        options,
        preferred_family=preferred_family,
        prefer_unsupported=True,
        limit=3,
    )
    return options, rejections
