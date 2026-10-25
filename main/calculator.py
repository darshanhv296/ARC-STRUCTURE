"""BOM quantities, pipe purchasing, and roofing takeoff from generated geometry.

This module intentionally consumes existing generated geometry. It does not change
truss selection, support placement, geometry generation, or ML predictions.
"""
from __future__ import annotations

import itertools
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT

PIPE_RESERVE_PCT = 10
SHEET_RESERVE_PCT = 10
SCREW_RESERVE_PCT = 20
TILE_RESERVE_PCT = 15
BOM_CALCULATION_VERSION = "bom-v3-total-length-reserves"
SHEET_LENGTHS_FT = (6, 8, 10, 12, 14, 16, 18, 20, 22)
SHEET_NOMINAL_WIDTH_IN = 42.0
SHEET_SIDE_OVERLAP_IN = 3.0
SHEET_END_LAP_FT = 2.0
TILE_PITCH_IN = 9.0
SCREWS_PER_SHEET_PER_PURLIN = 3


def _normalize(value):
    return "".join(str(value).casefold().split()).replace("×", "x")


def _pipe_record(section, sections):
    label = str(section).strip()
    record = sections.get(_normalize(label))
    if record is not None:
        return record
    without_id = re.sub(r"\s*\([A-Za-z]*\d+\)\s*$", "", label).strip()
    return sections.get(_normalize(without_id))


def _to_float(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=4)
def _load_pipe_sections_cached(pipe_file_text: str, mtime_ns: int, file_size: int):
    sections = {}
    frame = pd.read_csv(Path(pipe_file_text))
    required_columns = {"pipe_size", "standard_length_ft", "weight_kg"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "Pipe catalogue is missing required columns: "
            + ", ".join(sorted(missing_columns))
            + "."
        )
    for row in frame.to_dict("records"):
        sections[_normalize(row["pipe_size"])] = row
        if row.get("pipe_id"):
            sections[_normalize(row["pipe_id"])] = row
    return sections


def _best_sheet_combo(target_coverage_ft: float, end_lap_ft: float = SHEET_END_LAP_FT):
    """Return a minimum-purchase-length combination from the allowed sheet lengths."""
    if target_coverage_ft <= 0:
        raise ValueError("Roof slope coverage must be positive.")
    # Each additional sheet adds its length minus one end-lap to effective coverage.
    min_count = max(1, math.ceil((target_coverage_ft - end_lap_ft) / (max(SHEET_LENGTHS_FT) - end_lap_ft)))
    for count in range(min_count, min_count + 20):
        required_purchase = target_coverage_ft + end_lap_ft * (count - 1)
        total_purchase = int(math.ceil((required_purchase - 1e-9) / 2.0) * 2)
        if total_purchase < count * min(SHEET_LENGTHS_FT) or total_purchase > count * max(SHEET_LENGTHS_FT):
            continue
        combos = [c for c in itertools.combinations_with_replacement(SHEET_LENGTHS_FT, count) if sum(c) == total_purchase]
        if not combos:
            continue
        # For equal purchase totals, prefer the most balanced cut combination.
        combo = min(combos, key=lambda c: (max(c) - min(c), c))
        effective = sum(combo) - end_lap_ft * (len(combo) - 1)
        if effective + 1e-9 >= target_coverage_ft:
            finished = [float(length) for length in combo]
            excess = max(0.0, effective - target_coverage_ft)
            # Trim only the final piece to the required finished coverage. The
            # stock purchase lengths remain the selected standard lengths.
            if finished and excess > 1e-9:
                finished[-1] = max(0.0, finished[-1] - excess)
            return {
                "sheet_lengths_ft": list(combo),
                "finished_cut_lengths_ft": [round(length, 4) for length in finished],
                "sheet_piece_count_per_width_position": len(combo),
                "purchase_length_per_width_position_ft": float(sum(combo)),
                "effective_coverage_ft": round(sum(finished) - end_lap_ft * (len(finished) - 1), 4),
                "end_lap_ft": end_lap_ft,
                "end_lap_joint_count": len(combo) - 1,
                "trim_waste_per_width_position_ft": round(sum(combo) - sum(finished), 4),
            }
    raise ValueError(f"Could not find a valid sheet combination for {target_coverage_ft:.3f} ft roof coverage.")


def _purlin_rows_by_slope(geometry):
    """Read row counts from generated purlin members first, with layout as fallback."""
    purlins = [m for m in geometry.get("members", []) if str(m.get("member_type", "")).casefold() == "purlin"]
    row_members = defaultdict(set)
    row_runs = defaultdict(set)
    for member in purlins:
        slope = str(member.get("roof_slope", "single")).casefold()
        row_index = member.get("purlin_row_index", member.get("longitudinal_run_index"))
        if row_index is not None:
            row_members[slope].add(str(row_index))
            run_index = member.get("longitudinal_run_index", row_index)
            row_runs[slope].add(str(run_index))
    layout = geometry.get("purlin_layout", {})
    roof = str(geometry.get("roof_configuration", "single_slope")).casefold()
    if row_members:
        # The design labels each purlin run by its roof slope.
        left = len(row_members.get("left", set()))
        right = len(row_members.get("right", set()))
        single = len(row_members.get("single", set()))
        if roof == "gable" and (left or right):
            return {"left": left, "right": right}
        if single:
            return {"single": single}
    if roof == "gable":
        return {
            "left": int(layout.get("left_slope_row_count", 0) or 0),
            "right": int(layout.get("right_slope_row_count", 0) or 0),
        }
    return {"single": int(layout.get("single_slope_row_count", layout.get("total_slope_rows", 0)) or 0)}


def _purlin_run_length_ft(geometry):
    """Recover a full longitudinal purlin run by summing its generated segments."""
    by_run = defaultdict(float)
    for member in geometry.get("members", []):
        if str(member.get("member_type", "")).casefold() != "purlin":
            continue
        run = member.get("longitudinal_run_index", member.get("purlin_row_index"))
        if run is not None:
            by_run[str(run)] += _to_float(member.get("length_ft"), 0.0) or 0.0
    if by_run:
        return max(by_run.values())
    return _to_float(geometry.get("length_ft"), _to_float(geometry.get("overall_building_length_ft"), 0.0)) or 0.0


def _purlin_run_lengths_by_slope(geometry):
    """Recover the longest generated longitudinal purlin run on each roof slope."""
    by_run = defaultdict(float)
    for member in geometry.get("members", []):
        if str(member.get("member_type", "")).casefold() != "purlin":
            continue
        run = member.get("longitudinal_run_index", member.get("purlin_row_index"))
        if run is not None:
            slope = str(member.get("roof_slope", "single")).casefold()
            by_run[(slope, str(run))] += _to_float(member.get("length_ft"), 0.0) or 0.0
    lengths = defaultdict(list)
    for (slope, _), length in by_run.items():
        lengths[slope].append(length)
    return {slope: max(run_lengths) for slope, run_lengths in lengths.items()}


def _purlin_member_summary(geometry):
    purlins = [
        member for member in geometry.get("members", [])
        if str(member.get("member_type", "")).casefold() == "purlin"
    ]
    row_keys = {
        (
            str(member.get("roof_slope", "single")).casefold(),
            str(member.get("purlin_row_index", member.get("longitudinal_run_index", member.get("member_id")))),
        )
        for member in purlins
    }
    layout = geometry.get("purlin_layout", {}) or {}
    fallback_rows = sum(
        int(layout.get(key, 0) or 0)
        for key in ("left_slope_row_count", "right_slope_row_count", "single_slope_row_count")
    )
    return {
        "row_count": len(row_keys) if row_keys else fallback_rows,
        "physical_segment_count": len(purlins),
        "net_length_ft": round(sum(_to_float(member.get("length_ft"), 0.0) or 0.0 for member in purlins), 6),
    }


def _roofing_takeoff(geometry):
    roofing = geometry.get("roofing_layer", {}) or {}
    roofing_type = str(roofing.get("roofing_type", "")).strip()
    label = roofing_type.casefold()
    rows = _purlin_rows_by_slope(geometry)
    purlin_length_ft = _purlin_run_length_ft(geometry)
    slope_length_ft = _to_float(
        geometry.get("purlin_layout", {}).get("sloped_roof_length_ft"),
        _to_float(geometry.get("roof_slope_length_ft"), None),
    )
    if slope_length_ft is None or slope_length_ft <= 0:
        return {"status": "incomplete", "roofing_type": roofing_type, "warning": "Generated roof slope length is missing; roofing takeoff was not estimated."}
    if purlin_length_ft <= 0:
        return {"status": "incomplete", "roofing_type": roofing_type, "warning": "Generated purlin run length is missing; roofing takeoff was not estimated."}

    roof = str(geometry.get("roof_configuration", "single_slope")).casefold()
    slopes = [(name, count) for name, count in rows.items() if count > 0]
    if not slopes:
        return {"status": "incomplete", "roofing_type": roofing_type, "warning": "No purlin rows were found in the generated design."}

    if "tile" in label:
        slope_results = []
        total_base = 0
        purlin_run_lengths = _purlin_run_lengths_by_slope(geometry)
        for side, purlin_count in slopes:
            slope_purlin_length_ft = _to_float(purlin_run_lengths.get(side), purlin_length_ft)
            if slope_purlin_length_ft <= 0:
                return {"status": "incomplete", "roofing_type": roofing_type, "warning": f"Generated purlin run length is missing for the {side} roof slope; tile takeoff was not estimated."}
            box_rows = max(0, purlin_count - 1)
            tiles_per_row = int(math.ceil(slope_purlin_length_ft * 12.0 / TILE_PITCH_IN - 1e-12))
            base = box_rows * tiles_per_row
            total_base += base
            slope_results.append({
                "slope": side,
                "generated_purlin_rows": purlin_count,
                "tile_box_rows": box_rows,
                "purlin_length_ft": round(slope_purlin_length_ft, 4),
                "tile_length_in": TILE_PITCH_IN,
                "tiles_per_row": tiles_per_row,
                "base_tiles": base,
            })
        reserve = int(math.ceil(total_base * TILE_RESERVE_PCT / 100.0 - 1e-12))
        purchased = total_base + reserve
        return {
            "status": "calculated", "roofing_type": roofing_type,
            "tile_pitch_in": TILE_PITCH_IN,
            "tile_length_in": TILE_PITCH_IN,
            "purlin_length_ft": round(purlin_length_ft, 4),
            "slope_details": slope_results, "base_tile_count": total_base,
            "reserve_pct": TILE_RESERVE_PCT, "reserve_tile_count": reserve,
            "tile_count_to_purchase": purchased,
            "warning": "Tile-box rows count the spaces between adjacent generated purlin rows; the tile takeoff uses each slope's generated longitudinal purlin length.",
        }

    if "metal" in label or "sheet" in label:
        site_length_ft = _to_float(geometry.get("length_ft"), purlin_length_ft) or purlin_length_ft
        across_width_in = site_length_ft * 12.0
        # Effective width = 42*n - 3*(n-1) = 39*n + 3 inches.
        sheets_across = max(1, int(math.ceil((across_width_in - SHEET_SIDE_OVERLAP_IN) / (SHEET_NOMINAL_WIDTH_IN - SHEET_SIDE_OVERLAP_IN) - 1e-12)))
        combo = _best_sheet_combo(slope_length_ft, SHEET_END_LAP_FT)
        slope_details = []
        total_sheet_pieces = 0
        total_screws_base = 0
        for side, purlin_count in slopes:
            sheet_pieces = sheets_across * len(combo["sheet_lengths_ft"])
            total_sheet_pieces += sheet_pieces
            screws_for_slope = sheet_pieces * purlin_count * SCREWS_PER_SHEET_PER_PURLIN
            total_screws_base += screws_for_slope
            slope_details.append({
                "slope": side,
                "generated_purlin_rows": purlin_count,
                "sheets_across_building_length": sheets_across,
                "purchase_sheet_lengths_ft_per_width_position": combo["sheet_lengths_ft"],
                "finished_cut_lengths_ft_per_width_position": combo["finished_cut_lengths_ft"],
                "trim_waste_per_width_position_ft": combo["trim_waste_per_width_position_ft"],
                "physical_sheet_piece_count": sheet_pieces,
                "base_sheet_count": sheet_pieces,
                "required_slope_coverage_ft": round(slope_length_ft, 4),
                "effective_slope_coverage_ft": combo["effective_coverage_ft"],
                "purchased_sheet_length_per_width_position_ft": combo["purchase_length_per_width_position_ft"],
                "base_screws": screws_for_slope,
            })
        sheet_reserve = int(math.ceil(total_sheet_pieces * SHEET_RESERVE_PCT / 100.0 - 1e-12))
        sheets_purchase = total_sheet_pieces + sheet_reserve
        screw_reserve = int(math.ceil(total_screws_base * SCREW_RESERVE_PCT / 100.0 - 1e-12))
        screws_purchase = total_screws_base + screw_reserve
        return {
            "status": "calculated", "roofing_type": roofing_type,
            "nominal_sheet_width_in": SHEET_NOMINAL_WIDTH_IN,
            "side_overlap_in": SHEET_SIDE_OVERLAP_IN,
            "effective_sheet_width_in": SHEET_NOMINAL_WIDTH_IN - SHEET_SIDE_OVERLAP_IN,
            "building_length_ft": round(site_length_ft, 4),
            "sheets_across_building_length_per_slope": sheets_across,
            "slope_length_ft": round(slope_length_ft, 4),
            "end_lap_ft": SHEET_END_LAP_FT,
            "selected_purchase_sheet_lengths_ft_per_width_position": combo["sheet_lengths_ft"],
            "finished_cut_lengths_ft_per_width_position": combo["finished_cut_lengths_ft"],
            "trim_waste_per_width_position_ft": combo["trim_waste_per_width_position_ft"],
            "sheet_piece_count_per_width_position": len(combo["sheet_lengths_ft"]),
            "base_sheet_count": total_sheet_pieces,
            "sheet_reserve_pct": SHEET_RESERVE_PCT,
            "sheet_reserve_count": sheet_reserve,
            "total_sheet_piece_count": sheets_purchase,
            "final_sheet_count": sheets_purchase,
            "slope_details": slope_details,
            "base_screw_count": total_screws_base,
            "screw_reserve_pct": SCREW_RESERVE_PCT,
            "screw_reserve_count": screw_reserve,
            "screw_count_to_purchase": screws_purchase,
            "warning": "Sheet quantities use generated slope length and actual purlin rows. Sheet and screw reserves are applied once to their respective base quantities. Confirm supplier usable width, laps, and fixing schedule before ordering.",
        }
    return {"status": "unsupported", "roofing_type": roofing_type, "warning": f"No roofing takeoff rule is configured for '{roofing_type}'."}


def calculate_bom(geometry: dict) -> dict:
    members = geometry.get("members", []) if geometry else []
    family = str(geometry.get("truss_family", "")).strip().casefold() if geometry else ""
    if "plain pipe" in family or "single rafter" in family:
        if any(str(member.get("member_type", "")).startswith("Web") for member in members):
            return {"connected": False, "message": "Plain Pipe geometry contains unsupported web members; no BOM was generated."}
    if not members:
        return {"connected": False, "message": "No generated members were provided to the material calculator."}

    validation = geometry.get("validation", {})
    required_checks = {
        "all_endpoints_have_node_ids": validation.get("all_endpoints_have_node_ids", False),
        "truss_components_connected": validation.get("truss_components_connected", False),
        "all_members_connected": validation.get("all_members_connected", False),
        "all_support_surfaces_aligned": validation.get("all_support_surfaces_aligned", False),
        "purlins_bear_on_trusses": validation.get("purlins_bear_on_trusses", False),
        "no member bounds violations": not validation.get("member_bounds_violations", ["unvalidated"]),
        "no excluded zero-length members": validation.get("zero_length_members_excluded", 1) == 0,
        "no removed duplicate members": validation.get("duplicate_members_removed", 1) == 0,
    }
    failed_checks = [name for name, passed in required_checks.items() if not passed]
    if failed_checks:
        return {"connected": False, "message": "Geometry validation failed; no BOM was generated. Failed checks: " + ", ".join(failed_checks) + "."}

    pipe_file = Path(PROJECT_ROOT) / "pipes.csv"
    if not pipe_file.is_file():
        return {"connected": False, "message": f"Pipe database not found: {pipe_file.name}."}
    stat = pipe_file.stat()
    try:
        sections = _load_pipe_sections_cached(str(pipe_file), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return {"connected": False, "message": f"Pipe catalogue could not be read: {exc}"}
    grouped = defaultdict(list)
    section_records = {}
    seen_members = set()
    oversized_members = []
    for member in members:
        start, end = member.get("start"), member.get("end")
        if not isinstance(start, (tuple, list)) or not isinstance(end, (tuple, list)) or len(start) != 3 or len(end) != 3:
            return {"connected": False, "message": f"Generated member {member.get('member_id', '?')} has missing or invalid 3D endpoints."}
        try:
            coords = [float(v) for v in (*start, *end)]
        except (TypeError, ValueError):
            coords = []
        if len(coords) != 6 or not all(math.isfinite(v) for v in coords):
            return {"connected": False, "message": f"Generated member {member.get('member_id', '?')} has non-finite endpoint coordinates."}
        coord_length = math.dist(coords[:3], coords[3:])
        if coord_length <= 1e-8:
            return {"connected": False, "message": f"Generated member {member.get('member_id', '?')} has zero length."}
        section = str(member.get("pipe_section", "")).strip()
        pipe = _pipe_record(section, sections)
        if pipe is None:
            return {"connected": False, "message": f"Pipe section '{section or 'unspecified'}' is not present in pipes.csv."}
        stock_length = _to_float(pipe.get("standard_length_ft"))
        stock_weight = _to_float(pipe.get("weight_kg"))
        if stock_length is None or stock_length <= 0:
            return {"connected": False, "message": f"Pipe section '{section}' has no valid catalog stock length."}
        if stock_weight is None or stock_weight <= 0:
            return {"connected": False, "message": f"Pipe section '{section}' has no valid catalog weight."}
        length = _to_float(member.get("length_ft"))
        if length is None or length <= 0 or not math.isclose(length, coord_length, rel_tol=1e-8, abs_tol=1e-7):
            return {"connected": False, "message": f"Generated member {member.get('member_id', '?')} has a cut length inconsistent with its endpoints."}
        endpoints = tuple(sorted((tuple(round(v, 7) for v in coords[:3]), tuple(round(v, 7) for v in coords[3:]))))
        member_key = (str(member.get("member_type", "Member")), section, endpoints, member.get("truss_index"))
        if member_key in seen_members:
            return {"connected": False, "message": f"Generated member {member.get('member_id', '?')} duplicates another member."}
        seen_members.add(member_key)
        if length > stock_length + 1e-9:
            oversized_members.append({
                "member_id": member.get("member_id"),
                "member_type": str(member.get("member_type", "Member")),
                "pipe_section": pipe["pipe_size"],
                "member_length_ft": round(length, 6),
                "stock_length_ft": stock_length,
            })
        member_type = str(member.get("member_type", "Member"))
        grouped[(member_type, pipe["pipe_size"])].append(member)
        section_records[pipe["pipe_size"]] = pipe

    items = []
    net_length_ft = 0.0
    net_weight_kg = 0.0
    section_summaries = []
    section_totals = defaultdict(lambda: {
        "count": 0,
        "net_member_length_ft": 0.0,
        "net_member_weight_kg": 0.0,
        "member_types": set(),
    })
    for (member_type, section), group in sorted(grouped.items()):
        pipe = section_records[section]
        length_values = [float(m["length_ft"]) for m in group]
        total = sum(length_values)
        stock_length = _to_float(pipe.get("standard_length_ft"))
        stock_weight = _to_float(pipe.get("weight_kg"))
        weight = total * (stock_weight / stock_length)
        net_length_ft += total
        net_weight_kg += weight
        aggregate = section_totals[section]
        aggregate["count"] += len(group)
        aggregate["net_member_length_ft"] += total
        aggregate["net_member_weight_kg"] += weight
        aggregate["member_types"].add(member_type)
        items.append({"category": "Structural pipe", "member_type": member_type, "pipe_section": section,
                      "count": len(group), "total_length_ft": round(total, 6),
                      "net_weight_kg": round(weight, 2),
                      "stock_length_ft": stock_length})

    member_group_totals = [{
        "member_type": member_type,
        "pipe_section": section,
        "count": len(group),
        "net_length_ft": round(sum(float(member["length_ft"]) for member in group), 6),
        "net_weight_kg": round(
            sum(float(member["length_ft"]) for member in group)
            * _to_float(section_records[section].get("weight_kg"))
            / _to_float(section_records[section].get("standard_length_ft")),
            2,
        ),
    } for (member_type, section), group in sorted(grouped.items())]

    base_pipes_total = 0
    extra_pipes_total = 0
    final_pipes_total = 0
    purchased_weight_kg = 0.0
    for section, aggregate in sorted(section_totals.items()):
        pipe = section_records[section]
        stock_length = _to_float(pipe.get("standard_length_ft"))
        stock_weight = _to_float(pipe.get("weight_kg"))
        section_net = aggregate["net_member_length_ft"]
        required_pipes = int(math.ceil(section_net / stock_length - 1e-12))
        extra_pipes = int(math.ceil(required_pipes * PIPE_RESERVE_PCT / 100.0 - 1e-12))
        final_pipes = required_pipes + extra_pipes
        section_base_weight = required_pipes * stock_weight
        section_reserve_weight = extra_pipes * stock_weight
        section_purchase_weight = final_pipes * stock_weight
        base_pipes_total += required_pipes
        extra_pipes_total += extra_pipes
        final_pipes_total += final_pipes
        purchased_weight_kg += section_purchase_weight
        section_summaries.append({
            "pipe_section": section,
            "member_types": sorted(aggregate["member_types"]),
            "member_count": aggregate["count"],
            "stock_length_ft": stock_length,
            "net_member_length_ft": round(section_net, 6),
            "pipes_required": required_pipes,
            "base_stock_bars": required_pipes,
            "effective_base_stock_bars": required_pipes,
            "extra_pipes": extra_pipes,
            "reserve_bars": extra_pipes,
            "final_pipes_to_buy": final_pipes,
            "purchase_bars": final_pipes,
            "purchase_stock_bars": final_pipes,
            "reserve_pct": PIPE_RESERVE_PCT,
            "unit_weight_kg": round(stock_weight, 3),
            "weight_per_stock_bar_kg": round(stock_weight, 3),
            "total_purchase_weight_kg": round(section_purchase_weight, 2),
            "base_purchased_material_weight_kg": round(section_base_weight, 2),
            "reserve_weight_kg": round(section_reserve_weight, 2),
            "purchased_material_weight_kg": round(section_purchase_weight, 2),
            "net_member_weight_kg": round(aggregate["net_member_weight_kg"], 2),
        })

    roofing = _roofing_takeoff(geometry)
    purlin_summary = _purlin_member_summary(geometry)
    if roofing.get("status") == "calculated":
        if "tile_count_to_purchase" in roofing:
            items.append({"category": "Roofing", "member_type": "Roofing tile", "pipe_section": "N/A",
                          "count": roofing["tile_count_to_purchase"], "single_length_ft": None, "total_length_ft": None,
                          "net_weight_kg": None, "stock_length_ft": None})
        else:
            items.append({"category": "Roofing", "member_type": "Roofing sheet", "pipe_section": "N/A",
                          "count": roofing["total_sheet_piece_count"], "single_length_ft": None, "total_length_ft": None,
                          "net_weight_kg": None, "stock_length_ft": None})
            items.append({"category": "Fasteners", "member_type": "Roofing screw", "pipe_section": "N/A",
                          "count": roofing["screw_count_to_purchase"], "single_length_ft": None, "total_length_ft": None,
                          "net_weight_kg": None, "stock_length_ft": None})

    base_purchase_weight_kg = sum(
        section.get("base_purchased_material_weight_kg", 0) for section in section_summaries
    )
    reserve_weight_kg = sum(
        section.get("reserve_weight_kg", 0) for section in section_summaries
    )
    total_purchase_weight_kg = sum(
        section.get("purchased_material_weight_kg", 0) for section in section_summaries
    )
    return {
        "connected": True,
        "calculation_version": BOM_CALCULATION_VERSION,
        "excluded_plain_pipe_web_members": 0,
        "items": items,
        "member_group_totals": member_group_totals,
        "pipe_sections": section_summaries,
        "member_count": len(members),
        "purlin_rows": purlin_summary["row_count"],
        "purlin_physical_segment_count": purlin_summary["physical_segment_count"],
        "purlin_net_length_ft": purlin_summary["net_length_ft"],
        "total_length_ft": round(net_length_ft, 2),
        "material_weight_kg": round(net_weight_kg, 2),
        "pipes_required": base_pipes_total,
        "extra_pipes": extra_pipes_total,
        "final_pipes_to_buy": final_pipes_total,
        "total_purchase_weight_kg": round(total_purchase_weight_kg, 2),
        "base_purchased_material_weight_kg": round(base_purchase_weight_kg, 2),
        "reserve_weight_kg": round(reserve_weight_kg, 2),
        "purchased_material_weight_kg": round(total_purchase_weight_kg, 2),
        "purchase_stock_bars": final_pipes_total,
        "reserve_pct": PIPE_RESERVE_PCT,
        "oversized_members": oversized_members,
        "roofing_material_takeoff": roofing,
        "message": "Structural pipe quantities use total generated member length divided by catalogue stock length, rounded up, with one reserve applied per pipe size. No cutting optimization or splice design is implied.",
    }
