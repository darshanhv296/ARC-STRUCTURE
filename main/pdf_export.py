"""PDF export for the current generated TRUSS geometry and BOM."""
from __future__ import annotations

from collections import Counter
from io import BytesIO
import math
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether,
)


def _fmt(value, digits=2):
    try:
        number = float(value)
        return f"{number:,.{digits}f}" if math.isfinite(number) else "N/A"
    except (TypeError, ValueError):
        return "N/A" if value is None else str(value)


def _fmt_ft_in(value_ft):
    """Format a length as feet and whole inches (e.g. 12 ft 3 in)."""
    try:
        inches = max(0, int(round(float(value_ft) * 12.0)))
    except (TypeError, ValueError):
        return "N/A"
    feet, rem = divmod(inches, 12)
    return f"{feet} ft {rem} in"

def _truss_member_frame(geometry):
    """Select one generated truss frame and retain its actual 3D coordinates."""
    truss_members = []
    for member in geometry.get("members", []):
        kind = str(member.get("member_type", "")).casefold()
        if any(token in kind for token in ("column", "beam", "purlin", "roofing", "roof sheet")):
            continue
        start, end = member.get("start"), member.get("end")
        if not (isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) == len(end) == 3):
            continue
        try:
            p1, p2 = tuple(float(v) for v in start), tuple(float(v) for v in end)
            length = float(member.get("length_ft", math.dist(p1, p2)))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (*p1, *p2, length)) or length <= 0:
            continue
        truss_members.append((member, p1, p2, length))
    if not truss_members:
        raise ValueError("No generated truss members available for export.")
    indexed = {}
    for item in truss_members:
        idx = item[0].get("truss_index")
        if idx is not None:
            indexed.setdefault(str(idx), []).append(item)
    if indexed:
        return max(indexed.values(), key=len)
    planes = {}
    for item in truss_members:
        y = round((item[1][1] + item[2][1]) / 2.0, 2)
        planes.setdefault(y, []).append(item)
    return max(planes.values(), key=len)

def _top_chord_totals(geometry, frame):
    top_chords = [item for item in frame if "top chord" in str(item[0].get("member_type", "")).casefold()]
    if not top_chords:
        raise ValueError("The selected generated truss contains no top-chord members.")
    lengths = [float(item[3]) for item in top_chords]
    if str(geometry.get("roof_configuration", "")).casefold() == "single_slope":
        return {
            "single_slope_ft": sum(lengths),
            "single_slope_piece_count": len(lengths),
            "other_slope_ft": None,
            "other_slope_piece_count": 0,
            "combined_ft": sum(lengths),
            "combined_piece_count": len(lengths),
        }
    x_mid = (min(min(item[1][0], item[2][0]) for item in top_chords)
             + max(max(item[1][0], item[2][0]) for item in top_chords)) / 2.0
    left = [item for item in top_chords if (item[1][0] + item[2][0]) / 2.0 <= x_mid]
    right = [item for item in top_chords if (item[1][0] + item[2][0]) / 2.0 > x_mid]
    return {
        "left_ft": sum(float(item[3]) for item in left),
        "left_piece_count": len(left),
        "right_ft": sum(float(item[3]) for item in right),
        "right_piece_count": len(right),
        "combined_ft": sum(lengths),
        "combined_piece_count": len(lengths),
    }


def _make_cad_front_image(geometry):
    """Create a dimensioned 2D CAD-like elevation of one generated truss."""
    frame = _truss_member_frame(geometry)

    fig, ax = plt.subplots(figsize=(12.0, 5.4), dpi=180)
    colors_by_type = {
        "top": "#176b87", "bottom": "#d65f3c", "web": "#32866f",
        "king": "#8c6bb1", "default": "#536878",
    }
    all_x, all_z = [], []
    top_totals = _top_chord_totals(geometry, frame)
    for member, p1, p2, length in frame:
        kind = str(member.get("member_type", "member")).casefold()
        if "top chord" in kind or "top_chord" in kind:
            color = colors_by_type["top"]
        elif "bottom chord" in kind or "bottom_chord" in kind:
            color = colors_by_type["bottom"]
        elif "king" in kind:
            color = colors_by_type["king"]
        elif "web" in kind or "vertical" in kind:
            color = colors_by_type["web"]
        else:
            color = colors_by_type["default"]
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], color=color, linewidth=2.0, solid_capstyle="round", zorder=2)
        all_x.extend([p1[0], p2[0]])
        all_z.extend([p1[2], p2[2]])
        # Label member lengths in feet/inches; offset alternating labels to reduce overlap.
        mx, mz = (p1[0]+p2[0])/2.0, (p1[2]+p2[2])/2.0
        dx, dz = p2[0]-p1[0], p2[2]-p1[2]
        norm = max(math.hypot(dx, dz), 1e-9)
        offset = 0.16 if "bottom chord" not in kind.casefold() else -0.18
        ax.text(mx - dz/norm*offset, mz + dx/norm*offset, _fmt_ft_in(length), fontsize=6.1,
                color="#17212b", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.13", fc="white", ec="#b8c2ca", alpha=.88, lw=.45), zorder=4)

    if not all_x:
        plt.close(fig)
        raise ValueError("Selected truss frame has no drawable coordinates.")
    min_x, max_x = min(all_x), max(all_x)
    min_z, max_z = min(all_z), max(all_z)
    span = max_x - min_x
    rise = max_z - min_z
    mid_x = (min_x + max_x) / 2.0
    # Overall span and rise dimension lines, with explicit ft/in labels.
    y_dim = min_z - max(rise * .14, .45)
    ax.annotate("", xy=(min_x, y_dim), xytext=(max_x, y_dim),
                arrowprops=dict(arrowstyle="<->", color="#374151", lw=1.0))
    ax.text(mid_x, y_dim - max(rise * .045, .12), f"SPAN {_fmt_ft_in(span)}",
            ha="center", va="top", fontsize=8.5, fontweight="bold", color="#263746")
    x_dim = min_x - max(span * .045, .45)
    ax.annotate("", xy=(x_dim, min_z), xytext=(x_dim, max_z),
                arrowprops=dict(arrowstyle="<->", color="#374151", lw=1.0))
    ax.text(x_dim - max(span * .012, .1), (min_z+max_z)/2, f"RISE\n{_fmt_ft_in(rise)}",
            ha="right", va="center", fontsize=7.5, color="#263746")
    # Report the cumulative top-chord lengths on both roof slopes.
    summary_y = max_z + max(rise * .10, .35)
    if str(geometry.get("roof_configuration", "")).casefold() == "single_slope":
        summary = (
                f"Single slope top chord: {_fmt_ft_in(top_totals['single_slope_ft'])} "
                f"({top_totals['single_slope_piece_count']} pieces) · Other slope: Not applicable"
        )
    else:
        summary = (
            f"Left: {_fmt_ft_in(top_totals['left_ft'])} ({top_totals['left_piece_count']} pieces)\n"
            f"Right: {_fmt_ft_in(top_totals['right_ft'])} ({top_totals['right_piece_count']} pieces)\n"
            f"Combined: {_fmt_ft_in(top_totals['combined_ft'])} ({top_totals['combined_piece_count']} pieces)"
        )
    ax.text((min_x + max_x) / 2.0, summary_y, summary, ha="center", va="bottom",
            fontsize=7.5, color=colors_by_type["top"], fontweight="bold")
    resolved = geometry.get("resolved_parameters", {})
    family = geometry.get("truss_family", resolved.get("truss_family", "N/A"))
    chord = resolved.get("frame_chord_pipe_size", "N/A")
    web = resolved.get("web_pipe_size", "Not applicable")
    ax.set_title(
        f"{family} — CAD Front Elevation / Member Lengths\n"
        f"Chord: {chord} · Web: {web}",
        fontsize=11, pad=12,
    )
    ax.set_xlabel("Horizontal projection (ft)", fontsize=8)
    ax.set_ylabel("Height (ft)", fontsize=8)
    ax.grid(True, color="#d9e0e5", linewidth=.55, alpha=.75)
    ax.set_aspect("auto")
    ax.set_xlim(min_x - max(span*.10, 1.0), max_x + max(span*.04, .5))
    ax.set_ylim(y_dim - max(rise*.10, .35), summary_y + max(rise*.22, .65))
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#9aa7b1")
    fig.tight_layout(pad=1.2)
    output = BytesIO()
    fig.savefig(output, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output.seek(0)
    return output


def _make_3d_image(geometry, individual_truss=False):
    members = geometry.get("members", [])
    if individual_truss:
        members = [item[0] for item in _truss_member_frame(geometry)]
    fig = plt.figure(figsize=(11.2, 6.5), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    palette = {
        "column": "#526c62", "beam": "#c56a4a", "purlin": "#c6a15b",
        "web": "#9b815d", "chord": "#28677d", "roof": "#c6a15b",
    }
    all_points = []

    # Draw the generated roof covering as a translucent surface so the PDF
    # represents the full building envelope while keeping framing visible.
    try:
        span = float(geometry.get("span_ft", 0) or 0)
        length = float(geometry.get("length_ft", 0) or 0)
        rise = float(geometry.get("rise_height_ft", 0) or 0)
        base = float(geometry.get("roof_surface_base_z_ft", 0) or 0)
        roof_left, roof_right = geometry.get("roof_profile_x_bounds_ft", [0.0, span])
        roof_left, roof_right = float(roof_left), float(roof_right)
        roof_kind = str(geometry.get("roof_configuration", "single_slope")).casefold()

        def roof_z(x):
            if span <= 0:
                return base
            if roof_kind == "gable":
                profile = rise * (2.0 * x / span if x <= span / 2.0 else 2.0 - 2.0 * x / span)
            else:
                profile = rise * x / span
            return base + profile

        if not individual_truss and span > 0 and length > 0 and roof_right > roof_left:
            if roof_kind == "gable":
                ridge = span / 2.0
                roof_faces = [
                    [(roof_left, 0.0, roof_z(roof_left)), (ridge, 0.0, roof_z(ridge)),
                     (ridge, length, roof_z(ridge)), (roof_left, length, roof_z(roof_left))],
                    [(ridge, 0.0, roof_z(ridge)), (roof_right, 0.0, roof_z(roof_right)),
                     (roof_right, length, roof_z(roof_right)), (ridge, length, roof_z(ridge))],
                ]
            else:
                roof_faces = [[(roof_left, 0.0, roof_z(roof_left)), (roof_right, 0.0, roof_z(roof_right)),
                               (roof_right, length, roof_z(roof_right)), (roof_left, length, roof_z(roof_left))]]
            ax.add_collection3d(Poly3DCollection(roof_faces, facecolors="#c6a15b", edgecolors="#b58b43",
                                                  linewidths=0.35, alpha=0.20))
            for face in roof_faces:
                all_points.extend(face)
    except (TypeError, ValueError):
        # A missing roof surface should not prevent the generated members from exporting.
        pass

    for member in members:
        start, end = member.get("start"), member.get("end")
        if not (isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) == len(end) == 3):
            continue
        try:
            p1 = tuple(float(v) for v in start)
            p2 = tuple(float(v) for v in end)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (*p1, *p2)):
            continue
        kind = str(member.get("member_type", "member")).casefold()
        if "column" in kind:
            color = palette["column"]
        elif "beam" in kind:
            color = palette["beam"]
        elif "purlin" in kind:
            color = palette["purlin"]
        elif "web" in kind or "king" in kind or "vertical" in kind:
            color = palette["web"]
        elif "roof" in kind:
            color = palette["roof"]
        else:
            color = palette["chord"]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=0.85, alpha=0.9)
        all_points.extend([p1, p2])
    if not all_points:
        plt.close(fig)
        raise ValueError("Generated geometry contains no usable 3D member endpoints for PDF export.")
    xs, ys, zs = zip(*all_points)
    mins = [min(xs), min(ys), min(zs)]
    maxs = [max(xs), max(ys), max(zs)]
    spans = [max(1.0, maxs[i] - mins[i]) for i in range(3)]
    ax.set_xlim(mins[0] - spans[0] * .04, maxs[0] + spans[0] * .04)
    ax.set_ylim(mins[1] - spans[1] * .04, maxs[1] + spans[1] * .04)
    ax.set_zlim(mins[2] - spans[2] * .04, maxs[2] + spans[2] * .04)
    ax.set_xlabel("Width / span (ft)", labelpad=7, fontsize=8)
    ax.set_ylabel("Frame depth (ft)" if individual_truss else "Building length (ft)", labelpad=7, fontsize=8)
    ax.set_zlabel("Height (ft)", labelpad=5, fontsize=8)
    title = "Complete Building — 3D Structural Model"
    if individual_truss:
        title = (
            "Individual Truss — 3D Member Model\n"
            f"Span {_fmt_ft_in(geometry.get('span_ft'))} · "
            f"Rise {_fmt_ft_in(geometry.get('rise_height_ft'))}"
        )
    ax.set_title(title, fontsize=13, pad=14)
    ax.view_init(elev=25, azim=-55)
    ax.tick_params(labelsize=7, pad=0)
    ax.grid(True, alpha=.25)
    ax.set_box_aspect(spans)
    fig.tight_layout()
    output = BytesIO()
    fig.savefig(output, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output.seek(0)
    return output


def _pipe_rows(bom):
    """Return readable purchase-quantity and weight tables from the same BOM."""
    quantity_rows = [[
        "Pipe size", "Structural use", "Pieces", "Total length (ft)",
        "Stock length (ft)", "Required pipes", "Extra pipes", "Final pipes to buy",
    ]]
    weight_rows = [[
        "Pipe size", "Weight / stock bar (kg)", "Net member weight (kg)",
        "Base purchase weight (kg)", "Reserve weight (kg)",
        "Final purchase weight (kg)",
    ]]
    sections = bom.get("pipe_sections", [])
    items = [x for x in bom.get("items", []) if x.get("category") == "Structural pipe"]
    counts = Counter()
    types = {}
    for item in items:
        section_name = str(item.get("pipe_section", "N/A"))
        counts[section_name] += int(item.get("count", 0) or 0)
        types.setdefault(section_name, set()).add(str(item.get("member_type", "Member")))
    total_pieces = 0
    all_weights_known = bool(sections)
    for section in sections:
        name = str(section.get("pipe_section", "N/A"))
        pieces = counts.get(name, 0)
        total_pieces += pieces
        required = int(section.get("pipes_required", section.get("base_stock_bars", 0)) or 0)
        extra = int(section.get("extra_pipes", section.get("reserve_bars", 0)) or 0)
        final = int(section.get("final_pipes_to_buy", section.get("purchase_bars", section.get("purchase_stock_bars", 0))) or 0)
        unit = section.get("weight_per_stock_bar_kg", section.get("unit_weight_kg"))
        if any(section.get(field) is None for field in (
            "net_member_weight_kg", "base_purchased_material_weight_kg",
            "reserve_weight_kg", "purchased_material_weight_kg", "weight_per_stock_bar_kg",
        )):
            all_weights_known = False
        used_for = ", ".join(sorted(types.get(name, set()))) or "Not assigned"
        quantity_rows.append([
            name,
            used_for,
            str(pieces),
            _fmt(section.get("net_member_length_ft"), 3),
            _fmt(section.get("stock_length_ft")),
            str(required),
            str(extra),
            str(final),
        ])
        weight_rows.append([
            name,
            _fmt(unit),
            _fmt(section.get("net_member_weight_kg")),
            _fmt(section.get("base_purchased_material_weight_kg", section.get("base_stock_bars", 0) * unit if unit else None)),
            _fmt(section.get("reserve_weight_kg")),
            _fmt(section.get("purchased_material_weight_kg", final * unit if unit else None)),
        ])
    quantity_rows.append([
        "TOTAL", "All structural members", str(total_pieces),
        _fmt(bom.get("total_length_ft"), 3),
        "—",
        str(sum(int(section.get("pipes_required", section.get("base_stock_bars", 0)) or 0) for section in sections)),
        str(sum(int(section.get("extra_pipes", section.get("reserve_bars", 0)) or 0) for section in sections)),
        str(sum(int(section.get("final_pipes_to_buy", section.get("purchase_bars", section.get("purchase_stock_bars", 0))) or 0) for section in sections)),
    ])
    weight_rows.append([
        "TOTAL", "—",
        _fmt(bom.get("material_weight_kg")),
        _fmt(bom.get("base_purchased_material_weight_kg")) if all_weights_known else "Unavailable",
        _fmt(bom.get("reserve_weight_kg")) if all_weights_known else "Unavailable",
        _fmt(bom.get("purchased_material_weight_kg")) if all_weights_known else "Unavailable",
    ])
    return quantity_rows, weight_rows


def _member_group_rows(bom):
    rows = [["Member type", "Pipe size", "Physical pieces", "Net length (ft)", "Net member weight (kg)"]]
    for group in bom.get("member_group_totals", []):
        rows.append([
            str(group.get("member_type", "Member")),
            str(group.get("pipe_section", "N/A")),
            str(group.get("count", 0)),
            _fmt(group.get("net_length_ft"), 3),
            _fmt(group.get("net_weight_kg")),
        ])
    rows.append([
        "TOTAL", "All generated members", str(sum(int(group.get("count", 0) or 0) for group in bom.get("member_group_totals", []))),
        _fmt(bom.get("total_length_ft"), 3),
        _fmt(bom.get("material_weight_kg")),
    ])
    return rows


def _roof_rows(bom):
    rows = [["Material", "Length / specification", "Quantity to purchase", "Notes"]]
    takeoff = bom.get("roofing_material_takeoff", {}) or {}
    if takeoff.get("status") != "calculated":
        rows.append(["Roofing", "—", "Not calculated", str(takeoff.get("warning", "Generated roof data unavailable."))])
        return rows
    if "total_sheet_piece_count" in takeoff:
        counts = Counter()
        for slope in takeoff.get("slope_details", []):
            lengths = slope.get("purchase_sheet_lengths_ft_per_width_position", [])
            across = int(slope.get("sheets_across_building_length", 0) or 0)
            for length in lengths:
                counts[float(length)] += across
        for length in sorted(counts):
            rows.append(["Roofing sheet", f"{length:g} ft stock length", str(counts[length]), "Purchase length; finished cut lengths may differ"])
        rows.append(["TOTAL roofing sheets", "All generated roof slopes", str(takeoff.get("total_sheet_piece_count", sum(counts.values()))), "Total sheet pieces to purchase"])
        rows.append(["Roofing screws — base", "3 per sheet per generated purlin row", str(takeoff.get("base_screw_count", 0)), "Required quantity"])
        rows.append([f"Roofing screws — {takeoff.get('screw_reserve_pct', 20)}% reserve", "Rounded up to a whole screw", str(takeoff.get("screw_reserve_count", 0)), f"Reserve rate: {takeoff.get('screw_reserve_pct', 20)}%"])
        rows.append(["Roofing screws — final purchase", "Base + reserve", str(takeoff.get("screw_count_to_purchase", 0)), "Final purchase quantity"])
    elif "tile_count_to_purchase" in takeoff:
        tile_style = ParagraphStyle(
            name="RoofTileDetail",
            parent=getSampleStyleSheet()["BodyText"],
            fontSize=7,
            leading=8,
        )

        def tile_cell(value):
            return Paragraph(escape(str(value)), tile_style)

        rows.append([
            tile_cell("Tile length"),
            tile_cell(f"{takeoff.get('tile_length_in', takeoff.get('tile_pitch_in', 9))} in along purlin direction"),
            "",
            "Configured tile coverage length",
        ])
        rows.append([
            tile_cell("Generated purlin length"),
            tile_cell(f"{_fmt(takeoff.get('purlin_length_ft'))} ft from generated longitudinal purlin runs"),
            "",
            "Read from generated purlin members",
        ])
        for slope in takeoff.get("slope_details", []):
            detail = (
                f"{_fmt(slope.get('purlin_length_ft', takeoff.get('purlin_length_ft')))} ft × 12 / "
                f"{_fmt(slope.get('tile_length_in', takeoff.get('tile_length_in', 9)), 0)} in = "
                f"{slope.get('tiles_per_row', 0)} tiles/row; "
                f"{slope.get('generated_purlin_rows', 0)} purlin rows = "
                f"{slope.get('tile_box_rows', 0)} tile-box rows"
            )
            rows.append([
                tile_cell(f"{str(slope.get('slope', 'slope')).title()} slope — base tiles"),
                tile_cell(detail),
                str(slope.get("base_tiles", 0)),
                "Base quantity for this slope",
            ])
        rows.append([tile_cell("Roofing tiles — combined base"), "Sum of both roof slopes", str(takeoff.get("base_tile_count", 0)), "Base quantity"])
        rows.append([tile_cell(f"Roofing tiles — {takeoff.get('reserve_pct', 15)}% reserve"), "Rounded up once on combined base", str(takeoff.get("reserve_tile_count", 0)), f"Reserve rate: {takeoff.get('reserve_pct', 15)}%"])
        rows.append([tile_cell("Roofing tiles — final purchase"), "Combined base + reserve", str(takeoff.get("tile_count_to_purchase", 0)), "Final purchase quantity"])
    else:
        rows.append(["Roofing", str(takeoff.get("roofing_type", "N/A")), "Not configured", str(takeoff.get("warning", ""))])
    return rows


def build_design_bom_pdf(state, selected, geometry, bom):
    """Return a PDF based on the selected generated geometry and calculated BOM."""
    pdf_buffer = BytesIO()
    page = landscape(A4)
    doc = SimpleDocTemplate(pdf_buffer, pagesize=page, rightMargin=12*mm, leftMargin=12*mm,
                            topMargin=10*mm, bottomMargin=10*mm, title="TRUSS Design and Bill of Materials")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22, spaceAfter=5))
    styles.add(ParagraphStyle(name="SectionHeading", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontSize=7.5, leading=9, textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name="SummaryLabel", parent=styles["BodyText"], fontSize=8, leading=10, fontName="Helvetica-Bold", wordWrap="CJK"))
    styles.add(ParagraphStyle(name="SummaryValue", parent=styles["BodyText"], fontSize=8, leading=10, wordWrap="CJK"))
    story = []
    story.append(Paragraph("TRUSS AI DESIGN STUDIO", styles["ReportTitle"]))
    story.append(Paragraph("Generated Design & Bill of Materials", styles["Heading2"]))
    inputs = state.get("inputs", {})
    design = selected.get("design", state.get("design", {}))
    purlin_layout = geometry.get("purlin_layout", {}) or {}
    purlin_rows_text = (
        f"Left: {purlin_layout.get('left_slope_row_count', 0)}; "
        f"right: {purlin_layout.get('right_slope_row_count', 0)}; "
        f"single: {purlin_layout.get('single_slope_row_count', 0)}"
    )
    resolved = geometry.get("resolved_parameters", {})
    members = geometry.get("members", [])
    column_count = sum(str(member.get("member_type", "")).casefold() == "column" for member in members)
    beam_segment_count = sum(str(member.get("member_type", "")).casefold() == "beam" for member in members)
    beam_run_count = int(resolved.get("beam_run_count", resolved.get("beam_longitudinal_line_count", 0)) or 0)
    truss_count = int(resolved.get("truss_total_count", len(geometry.get("truss_positions_ft", []))) or 0)
    base_pipe_weight = bom.get("base_purchased_material_weight_kg")
    reserve_pipe_weight = bom.get("reserve_weight_kg")
    summary = [
        ["Site dimensions", f"{_fmt(inputs.get('site_width_ft'))} ft × {_fmt(inputs.get('site_length_ft'))} ft", "Selected family", str(selected.get("truss_family", design.get("truss_family", "N/A")))],
        ["Roof configuration", str(geometry.get("roof_configuration", "N/A")).replace("_", " ").title(), "Roofing type", str(inputs.get("roofing_type", "N/A"))],
        ["Generated trusses", str(truss_count), "Physical columns", str(column_count)],
        ["Beam runs", str(beam_run_count), "Physical beam segments", str(beam_segment_count)],
        ["Generated structural pieces", str(bom.get("member_count", len(members))), "Support layout", str(selected.get("support_configuration", "N/A"))],
        ["Net member steel", f"{_fmt(bom.get('material_weight_kg'))} kg", "Base pipe purchase", f"{_fmt(base_pipe_weight)} kg"],
        ["10% reserve weight", f"{_fmt(reserve_pipe_weight)} kg", "Final purchased pipe weight", f"{_fmt(bom.get('purchased_material_weight_kg'))} kg"],
        ["Slope length", f"{_fmt(purlin_layout.get('sloped_roof_length_ft', geometry.get('roof_slope_length_ft')))} ft", "Purlin rows / physical segments", f"{bom.get('purlin_rows', 0)} / {bom.get('purlin_physical_segment_count', 0)}"],
        ["Net purlin length", f"{_fmt(bom.get('purlin_net_length_ft'))} ft", "Rows by roof slope", purlin_rows_text],
    ]
    summary_cells = [
        [
            Paragraph(escape(str(value)), styles["SummaryLabel"] if column in (0, 2) else styles["SummaryValue"])
            for column, value in enumerate(row)
        ]
        for row in summary
    ]
    summary_col_widths = [42*mm, 78*mm, 54*mm, doc.width - 174*mm]
    summary_table = Table(summary_cells, colWidths=summary_col_widths)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eaf0f3")),
        ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#eaf0f3")),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#c8d0d5")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    member_group_table = Table(
        _member_group_rows(bom), repeatRows=1,
        colWidths=[46*mm, 42*mm, 38*mm, 48*mm, 55*mm],
    )
    member_group_table.setStyle(_table_style())
    story.extend([
        summary_table, Spacer(1, 3*mm),
        Paragraph(
            "Geometry and BOM quantities are estimates; this report does not establish structural capacity.",
            styles["SmallNote"],
        ),
        PageBreak(),
        Paragraph("View 1 — Individual truss: dimensioned 2D elevation", styles["SectionHeading"]),
    ])
    cad_stream = _make_cad_front_image(geometry)
    story.extend([
        Paragraph(
            f"Family: {geometry.get('truss_family', 'N/A')} · "
            f"Chord: {resolved.get('frame_chord_pipe_size', 'N/A')} · "
            f"Web: {resolved.get('web_pipe_size', 'Not applicable')}",
            styles["SmallNote"],
        ),
        Image(cad_stream, width=270*mm, height=155*mm, kind="proportional"),
        PageBreak(),
        Paragraph("View 2 — Individual truss: 3D member model", styles["SectionHeading"]),
        Image(_make_3d_image(geometry, individual_truss=True), width=270*mm, height=165*mm, kind="proportional"),
        PageBreak(),
        Paragraph("View 3 — Complete building: 3D structural model", styles["SectionHeading"]),
        Image(_make_3d_image(geometry), width=270*mm, height=165*mm, kind="proportional"),
        PageBreak(),
        Paragraph("Structural pipe bill of materials", styles["SectionHeading"]),
        Paragraph("Generated member groups", styles["Heading3"]),
        member_group_table,
    ])
    quantity_rows, weight_rows = _pipe_rows(bom)
    quantity_table = Table(
        quantity_rows, repeatRows=1,
        colWidths=[22*mm, 42*mm, 18*mm, 25*mm, 23*mm, 29*mm, 24*mm, 25*mm, 25*mm],
    )
    quantity_table.setStyle(_table_style())
    quantity_table.setStyle(TableStyle([("FONTSIZE", (0,0), (-1,-1), 7), ("LEADING", (0,0), (-1,-1), 8)]))
    weight_table = Table(
        weight_rows, repeatRows=1,
        colWidths=[30*mm, 35*mm, 45*mm, 45*mm, 40*mm, 43*mm],
    )
    weight_table.setStyle(_table_style())
    weight_table.setStyle(TableStyle([("FONTSIZE", (0,0), (-1,-1), 7), ("LEADING", (0,0), (-1,-1), 8)]))
    story.extend([
        Paragraph("Stock quantities and feasible cuts", styles["Heading3"]),
        quantity_table,
        Spacer(1, 3*mm),
        Paragraph("Member and purchase weights", styles["Heading3"]),
        weight_table,
    ])
    story.extend([PageBreak(), Paragraph("Roofing and fastener quantities", styles["SectionHeading"])])
    roof_table = Table(_roof_rows(bom), repeatRows=1, colWidths=[40*mm, 66*mm, 35*mm, 92*mm])
    roof_table.setStyle(_table_style())
    story.append(roof_table)
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Material quantities are estimates based on the generated geometry and configured catalogue. "
        "Confirm member connections, splice details, sheet fixing, tile layout, and structural capacity with a qualified professional before fabrication or purchase.",
        styles["SmallNote"]
    ))
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def _table_style():
    return TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#263746")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 7.5),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#b8c2ca")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f5f7")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ])
