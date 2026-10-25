import logging
import json
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    raise SystemExit(subprocess.call([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
    ]))

from config import MODEL_DIR, PROJECT_ROOT
from calculator import BOM_CALCULATION_VERSION, calculate_bom
from pdf_export import build_design_bom_pdf
from design import (
    compatible_truss_families,
    draw_building_figure,
    draw_truss_figure,
    generate_design,
)
from design_options import evaluate_design_options
from result_history import make_cache_key, load_exact, save_success, similar_history

logger = logging.getLogger(__name__)

st.set_page_config(page_title="TRUSS AI Design Studio", page_icon="🏗️", layout="wide")

# Preserve each browser URL's latest design across a hard refresh. A unique token
# in the URL keeps separate browser sessions from overwriting each other's state.
SESSION_DIR = PROJECT_ROOT / "results" / "browser_sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)
try:
    browser_session_id = str(st.query_params.get("truss_session", ""))
except Exception:
    browser_session_id = ""
if not browser_session_id or not all(c in "0123456789abcdef-" for c in browser_session_id.lower()):
    browser_session_id = str(uuid.uuid4())
    try:
        st.query_params["truss_session"] = browser_session_id
    except Exception:
        pass
SESSION_FILE = SESSION_DIR / f"{browser_session_id}.json"

def _json_default(value):
    """Convert NumPy-like values without turning numbers into strings."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)

def _save_browser_state():
    """Atomically save this browser session's last result for refresh recovery."""
    payload = {
        "active_design": st.session_state.get("active_design"),
        "last_attempt": st.session_state.get("last_attempt"),
    }
    temp_file = SESSION_FILE.with_suffix(".tmp")
    temp_file.write_text(json.dumps(payload, ensure_ascii=False, default=_json_default), encoding="utf-8")
    temp_file.replace(SESSION_FILE)

def _restore_browser_state():
    if st.session_state.get("_truss_state_restored"):
        return
    st.session_state["_truss_state_restored"] = True
    if not SESSION_FILE.exists():
        return
    try:
        payload = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        if payload.get("active_design") is not None:
            st.session_state.setdefault("active_design", payload["active_design"])
        if payload.get("last_attempt") is not None:
            st.session_state.setdefault("last_attempt", payload["last_attempt"])
    except (OSError, json.JSONDecodeError, TypeError):
        logger.exception("Could not restore the last design for this browser session")

def _is_prediction_diagnostic_warning(message):
    text = str(message).casefold()
    return (
        text.startswith("predicted ")
        or text.startswith("model-predicted ")
        or text.startswith("model purlin spacing ")
        or text.startswith("metal-sheet actual clear gap is ")
    )

_restore_browser_state()
st.title("TRUSS AI Design Studio")

try:
    from model import get_model_status, predict_design
except ImportError as exc:
    logger.exception("Required model inference module could not be imported")
    st.error(f"Required model pipeline module is unavailable: {exc}")
    st.stop()

status = get_model_status(MODEL_DIR)
if not status["directory_exists"] or "metadata" not in status:
    logger.error("Saved model catalog unavailable: %s", status.get("message", "unknown error"))

pipe_path = PROJECT_ROOT / "pipes.csv"
try:
    pipe_sections = pd.read_csv(pipe_path)["pipe_size"].dropna().astype(str).tolist()
except (OSError, KeyError, pd.errors.ParserError):
    logger.exception("Pipe database could not be loaded")
    pipe_sections = []

with st.sidebar:
    st.subheader("Status")
    st.write(f"ML models: {'Ready' if status.get('metadata') else 'Unavailable'}")
    st.write(f"Pipe catalogue: {'Ready' if pipe_sections else 'Unavailable'}")
    if not status.get("metadata"):
        st.error("Model files are unavailable. Check your local models folder and application log.")
    with st.expander("System diagnostics", expanded=False):
        st.caption(f"Project root: {PROJECT_ROOT}")
        st.caption(f"Model directory: {MODEL_DIR}")
        st.caption(f"Saved model artifacts: {status.get('artifact_count', 0)}")
    st.caption("Design outputs require qualified structural review.")

def _on_roof_compatibility_input_change():
    roof_label = st.session_state.get("roof_selection", "Two sides / gable")
    options = compatible_truss_families({
        "roof_sides": 2 if roof_label == "Two sides / gable" else 1,
        "roof_configuration_detail": "Gable" if roof_label == "Two sides / gable" else "Single Slope",
    })
    current_family = st.session_state.get("manual_family")
    if current_family not in options:
        st.session_state["manual_family"] = "Select a compatible family"


st.subheader("Design mode")
mode = st.radio(
    "Choose design mode",
    ["AUTO — ML-generated design", "MANUAL — User-configured design"],
    horizontal=True,
    key="design_mode",
    label_visibility="collapsed",
)
manual = mode.startswith("MANUAL")

roofing_columns = st.columns(3)
with roofing_columns[0]:
    roofing = st.selectbox(
        "Roofing type",
        ["Metal Sheet", "Clay / Concrete Tile"],
        key="roofing_type",
        on_change=_on_roof_compatibility_input_change,
    )
with roofing_columns[1]:
    roof_label = st.selectbox(
        "Roof sides",
        ["Two sides / gable", "Single slope"],
        key="roof_selection",
        on_change=_on_roof_compatibility_input_change,
    )
family_inputs = {
    "roof_sides": 2 if roof_label == "Two sides / gable" else 1,
    "roof_configuration_detail": "Gable" if roof_label == "Two sides / gable" else "Single Slope",
}
family_options = compatible_truss_families(family_inputs)
manual_family = None
if manual:
    with roofing_columns[2]:
        if family_options:
            family_placeholder = "Select a compatible family"
            if st.session_state.get("manual_family") not in family_options:
                st.session_state["manual_family"] = family_placeholder
            selected_family = st.selectbox(
                "Truss family",
                [family_placeholder, *family_options],
                key="manual_family",
            )
            manual_family = (
                selected_family if selected_family in family_options else None
            )
            st.caption("You choose the truss family. The app automatically tests unsupported, middle-support, and multi-support layouts; it uses another family only if the selected family has no valid option.")
        else:
            st.error("No truss families are available for the selected roof configuration.")
else:
    roofing_columns[2].info("Truss family and design parameters are predicted by the trained pipeline.")

with st.form("design_form", clear_on_submit=False):
    left, middle = st.columns(2)
    with left:
        width = st.number_input("Building width / horizontal span (ft)", min_value=1.0, value=20.0, step=1.0)
        length = st.number_input("Building length (ft)", min_value=1.0, value=40.0, step=1.0)
    with middle:
        # Support configuration is chosen automatically by the candidate
        # evaluator in both modes; there is intentionally no manual toggle.
        center = "No"
        st.caption("Support layout is selected automatically: valid unsupported designs are preferred, with supported alternatives used when needed.")
        column_height = st.number_input(
            "Column height (ft)", min_value=1.0, max_value=18.0, value=8.0, step=0.5,
            help="Maximum permitted column height is 18 ft."
        )
        overhang = st.number_input(
            "Eave overhang per end (ft)", min_value=0.0, max_value=2.0, value=0.0, step=0.5,
            help="Building width is the column-to-column span. The truss/roof extends by this amount beyond each end; column lines remain at the entered span."
        )
        st.caption("Overhang extends the roof/truss only; column lines remain at the entered building width.")
        gable_end_overhang = 0.0
        if roof_label == "Two sides / gable":
            st.caption(
                "Gable-end overhang is unavailable because the reference data does not define its framing."
            )
    submitted = st.form_submit_button(
        "Generate design",
        type="primary",
        width="stretch",
        disabled=not family_options,
    )

inputs = {
    "site_width_ft": width,
    "site_length_ft": length,
    "roofing_type": roofing,
    "roof_sides": 2 if roof_label == "Two sides / gable" else 1,
    "roof_configuration_detail": "Gable" if roof_label == "Two sides / gable" else "Single Slope",
    "center_support_present": center,
    "column_height_B_ft": column_height,
    "overhang_ft": overhang,
    "gable_end_overhang_ft": gable_end_overhang,
}

def _restore_exact_cached_submission():
    """Reuse only a previously validated result with exactly matching inputs/models/data."""
    mode_name = "MANUAL" if manual else "AUTO"
    requested = manual_family if manual else None
    cache_key, signature = make_cache_key(inputs, mode_name, requested, PROJECT_ROOT, MODEL_DIR)
    st.session_state["_pending_result_cache_key"] = cache_key
    st.session_state["_pending_result_signature"] = signature
    cached = load_exact(cache_key, PROJECT_ROOT)
    if cached and cached.get("candidates"):
        st.session_state.pop("candidate_selector", None)
        st.session_state["active_design"] = cached
        st.session_state["last_attempt"] = {
            "inputs": cached.get("inputs", inputs),
            "mode": cached.get("mode", mode_name),
            "requested_family": cached.get("requested_family", requested),
            "candidate_count": len(cached.get("candidates", [])),
            "rejections": cached.get("rejections", []),
            "accepted_count": len(cached.get("candidates", [])),
            "generation_error": None,
        }
        st.session_state["_result_cache_hit"] = True
        st.session_state["_history_suggestions"] = []
        _save_browser_state()
        st.toast("Loaded an exactly matching validated design from history.", icon="⚡")
        return True
    st.session_state["_result_cache_hit"] = False
    st.session_state["_history_suggestions"] = similar_history(
        inputs, mode_name, requested, PROJECT_ROOT, limit=3
    )
    return False


if submitted and not _restore_exact_cached_submission():
    try:
        with st.status("Checking model/data signatures and design history…", expanded=True) as progress:
            primary = predict_design(inputs, MODEL_DIR, manual_family if manual else None)
            raw_values = primary.get("design", {})
            requested_family = manual_family if manual else raw_values.get("truss_family")
            prediction_cache = {}

            def _predict_candidate(candidate_inputs, family):
                # A different unsupported bay or support-line layout needs fresh inference.
                has_middle_support = candidate_inputs.get("center_support_present") == "Yes"
                unsupported_span = candidate_inputs.get(
                    "unsupported_span_ft", candidate_inputs.get("site_width_ft", inputs["site_width_ft"])
                )
                positions = candidate_inputs.get("support_line_positions_ft", [])
                key = (
                    str(family), has_middle_support, round(float(unsupported_span), 3),
                    tuple(round(float(position), 3) for position in positions),
                    str(candidate_inputs.get("support_strategy", "unsupported")),
                )
                # Reuse the already-computed primary prediction for the exact same
                # unsupported request; supported bay layouts still get independent inference.
                if (
                    family == requested_family
                    and not has_middle_support
                    and str(candidate_inputs.get("support_strategy", "unsupported")) == "unsupported"
                    and abs(float(unsupported_span) - float(inputs["site_width_ft"])) < 1e-6
                    and len(positions) == 2
                    and abs(float(positions[0])) < 1e-6
                    and abs(float(positions[1]) - float(inputs["site_width_ft"])) < 1e-6
                ):
                    return primary
                if key not in prediction_cache:
                    prediction_cache[key] = predict_design(candidate_inputs, MODEL_DIR, family)
                return prediction_cache[key]

            # AUTO compares compatible families. MANUAL prioritizes the selected family;
            # support layouts are always resolved automatically, and family fallback is
            # allowed only if the selected family produces no validated candidate.
            candidates, rejections = evaluate_design_options(
                inputs,
                family_options,
                _predict_candidate,
                "MANUAL" if manual else "AUTO",
                requested_family,
            )
            if not candidates and not rejections:
                rejections = [{
                    "truss_family": requested_family or "Unresolved",
                    "support_configuration": "not evaluated",
                    "failure_category": "Model inference",
                    "reason": primary.get("message", "No compatible family produced a valid design."),
                }]

            selected_candidate = candidates[0] if candidates else None
            displayed_design = selected_candidate.get("design", raw_values) if selected_candidate else raw_values
            sources = {key: "ML prediction / resolved candidate" for key in displayed_design}
            if manual and manual_family:
                sources["truss_family"] = (
                    "User-selected family"
                    if selected_candidate and selected_candidate.get("truss_family") == manual_family
                    else "Fallback family (selected family had no valid candidate)"
                )
            state = {
                "mode": "MANUAL" if manual else "AUTO",
                "inputs": inputs,
                "design": displayed_design,
                "requested_family": requested_family,
                "sources": sources,
                "prediction_status": "Succeeded" if candidates or primary.get("connected") else "Failed",
                "generation_status": "Generated" if candidates else "Rejected",
                "candidates": candidates,
                "rejections": rejections,
                "candidate_count": len(candidates),
                "generation_error": None if candidates else (
                    rejections[0]["reason"] if rejections
                    else "The selected family did not produce a valid geometry and BOM."
                ),
            }
            st.session_state["last_attempt"] = {
                "inputs": inputs,
                "mode": state["mode"],
                "requested_family": requested_family,
                "candidate_count": len(candidates),
                "rejections": rejections,
                "accepted_count": len(candidates),
                "generation_error": state["generation_error"],
            }
            if candidates:
                # Reset the selector after each successful submission so the first
                # ranked option is displayed instead of a stale selection.
                st.session_state.pop("candidate_selector", None)
                st.session_state["active_design"] = state
                try:
                    save_success(
                        st.session_state.get("_pending_result_cache_key", ""),
                        st.session_state.get("_pending_result_signature", ""),
                        state,
                        PROJECT_ROOT,
                    )
                except Exception:
                    # History is an optimization, never a dependency for generating a design.
                    logger.exception("Could not save validated result to history")
            progress.update(label="Design generation finished", state="complete", expanded=False)
        _save_browser_state()
    except Exception as exc:
        logger.exception("Design request failed")
        st.session_state["last_attempt"] = {
            "inputs": inputs,
            "mode": "MANUAL" if manual else "AUTO",
            "requested_family": manual_family if manual else None,
            "candidate_count": 0,
            "rejections": [{
                "truss_family": manual_family if manual else "AUTO",
                "support_configuration": "not completed",
                "failure_category": type(exc).__name__,
                "reason": str(exc),
            }],
            "accepted_count": 0,
            "generation_error": f"Request failed safely: {exc}",
        }
        _save_browser_state()
        st.error(f"Design generation failed: {exc}")
        st.caption("The app is still available. Check the terminal traceback, correct the cause, and submit again.")

state = st.session_state.get("active_design")
last_attempt = st.session_state.get("last_attempt")

history_suggestions = st.session_state.get("_history_suggestions", [])
if history_suggestions and submitted and not st.session_state.get("_result_cache_hit", False):
    with st.expander("Similar previously validated designs (suggestions only)", expanded=False):
        st.caption(
            "Historical examples can help you compare faster. They are not used as the current design; "
            "the current request is still generated and validated using the current model and reference data."
        )
        suggestion_rows = [{
            "Past width (ft)": item.get("site_width_ft"),
            "Past length (ft)": item.get("site_length_ft"),
            "Family": item.get("selected_family"),
            "Support layout": item.get("support_configuration"),
            "Max bay (ft)": item.get("max_unsupported_bay_ft"),
            "Members": item.get("member_count"),
            "Weight (kg)": item.get("material_weight_kg"),
            "Input distance": item.get("similarity_distance"),
            "Generated (UTC)": item.get("created_at_utc"),
        } for item in history_suggestions]
        st.dataframe(pd.DataFrame(suggestion_rows), width="stretch", hide_index=True)
if last_attempt and last_attempt.get("generation_error"):
    family_label = last_attempt.get("requested_family") or "unresolved family"
    st.error(
        f"Latest submitted request for {family_label} failed: "
        f"{last_attempt['generation_error']}"
    )
    if st.session_state.get("active_design"):
        st.info("The geometry below is the previous successful design, not the failed request.")
    if last_attempt.get("rejections"):
        with st.expander("Latest request failure details", expanded=True):
            st.dataframe(pd.DataFrame(last_attempt["rejections"]), width="stretch", hide_index=True)

if state is None:
    if not last_attempt:
        st.info("Choose AUTO or MANUAL inputs and submit to generate a design.")
else:
    active_is_current = (
        state["inputs"] == inputs
        and state["mode"] == ("MANUAL" if manual else "AUTO")
        and (
            not manual
            or state.get("requested_family") == manual_family
        )
    )
    if not active_is_current:
        st.info("Showing the previous successful design. Generate a new design to apply the current inputs.")

    candidates = state["candidates"]
    if candidates:
        requested_family = state.get("requested_family")
        selected_family_name = candidates[0].get("truss_family")
        if (
            state.get("mode") == "MANUAL"
            and requested_family
            and selected_family_name != requested_family
        ):
            st.warning(
                f"{requested_family} could not produce a fully validated design for these inputs. "
                f"Showing the nearest available alternative that passed geometry and BOM checks: "
                f"{selected_family_name}. Rejected candidates are not shown as valid designs."
            )
        st.subheader("Validated design")
        st.caption(
            f"{len(candidates)} validated option(s) · support layout resolved automatically · "
            "geometry and material quantities are estimates and require engineering review."
        )
    selected_index = 0
    if len(candidates) > 1:
        candidate_labels = [
            f"{c['truss_family']} · {c.get('support_configuration', 'support layout')} · "
            f"{c['unsupported_span_ft']:.1f} ft max bay · {c['bom']['member_count']} members"
            for c in candidates
        ]
        selected_label = st.selectbox(
            f"Select design to view ({len(candidates)} valid alternatives)",
            candidate_labels,
            key="candidate_selector",
        )
        selected_index = candidate_labels.index(selected_label)
    selected = candidates[selected_index] if candidates else None
    if selected and selected.get("bom", {}).get("calculation_version") != BOM_CALCULATION_VERSION:
        selected = {
            **selected,
            "bom": calculate_bom(selected.get("geometry", {})),
        }
        candidates = list(candidates)
        candidates[selected_index] = selected
        state = {**state, "candidates": candidates}
        st.session_state["active_design"] = state
        _save_browser_state()
    if selected:
        mode_label = "AUTO" if state["mode"] == "AUTO" else "MANUAL"
        st.subheader(f"Design: {selected['truss_family']} · {mode_label}")
        st.caption(
            "Geometry and BOM checks do not establish structural capacity; engineering review is required."
        )
        geometry = selected["geometry"]
        bom = selected["bom"]
        state = {
            **state,
            "inputs": selected["inputs"],
            "design": selected["design"],
            "geometry": geometry,
            "bom": bom,
        }
    else:
        geometry = None
        bom = {"connected": False, "message": state.get("generation_error", "No valid candidate BOM is available.")}
    st.subheader("Design summary")
    cols = st.columns(4)
    cols[0].metric("Mode", state["mode"])
    cols[1].metric("Prediction", state["prediction_status"])
    cols[2].metric("Geometry", state["generation_status"])
    cols[3].metric("BOM", "Calculated" if bom.get("connected") else "Incomplete")
    if state["mode"] == "MANUAL" and selected:
        if selected.get("truss_family") == state.get("requested_family"):
            st.caption("Manual selection honored. The support layout was selected automatically from validated options.")
        else:
            st.warning(f"No valid design was found for {state.get('requested_family')}. Showing the next validated family: {selected.get('truss_family')}.")
    if state.get("generation_error"):
        st.error(f"Design could not be generated: {state['generation_error']}")
    if geometry:
        st.caption(
            f"{geometry['roof_configuration'].replace('_', ' ').title()} · "
            f"{geometry['truss_family']} · Rise {geometry['rise_height_ft']:.2f} ft · "
            f"Slope {geometry['roof_slope_angle_deg']:.2f}° · {len(geometry['members'])} members"
        )
        purlin_layout = geometry["purlin_layout"]
        st.caption(
            f"Generated purlin rows — left: {purlin_layout['left_slope_row_count']}, "
            f"right: {purlin_layout['right_slope_row_count']}, "
            f"single slope: {purlin_layout['single_slope_row_count']} · "
            f"{bom.get('purlin_physical_segment_count', 0)} physical segments · "
            f"{bom.get('purlin_net_length_ft', 0):.2f} ft net · "
            f"slope length: {purlin_layout['sloped_roof_length_ft']:.2f} ft"
        )
        st.caption(
            f"Column lanes {geometry['column_lanes_ft']} ft across · "
            f"stations {geometry['column_y_positions_ft']} ft along · "
            f"{geometry['final_column_total_count']} columns total "
            f"({geometry['columns_per_support_line']} per support line) · "
            f"section {state['design'].get('column_pipe_size', 'not predicted')}"
        )
        beam_summary = geometry["resolved_parameters"]
        st.caption(
            f"Generated beams: {beam_summary.get('beam_run_count', 0)} logical runs · "
            f"{beam_summary.get('beam_physical_segment_count', 0)} physical segments"
        )
        roofing_takeoff = bom.get("roofing_material_takeoff", {})
        if roofing_takeoff.get("status") != "calculated":
            st.warning(roofing_takeoff.get("warning", "Roofing quantity could not be calculated from the generated design."))
        resolved = geometry["resolved_parameters"]
        parameter_keys = [
            key for key in resolved
            if not key.startswith("predicted_")
            and not key.startswith("parameter_sources")
            and "diagnostic_only" not in key
            and key not in {"support_line_count"}
        ]
        summary_rows = [
            {
                "Parameter": key.replace("_", " ").title(),
                "Resolved geometry": str(resolved[key]),
            }
            for key in parameter_keys
        ]
        with st.expander("Resolved design parameters", expanded=False):
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

        st.subheader("Individual truss geometry")
        st.caption(
            f"{len(geometry['truss_nodes'])} numbered joints · "
            f"Chord section: {state['design'].get('frame_chord_pipe_size', 'user specified')} · "
            f"Web section: {state['design'].get('web_pipe_size', 'Not applicable (Plain Pipe)' if state['design'].get('truss_family') == 'Plain Pipe' else 'user specified')}"
        )
        st.plotly_chart(draw_truss_figure(geometry), width="stretch", key="truss_elevation")
        st.subheader("Building geometry")
        controls_left, controls_middle, controls_right = st.columns(3)
        with controls_left:
            visible_categories = st.multiselect(
                "Visible structure",
                ["trusses", "webs", "columns", "beams", "purlins", "roofing"],
                default=["trusses", "webs", "columns", "beams", "purlins", "roofing"],
            )
        with controls_middle:
            camera = st.selectbox("Camera", ["Isometric", "Front elevation", "End elevation", "Plan"])
        with controls_right:
            wireframe = st.toggle("Wireframe", value=False)
        st.plotly_chart(
            draw_building_figure(geometry, set(visible_categories), camera, wireframe),
            width="stretch",
            key="building_geometry",
        )

        st.subheader("Bill of materials")
        if bom.get("connected"):
            pipe_rows = []
            total_structure_length = 0.0
            total_pipes_required = 0
            total_extra_pipes = 0
            total_final_pipes = 0
            total_purchase_weight = 0.0
            for section in bom.get("pipe_sections", []):
                section_name = str(section.get("pipe_section", "N/A"))
                used_for = ", ".join(sorted(section.get("member_types", []))) if section.get("member_types") else "Not assigned"
                total_length = float(section.get("net_member_length_ft", 0.0) or 0.0)
                stock_length = section.get("stock_length_ft")
                required = int(section.get("pipes_required", 0) or 0)
                extra = int(section.get("extra_pipes", 0) or 0)
                final = int(section.get("final_pipes_to_buy", 0) or 0)
                unit_weight = section.get("unit_weight_kg")
                total_weight = section.get("total_purchase_weight_kg")
                pipe_rows.append({
                    "Pipe size": section_name,
                    "Used for": used_for,
                    "Total length (ft)": total_length,
                    "Stock length (ft)": stock_length,
                    "Pipes required": required,
                    "Extra pipes": extra,
                    "Final pipes to buy": final,
                    "Unit weight (kg/pipe)": unit_weight,
                    "Total purchase weight (kg)": total_weight,
                })
                total_structure_length += total_length
                total_pipes_required += required
                total_extra_pipes += extra
                total_final_pipes += final
                total_purchase_weight += float(total_weight or 0.0)
            pipe_rows.append({
                "Pipe size": "TOTAL",
                "Used for": "All structural members",
                "Total length (ft)": total_structure_length,
                "Stock length (ft)": None,
                "Pipes required": total_pipes_required,
                "Extra pipes": total_extra_pipes,
                "Final pipes to buy": total_final_pipes,
                "Unit weight (kg/pipe)": None,
                "Total purchase weight (kg)": total_purchase_weight,
            })
            st.markdown("**Structural pipes — purchase quantities**")
            st.dataframe(
                pd.DataFrame(pipe_rows),
                width="stretch",
                hide_index=True,
                column_config={
                    "Total length (ft)": st.column_config.NumberColumn(format="%.3f"),
                    "Pipes required": st.column_config.NumberColumn(format="%d"),
                    "Extra pipes": st.column_config.NumberColumn(format="%d"),
                    "Final pipes to buy": st.column_config.NumberColumn(format="%d"),
                    "Unit weight (kg/pipe)": st.column_config.NumberColumn(format="%.3f"),
                    "Total purchase weight (kg)": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            if bom.get("oversized_members"):
                st.warning(
                    "One or more members exceed the available stock length for their pipe size. The simplified total-length calculation does not define a splice design."
                )
            st.caption("Structural pipe reserve: 10% of the required pipes, rounded up separately by pipe size. Roofing sheets, tiles, and screws each use their own reserve rate and are not mixed with the structural pipe reserve.")

            takeoff = bom.get("roofing_material_takeoff", {}) or {}
            roofing_rows = []
            if takeoff.get("status") == "calculated" and "total_sheet_piece_count" in takeoff:
                sheet_counts = {}
                for slope in takeoff.get("slope_details", []):
                    across = int(slope.get("sheets_across_building_length", 0) or 0)
                    for sheet_length in slope.get("purchase_sheet_lengths_ft_per_width_position", []):
                        label = f"{sheet_length:g} ft"
                        sheet_counts[label] = sheet_counts.get(label, 0) + across
                for sheet_length, count in sorted(sheet_counts.items(), key=lambda x: float(x[0].split()[0])):
                    roofing_rows.append({"Material": "Roofing sheet", "Length / size": sheet_length, "Quantity to buy": count})
                roofing_rows.append({"Material": "TOTAL roofing sheets", "Length / size": "All generated roof slopes", "Quantity to buy": takeoff.get("total_sheet_piece_count", sum(sheet_counts.values()))})
                roofing_rows.append({"Material": "Roofing screws — base", "Length / size": "3 per sheet × actual generated purlin rows", "Quantity to buy": takeoff.get("base_screw_count", 0)})
                roofing_rows.append({"Material": f"Roofing screws — {takeoff.get('screw_reserve_pct', 20)}% reserve", "Length / size": "Rounded up to a whole screw", "Quantity to buy": takeoff.get("screw_reserve_count", 0)})
                roofing_rows.append({"Material": "Roofing screws — final purchase", "Length / size": "Base + reserve", "Quantity to buy": takeoff.get("screw_count_to_purchase", 0)})
            elif takeoff.get("status") == "calculated" and "tile_count_to_purchase" in takeoff:
                roofing_rows.append({
                    "Material": "Tile length",
                    "Length / size": f"{takeoff.get('tile_length_in', takeoff.get('tile_pitch_in', 9))} in along purlin direction",
                    "Quantity to buy": None,
                })
                roofing_rows.append({
                    "Material": "Generated purlin length",
                    "Length / size": f"{takeoff.get('purlin_length_ft')} ft from generated longitudinal purlin runs",
                    "Quantity to buy": None,
                })
                for slope in takeoff.get("slope_details", []):
                    roofing_rows.append({
                        "Material": f"{str(slope.get('slope', 'slope')).title()} slope — base tiles",
                        "Length / size": (
                            f"{slope.get('purlin_length_ft', takeoff.get('purlin_length_ft', 0))} ft × 12 / "
                            f"{slope.get('tile_length_in', takeoff.get('tile_length_in', 9))} in = "
                            f"{slope.get('tiles_per_row', 0)} tiles/row; "
                            f"{slope.get('generated_purlin_rows', 0)} purlin rows = "
                            f"{slope.get('tile_box_rows', 0)} tile-box rows"
                        ),
                        "Quantity to buy": slope.get("base_tiles", 0),
                    })
                roofing_rows.append({"Material": "Roofing tiles — combined base", "Length / size": "Sum of both roof slopes", "Quantity to buy": takeoff.get("base_tile_count", 0)})
                roofing_rows.append({"Material": f"Roofing tiles — {takeoff.get('reserve_pct', 15)}% reserve", "Length / size": "Rounded up once on combined base", "Quantity to buy": takeoff.get("reserve_tile_count", 0)})
                roofing_rows.append({"Material": "Roofing tiles — final purchase", "Length / size": "Combined base + reserve", "Quantity to buy": takeoff.get("tile_count_to_purchase", 0)})
                st.caption("Tile boxes are spaces between adjacent physical purlin rows; tile coverage is calculated along generated longitudinal purlin runs.")
            st.markdown("**Roofing materials**")
            if roofing_rows:
                st.dataframe(pd.DataFrame(roofing_rows), width="stretch", hide_index=True)
            else:
                st.warning(takeoff.get("warning", "Roofing quantities are unavailable from the generated design."))

            st.caption(bom.get("message", ""))
        else:
            st.error(f"BOM incomplete: {bom.get('message', 'Material calculation failed.')}" )

        with st.expander("Design validation checks", expanded=False):
            validation = geometry["validation"]
            bom_matches = bom.get("connected") and bom.get("member_count") == len(geometry["members"])
            design_checks = [
                {"Check": "Roof profile matches selection", "Status": geometry["roof_configuration"]},
                {"Check": "Rise used in generated profile", "Status": f"{geometry['rise_height_ft']:.3f} ft"},
                {"Check": "Trusses / end frames", "Status": f"{validation['truss_count']} / {state['design'].get('truss_total_count', validation['truss_count'])}"},
                {"Check": "End supports present", "Status": "Pass" if validation["end_supports_present"] else "Fail"},
                {"Check": "Beam-to-column connectivity", "Status": "Pass" if validation["beam_to_column_connectivity"] else "Fail"},
                {"Check": "Truss bearings align with beams", "Status": "Pass" if validation["truss_bearings_match_beams"] else "Fail"},
                {"Check": "Truss supports meet chord joints", "Status": "Pass" if validation["support_nodes_match_chords"] else "Fail"},
                {"Check": "Gable apex connectivity", "Status": "Pass" if geometry["roof_configuration"] != "gable" or geometry["ridge_top_chord_connections"] == 2 else "Fail"},
                {"Check": "Column count matches geometry", "Status": "Pass" if validation["column_count_matches_coordinates"] else "Fail"},
                {"Check": "Column count matches support-line distribution", "Status": "Pass" if validation["column_count_matches_support_distribution"] else "Fail"},
                {"Check": "Columns follow support elevations", "Status": "Pass" if validation["columns_match_support_elevations"] else "Fail"},
                {"Check": "Overhang within reference limit", "Status": "Pass" if validation["overhang_within_reference_limit"] else "Fail"},
                {"Check": "All members inside generated bounds", "Status": "Pass" if not validation["member_bounds_violations"] else f"Fail ({len(validation['member_bounds_violations'])})"},
                {"Check": "Purlin end coverage", "Status": "Pass" if validation["purlin_end_coverage"] else "Fail"},
                {"Check": "Purlin spacing rule", "Status": f"max {validation['purlin_max_spacing_ft']:.3f} ft / limit {validation['purlin_spacing_rule_ft']:.3f} ft"},
                {"Check": "Roofing purlin spacing within rule", "Status": "Pass" if validation["purlin_spacing_limit_met"] else "Fail"},
                {"Check": "Zero-length members", "Status": str(validation["zero_length_members_excluded"])},
                {"Check": "Duplicate members", "Status": str(validation["duplicate_members_removed"])},
                {"Check": "BOM/member count match", "Status": "Pass" if bom_matches else "Incomplete"},
            ]
            if state["design"].get("truss_family") == "Plain Pipe":
                has_webs = any(member["member_type"].startswith("Web") for member in geometry["members"])
                design_checks.append({"Check": "Plain Pipe has no web members", "Status": "Fail" if has_webs else "Pass"})
            st.dataframe(pd.DataFrame(design_checks), width="stretch", hide_index=True)

        st.subheader("Export final design")
        if bom.get("connected"):
            try:
                pdf_bytes = build_design_bom_pdf(state, selected, geometry, bom)
                st.download_button(
                    "Save / Download Design + BOM PDF",
                    data=pdf_bytes,
                    file_name="TRUSS_Design_and_BOM.pdf",
                    mime="application/pdf",
                    type="primary",
                    width="stretch",
                )
                st.caption("PDF includes a dimensioned truss elevation, individual-truss and complete-building 3D views, plus the purchase-oriented pipe and roofing tables.")
            except Exception as exc:
                logger.exception("Could not create design/BOM PDF")
                st.error(f"PDF export failed: {exc}")
        with st.expander("Optional data exports", expanded=False):
            export_data = {
                "mode": state["mode"], "inputs": state["inputs"], "predictions_or_manual_values": state["design"],
                "value_sources": state["sources"], "geometry": geometry, "bom": bom,
                "selected_family": state.get("requested_family"),
                "request_failures": state["rejections"],
            }
            export_left, export_middle, export_right = st.columns(3)
            with export_left:
                st.download_button("Design JSON", json.dumps(export_data, indent=2, default=str), file_name="truss_design.json", mime="application/json")
            with export_middle:
                member_csv = pd.DataFrame(geometry["members"]).to_csv(index=False)
                st.download_button("Members CSV", member_csv, file_name="truss_members.csv", mime="text/csv")
            with export_right:
                bom_csv = pd.DataFrame(bom.get("items", [])).to_csv(index=False)
                st.download_button("BOM CSV", bom_csv, file_name="truss_bom.csv", mime="text/csv")

    else:
        st.error("No current geometry is available; previous design output has been cleared.")
