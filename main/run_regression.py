from __future__ import annotations
import json, sys, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent
FILES = ROOT / "files"
sys.path.insert(0, str(FILES))
from config import MODEL_DIR, RESULTS_DIR
from model import predict_design, get_model_status
from design import compatible_truss_families, generate_design
from calculator import calculate_bom

CASES = [(10,20),(20,40),(15,30),(30,40)]
ROOFINGS = ["Metal Sheet", "Clay / Concrete Tile"]
def main():
    status = get_model_status(MODEL_DIR)
    if not status.get("metadata"):
        raise SystemExit(f"Real-model regression cannot run: {status.get('message')}. Preserve/install model artifacts at {MODEL_DIR}.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_cases=[]
    for width,length in CASES:
        for roofing in ROOFINGS:
            inputs={"site_width_ft":width,"site_length_ft":length,"column_height_B_ft":8.0,
                    "roofing_type":roofing,"roof_sides":2,"roof_configuration_detail":"Gable",
                    "center_support_present":"No"}
            for roof in ("Gable", "Single Slope"):
                run_inputs=dict(inputs, roof_sides=2 if roof=="Gable" else 1,
                                roof_configuration_detail=roof)
                families=compatible_truss_families(run_inputs)
                candidates=[]
                for family in families:
                    pred=predict_design(run_inputs,MODEL_DIR,family)
                    row={"truss_family":family,"prediction":pred,"validation":{"status":"INCOMPLETE","errors":[],"warnings":[]}}
                    if pred.get("connected"):
                        try:
                            geometry=generate_design(run_inputs,pred["design"],"MANUAL")
                            bom=calculate_bom(geometry)
                            row.update({"geometry":geometry,"bom":bom})
                            errs=[]
                            if not geometry["validation"]["column_count_matches_coordinates"]: errs.append("column count/coordinates mismatch")
                            if not geometry["validation"]["purlin_spacing_limit_met"]: errs.append("purlin spacing exceeds rule")
                            if not bom.get("connected"): errs.append(bom.get("message","BOM incomplete"))
                            row["validation"]={"status":"INVALID" if errs else "VALID_UNDER_IMPLEMENTED_RULES","errors":errs,"warnings":geometry.get("warnings",[])}
                        except Exception as exc:
                            row["validation"]={"status":"INVALID","errors":[str(exc)],"warnings":[]}
                    else:
                        row["validation"]={"status":"INCOMPLETE","errors":[pred.get("message","prediction failed")],"warnings":[]}
                    candidates.append(row)
                case={"case_id":f"{width}x{length}_{roofing.lower().replace(' ','_').replace('/','_')}_{roof.lower().replace(' ','_')}",
                      "inputs":run_inputs,"candidates":candidates}
                (RESULTS_DIR / f"{case['case_id']}.json").write_text(json.dumps(case,indent=2,allow_nan=False,default=str),encoding="utf-8")
                all_cases.append(case)
    summary={"project":"TRUSS AI Design Studio","cases":len(all_cases),
             "candidate_count":sum(len(c['candidates']) for c in all_cases),
             "valid":sum(x['validation']['status']=="VALID_UNDER_IMPLEMENTED_RULES" for c in all_cases for x in c['candidates']),
             "invalid":sum(x['validation']['status']=="INVALID" for c in all_cases for x in c['candidates']),
             "incomplete":sum(x['validation']['status']=="INCOMPLETE" for c in all_cases for x in c['candidates']),
             "results":all_cases}
    (RESULTS_DIR / "design_regression_tests.json").write_text(json.dumps(summary,indent=2,allow_nan=False,default=str),encoding="utf-8")
    print(f"Saved {len(all_cases)} cases to {RESULTS_DIR / 'design_regression_tests.json'}")
    print({k:summary[k] for k in ('cases','candidate_count','valid','invalid','incomplete')})
if __name__=='__main__': main()
