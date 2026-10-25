# 🏗️ TRUSS AI Design Studio — AI-Assisted Structural Design & Preliminary Estimation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-scikit--learn%20%7C%20CatBoost%20%7C%20XGBoost-brightgreen.svg)](https://scikit-learn.org/)
[![Model Pipeline](https://img.shields.io/badge/Pipeline-truss__hybrid__v4-blueviolet.svg)](#7-machine-learning-models-and-evaluation)

**TRUSS AI Design Studio** is a Python-based application for exploring truss configurations, predicting selected design fields using machine learning, visualizing candidate structural layouts, and estimating preliminary material quantities.

The project combines a Streamlit interface, target-specific ML models, deterministic application logic, a pipe catalogue, result-history helpers, and PDF export functionality.

> **Engineering safety notice:** TRUSS AI is a design-assistance/prototyping project. ML predictions and BOM calculations do not certify structural safety or code compliance. Any design intended for construction or fabrication must be independently verified and approved by a qualified structural engineer.

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Main Capabilities](#2-main-capabilities)
3. [Supported Truss Families](#3-supported-truss-families)
4. [System Architecture](#4-system-architecture)
5. [Repository Structure — Inline File Comments](#5-repository-structure--inline-file-comments)
6. [End-to-End Execution Flow](#6-end-to-end-execution-flow)
7. [Machine-Learning Models and Evaluation](#7-machine-learning-models-and-evaluation)
8. [Model Artifacts](#8-model-artifacts)
9. [Dataset and Preprocessing](#9-dataset-and-preprocessing)
10. [Installation](#10-installation)
11. [Run the Streamlit Application](#11-run-the-streamlit-application)
12. [Preprocess, Train, Validate, and Test](#12-preprocess-train-validate-and-test)
13. [Design Generation, Pipe Catalogue, and BOM](#13-design-generation-pipe-catalogue-and-bom)
14. [PDF Export and Result History](#14-pdf-export-and-result-history)
15. [Configuration and Data Contracts](#15-configuration-and-data-contracts)
16. [Troubleshooting](#16-troubleshooting)
17. [Security, Reproducibility, and Engineering Limitations](#17-security-reproducibility-and-engineering-limitations)
18. [License](#18-license)

---

## 1. Project Overview

TRUSS AI assists with early exploration of steel-truss and building-frame configurations. A user provides supported site/building inputs; the application predicts selected design fields, applies implemented rules, creates a candidate representation, calculates preliminary quantities, and displays results.

The system separates three responsibilities:

1. **Machine-learning inference** — estimates values from patterns in the training data.
2. **Deterministic application logic** — applies coded compatibility rules, geometry operations, quantity calculations, and formatting.
3. **UI and reporting** — presents results through Streamlit, supports result retrieval, and generates PDF output.

High model metrics do not establish that a design is safe. The training data, feature schema, implemented rules, and engineering assumptions all limit the reliability of the output.

---

## 2. Main Capabilities

| Capability | Main implementation | Role |
|---|---|---|
| Interactive design UI | `main/main.py` | Collects inputs and orchestrates the design workflow |
| ML model loading and inference | `main/model.py` | Loads model bundles and predicts design fields |
| Shared training schema | `files/truss_common.py` | Defines features, targets, paths, and ML helpers |
| Data preparation | `files/preprocess_truss.py` | Validates and splits the master dataset |
| Model training | `files/train_truss.py` | Trains candidates and selects models using validation data |
| Validation/testing | `files/validate_truss.py`, `files/test_truss.py` | Measures per-target performance |
| Candidate design generation | `main/design.py` | Builds supported design representations |
| Design alternatives | `main/design_options.py` | Evaluates supported design options |
| Preliminary BOM | `main/calculator.py` | Calculates quantities from design/catalogue information |
| PDF output | `main/pdf_export.py` | Produces a report from application result data |
| Result retrieval | `main/result_history.py` | Caches/looks up prior results |
| Pipe catalogue | `main/pipes.csv` | Stores pipe/section reference data |

---

## 3. Supported Truss Families

The dataset and reference assets include names such as:

- **Full-A Truss**
- **Half-A Truss**
- **Box Truss**
- **Semi-Truss**
- **Plain Pipe**

The exact supported options depend on the current dataset labels and application logic. Reference images in `DESIGNS/` and `main/reference_designs/` are illustrations, not structural calculations or fabrication drawings.

```text
      FULL-A TRUSS             HALF-A TRUSS             BOX TRUSS
           /\                       /|                 |---------|
          /  \                     / |                 |         |
         /____\                   /__|                 |---------|

      SEMI-TRUSS                PLAIN PIPE
         /====\              =================
        /      \
```

These silhouettes are illustrative only; they do not define member sizes, connections, load capacity, or engineering approval.

---

## 4. System Architecture

```text
                          USER INPUTS
                   Streamlit interface: main/main.py
                                |
                                v
                        main/model.py
                   Load model catalog/metadata
                                |
                                v
                     Sequential ML inference
               +----------------+----------------+
               |                |                |
               v                v                v
          Truss family      Rise height     Design fields
                                               |
                         +---------------------+---------------------+
                         |                     |                     |
                         v                     v                     v
                     Pipe sizes             Spacings              Counts
                         |                     |                     |
                         +---------------------+---------------------+
                                               |
                                               v
                                 Implemented design rules
                                   main/design.py
                                               |
                              +----------------+----------------+
                              |                                 |
                              v                                 v
                     Design representation              BOM calculator
                       and visualization                 main/calculator.py
                              |                                 |
                              +----------------+----------------+
                                               |
                                               v
                                Result UI / PDF / history
                         main/pdf_export.py / result_history.py
```

### Model development path

```text
DATASET/*.csv
    |
    v
files/preprocess_truss.py
    |
    +--> preprocessed/train.csv
    +--> preprocessed/validation.csv
    +--> preprocessed/test.csv
    +--> preprocessed/columns.json
    +--> preprocessed/preprocessing_report.json
                    |
                    v
          files/train_truss.py
                    |
                    v
          models/truss_hybrid_v4/
          target-specific .joblib files
          pipeline_metadata.json
                    |
           +--------+---------+
           |                  |
           v                  v
files/validate_truss.py  files/test_truss.py
           |                  |
           +--------+---------+
                    v
          results/truss_hybrid_v4/
          metrics and predictions
```

---

## 5. Repository Structure — Inline File Comments

**The comments are intentionally placed on the same line as each file/folder entry wherever practical, rather than underneath it.** The structure below is based on the supplied project ZIP and the separate screenshot of `models/truss_hybrid_v4/`.

```text
TRUSS/
├── DATASET/                                                     # Master datasets used by preprocessing and training.
│   └── truss_ai_complete_dataset_10000_purlin_remainder_fixed.csv  # Preferred master CSV; verify its actual row count and back it up before editing.
│
├── DESIGNS/                                                    # Reference illustrations used for supported truss-family examples.
│   ├── BOX.png                                                 # Box Truss profile reference.
│   ├── BOX_TRUSS_1SIDE.png                                     # One-side Box Truss variant reference.
│   ├── FULL A TRUSS.png                                        # Full-A Truss profile reference.
│   ├── HALF_A.png                                              # Half-A profile reference.
│   ├── HALF_A_TRUSS.png                                        # Additional Half-A Truss reference.
│   ├── SEMI_TRUSS.png                                          # Semi-Truss profile reference.
│   ├── ZIG-ZAG-A.png                                           # Zig-Zag-A layout reference.
│   ├── ZIGZAG_BEAM_1SIDE.png                                   # One-side Zig-Zag beam reference.
│   ├── ZIG_ZAG_BOX.png                                         # Zig-Zag Box layout reference.
│   └── ZIG_ZAG_HALF_A.png                                      # Zig-Zag Half-A layout reference.
│
├── files/                                                      # Dataset processing, model training, evaluation, and shared ML logic.
│   ├── filter.py                                               # Dataset filtering/inspection utility; inspect supported arguments before use.
│   ├── preprocess_truss.py                                     # Validates the master CSV and generates train/validation/test splits plus schema reports.
│   ├── train_truss.py                                          # Trains target-specific candidates and selects models using validation metrics.
│   ├── validate_truss.py                                      # Evaluates saved model bundles on validation.csv and writes metrics/predictions.
│   ├── test_truss.py                                           # Evaluates saved bundles on held-out test.csv; keep the test set out of tuning.
│   └── truss_common.py                                         # Training-side feature/target schema, paths, encoders, and shared ML helpers.
│
├── main/                                                       # Streamlit UI, inference, design logic, BOM, reporting, and app-side utilities.
│   ├── calculator.py                                           # Calculates preliminary BOM quantities from generated design data and the pipe catalogue.
│   ├── config.py                                               # Resolves project paths, model directory, results directory, and preprocessed-data directory.
│   ├── design.py                                               # Generates candidate design geometry/visualization and applies implemented design rules.
│   ├── design_options.py                                       # Evaluates supported design/layout alternatives; not a structural safety certification.
│   ├── main.py                                                 # Streamlit entry point; collects inputs and orchestrates inference, design, BOM, history, and reports.
│   ├── master datatset.csv                                    # Additional master/reference CSV; filename has a typo and is not automatically the active training source.
│   ├── model.py                                                # Loads metadata/model bundles and performs staged predictions and output decoding.
│   ├── pdf_export.py                                           # Creates PDF output from the result/design data supplied by the application.
│   ├── pipes.csv                                               # Pipe/section catalogue; keep expected columns/units and verify values against trusted specifications.
│   ├── reference_designs/                                      # Reference illustrations used by the application where configured.
│   │   ├── BOX.png                                             # Box Truss reference asset.
│   │   ├── BOX_TRUSS_1SIDE.png                                 # One-side Box Truss reference asset.
│   │   ├── FULL A TRUSS.png                                    # Full-A Truss reference asset.
│   │   ├── HALF_A.png                                          # Half-A Truss reference asset.
│   │   └── SEMI_TRUSS.png                                      # Semi-Truss reference asset.
│   ├── result_history.py                                       # Cache-key, exact-result lookup, persistence, and similar-result history helpers.
│   ├── run_regression.py                                       # Regression/smoke-test utility; inspect its cases/options before running.
│   └── truss_common.py                                         # Application-side shared helpers; separate from files/truss_common.py.
│
├── preprocessed/                                               # Generated dataset splits and preprocessing metadata.
│   ├── columns.json                                            # Schema/column metadata for the prepared data.
│   ├── preprocessing_report.json                               # Machine-readable preprocessing and split report; review after regeneration.
│   ├── reference_master_32_columns.csv                         # Reference master CSV with 32 columns; not automatically the active source.
│   ├── train.csv                                               # Training split used to fit candidate estimators.
│   ├── validation.csv                                          # Validation split used for model comparison/selection.
│   └── test.csv                                                # Held-out split for final evaluation.
│
├── results/                                                    # Generated metrics, predictions, and runtime output.
│   ├── truss_hybrid_v4/                                        # Evaluation artifacts for the V4 model pipeline.
│   │   ├── training_summary.json                               # Summary of the training run and selected candidates.
│   │   ├── validation_metrics.json                             # Per-target validation metrics.
│   │   ├── validation_predictions.csv                          # Validation predictions and target values for error analysis.
│   │   ├── test_metrics.json                                   # Per-target held-out test metrics.
│   │   └── test_predictions.csv                                # Test predictions and target values for error analysis.
│   └── browser_sessions/                                       # Runtime session files created by the UI when used; may not exist until first run.
│       └── *.json                                              # Session-specific state; do not publish private/user-supplied design data.
│
├── models/                                                     # Saved model bundles; required for inference but absent from the supplied ZIP.
│   └── truss_hybrid_v4/                                        # Model version expected by main/config.py and the inference loader.
│       ├── beam_pipe_size.joblib                               # Predicts beam pipe size.
│       ├── beam_total_count.joblib                             # Predicts total beam count.
│       ├── column_pipe_size.joblib                             # Predicts column pipe size.
│       ├── column_spacing_across_span_ft.joblib                # Predicts column spacing across the span.
│       ├── column_spacing_along_length_ft.joblib               # Predicts column spacing along building length.
│       ├── column_total_count.joblib                           # Predicts total column count.
│       ├── frame_chord_pipe_size.joblib                        # Predicts frame/chord pipe size.
│       ├── purlin_orientation.joblib                            # Predicts purlin orientation.
│       ├── purlin_pipe_size.joblib                              # Predicts purlin pipe size.
│       ├── purlin_spacing_ft.joblib                             # Predicts purlin spacing.
│       ├── rise_height_ft.joblib                                # Predicts truss rise height.
│       ├── truss_family.joblib                                   # Predicts the truss-family category.
│       ├── truss_spacing_ft.joblib                               # Predicts spacing between trusses.
│       ├── truss_total_count.joblib                              # Predicts total truss count.
│       ├── web_intermediate_member_count.joblib                  # Predicts intermediate web-member count.
│       ├── web_pipe_size.joblib                                  # Predicts web-member pipe size.
│       ├── web_spacing_ft.joblib                                 # Predicts web spacing; inspect metadata for target handling.
│       ├── web_total_member_count.joblib                         # Predicts total web-member count.
│       └── pipeline_metadata.json                                # Model manifest; filenames/schema must match the saved bundles.
│
└── README.md                                                   # This documentation: architecture, inline file map, setup, ML workflow, use, and limitations.
```

### Structure notes

- The model files above were visible in your screenshot but were **not included in the supplied ZIP**. Copy the matching files and metadata into `models/truss_hybrid_v4/` before inference.
- `main/config.py` and the training-side shared utilities must agree on the model output path.
- `main/truss_common.py` and `files/truss_common.py` are separate modules. Update the correct module(s) when changing schema or inference behavior.
- The two reference-image folders overlap. Keep both if different code paths use them.
- `results/browser_sessions/` is a runtime folder and may be created only after the UI is run.

---

## 6. End-to-End Execution Flow

### 6.1 Training

1. `files/preprocess_truss.py` reads the preferred master dataset.
2. It validates the schema/data and generates train, validation, and test splits.
3. `files/train_truss.py` trains target-specific candidates.
4. Candidate selection uses validation data.
5. The chosen bundles and `pipeline_metadata.json` are saved to the configured model directory.
6. Validation/test scripts evaluate the saved bundles and write metrics/prediction files.

### 6.2 Inference and design

1. The user enters supported site and roof inputs in Streamlit.
2. `main/model.py` reads metadata and loads the required model bundles.
3. The truss-family model predicts `truss_family`.
4. The rise model predicts `rise_height_ft` using base inputs plus the predicted family.
5. Downstream models predict pipe sizes, spacing, orientation, and counts.
6. The application decodes outputs and applies implemented guardrails.
7. Design code builds a candidate layout.
8. The calculator estimates preliminary BOM quantities.
9. The UI displays results and can generate a PDF or retrieve history.

Sequential inference can propagate errors: an incorrect family or rise prediction can affect downstream outputs. Evaluate the complete pipeline, not only isolated estimators.

---

## 7. Machine-Learning Models and Evaluation

### 7.1 Base feature schema

The current shared V4 schema in `files/truss_common.py` defines:

```python
BASE_FEATURES = [
    "site_width_ft",
    "site_length_ft",
    "roofing_type",
    "roof_sides",
    "center_support_present",
    "column_height_B_ft",
    "roof_configuration_detail",
]
```

The current schema excludes the constant `overhang_ft` field from model features. If `roof_configuration_detail` is missing in the source CSV, the shared schema documents a fallback of `"Unknown"` rather than fabricating a detailed value.

These are model inputs, not a complete engineering design brief.

### 7.2 Prediction stages

| Stage | Target | Feature dependency |
|---|---|---|
| 1 | `truss_family` | Base features |
| 2 | `rise_height_ft` | Base features + predicted family |
| 3 | Design targets | Base features + predicted family + predicted rise |

Downstream targets:

- `frame_chord_pipe_size`
- `web_pipe_size`
- `web_spacing_ft`
- `web_intermediate_member_count`
- `web_total_member_count`
- `truss_spacing_ft`
- `truss_total_count`
- `column_pipe_size`
- `column_total_count`
- `column_spacing_along_length_ft`
- `column_spacing_across_span_ft`
- `beam_pipe_size`
- `beam_total_count`
- `purlin_pipe_size`
- `purlin_orientation`
- `purlin_spacing_ft`

Together with family and rise, these correspond to the 18 model files shown in the screenshot.

### 7.3 Algorithms and target types

The available training source supports Random Forest and optional CatBoost/XGBoost candidates. The actual winner per target is defined by the training outputs/metadata. Do not claim LightGBM is part of the current trained pipeline unless the current code and artifacts confirm it.

- **Classification:** categorical outputs such as family, pipe-size labels, and orientation.
- **Regression:** continuous numeric outputs such as rise height.
- **Discrete values:** if a numeric target has only a few permitted values, check whether the current pipeline treats it as classification or regression and how it decodes predictions.

### 7.4 Reported evaluation results

The supplied `results/truss_hybrid_v4/test_metrics.json` reports 1,500 test rows. Selected values are:

| Target | Metric | Reported value |
|---|---|---:|
| Truss family | Accuracy | 98.73% |
| Truss family | Weighted F1 | 0.9882 |
| Rise height | MAE | 0.00213 ft |
| Rise height | RMSE | 0.00468 ft |
| Rise height | R² | 0.999998 |
| Frame/chord pipe size | Accuracy | 98.27% |
| Web pipe size | Accuracy | 98.73% |
| Web spacing | Accuracy | 98.87% |

These are dataset evaluation metrics, not guarantees of real-world correctness. Check the complete JSON for every target and per-class performance. High scores can occur when generated records follow deterministic rules; also check for duplicate designs, target leakage, and distribution shift.

### 7.5 Metric definitions

- **Accuracy:** fraction of correct classification predictions.
- **Macro F1:** equal-weight average of class-level F1 scores.
- **Weighted F1:** class-level F1 weighted by class support.
- **MAE:** average absolute error in the target's units.
- **RMSE:** root mean squared error, which penalizes large errors more heavily.
- **R²:** comparison against a constant-mean regression baseline; it is not a structural safety score.

---

## 8. Model Artifacts

The screenshot shows 18 `.joblib` model bundles and `pipeline_metadata.json` under `models/truss_hybrid_v4/`.

| Artifact | Predicted field |
|---|---|
| `truss_family.joblib` | Truss family |
| `rise_height_ft.joblib` | Rise height |
| `frame_chord_pipe_size.joblib` | Frame/chord pipe size |
| `web_pipe_size.joblib` | Web pipe size |
| `web_spacing_ft.joblib` | Web spacing |
| `web_intermediate_member_count.joblib` | Intermediate web-member count |
| `web_total_member_count.joblib` | Total web-member count |
| `truss_spacing_ft.joblib` | Truss spacing |
| `truss_total_count.joblib` | Total truss count |
| `column_pipe_size.joblib` | Column pipe size |
| `column_total_count.joblib` | Total column count |
| `column_spacing_along_length_ft.joblib` | Column spacing along building length |
| `column_spacing_across_span_ft.joblib` | Column spacing across span |
| `beam_pipe_size.joblib` | Beam pipe size |
| `beam_total_count.joblib` | Beam count |
| `purlin_pipe_size.joblib` | Purlin pipe size |
| `purlin_orientation.joblib` | Purlin orientation |
| `purlin_spacing_ft.joblib` | Purlin spacing |
| `pipeline_metadata.json` | Model catalog/manifest |

The manifest must match the actual filenames and schema. Keep it with bundles from the same training run. Never load untrusted `.joblib` files because pickle-based serialization can execute code during deserialization.

---

## 9. Dataset and Preprocessing

### Preferred master dataset

```text
DATASET/truss_ai_complete_dataset_10000_purlin_remainder_fixed.csv
```

The filename suggests 10,000 rows; verify the actual row count. The archive also includes `main/master datatset.csv` and `preprocessed/reference_master_32_columns.csv`; neither automatically replaces the preferred source.

### Preprocessing outputs

- `preprocessed/train.csv` — fits model candidates.
- `preprocessed/validation.csv` — compares candidates and supports model selection.
- `preprocessed/test.csv` — held-out final evaluation.
- `preprocessed/columns.json` — schema/column metadata.
- `preprocessed/preprocessing_report.json` — preparation and split summary.

### Data-quality checklist

- Required columns exist and are consistently named.
- Numeric measurements parse and use consistent units.
- Counts are valid and non-negative.
- Pipe labels match the expected categories/catalogue.
- Category spelling and capitalization are consistent.
- Related/duplicate designs are grouped appropriately across splits.
- No input feature directly or indirectly leaks a target.
- Rare target classes have adequate representation.

Do not fill missing target labels with invented structural values. Correct the source or use a documented filtering strategy.

---

## 10. Installation

### Requirements

- Python compatible with the model artifacts and dependency versions.
- pip.
- Streamlit, pandas, NumPy, scikit-learn, and joblib.
- Optional CatBoost/XGBoost if the training code uses them.
- Plotting/PDF libraries imported by the current source.

A virtual environment is recommended to isolate dependencies, but it is not inherently required by the application.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn joblib
python -m pip install catboost xgboost matplotlib plotly reportlab
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn joblib
python -m pip install catboost xgboost matplotlib plotly reportlab
```

The supplied archive does not contain a root `requirements.txt`, so these commands are a starting point, not a verified lockfile. Confirm all imports in a clean environment and create a pinned requirements file for reproducibility.

Saved scikit-learn pipelines can be version-sensitive. Use compatible versions from the training environment when possible.

---

## 11. Run the Streamlit Application

From the project root:

```bash
python -m streamlit run main/main.py
```

Open the URL printed by Streamlit, usually `http://localhost:8501`.

Before launch, verify that these paths exist:

```text
main/main.py
main/config.py
main/model.py
main/design.py
main/calculator.py
main/pipes.csv
files/truss_common.py
models/truss_hybrid_v4/pipeline_metadata.json
models/truss_hybrid_v4/*.joblib
```

The model directory was shown in the screenshot but was not included in the supplied ZIP. Copy the matching, trusted model files and metadata into the expected path before inference.

If port 8501 is occupied:

```bash
python -m streamlit run main/main.py --server.port 8502
```

---

## 12. Preprocess, Train, Validate, and Test

Run commands from the project root.

### Step 1 — Preprocess

```bash
python files/preprocess_truss.py
```

Review the preprocessing report and resolve missing columns, invalid values, unexpected split sizes, or leakage concerns.

### Step 2 — Train

```bash
python files/train_truss.py
```

Inspect supported arguments for the exact source version:

```bash
python files/train_truss.py --help
```

Confirm the output path agrees with `main/config.py`.

### Step 3 — Validate

```bash
python files/validate_truss.py
```

Inspect per-target metrics and predictions, including rare classes and large-error cases.

### Step 4 — Test

```bash
python files/test_truss.py
```

Use the held-out test split for final evaluation. Repeated tuning against it invalidates its role as an independent test set.

### Step 5 — Optional filtering

```bash
python files/filter.py --help
```

Inspect supported options and output behavior before filtering. Preserve a backup of the original master CSV.

### Release checklist

1. Record dataset version and code revision.
2. Run preprocessing and review the report.
3. Train and preserve the exact model bundles plus metadata.
4. Run validation and held-out testing.
5. Record dependency versions and metrics.
6. Smoke-test model loading/inference.
7. Test geometry, BOM, and PDF output on boundary cases.
8. Keep evaluation results associated with the model version they evaluate.

---

## 13. Design Generation, Pipe Catalogue, and BOM

### Design generation

`main/design.py` contains design-generation/drawing helpers; `main/design_options.py` evaluates supported alternatives. These modules use inputs, predictions, and coded rules, but do not replace structural analysis.

Test small, typical, large, and boundary-size sites, including each supported roof option and center-support case.

### Pipe catalogue

`main/pipes.csv` supplies pipe/section values. Preserve the expected column names and units. Verify dimensions, thickness, section weight, and material grade against trusted supplier/manufacturer data before using them for real fabrication.

### BOM calculator

`main/calculator.py` estimates material quantities from the generated design and catalogue. Independently verify:

- Member lengths and counts.
- Conversions among feet, inches, millimetres, and metres.
- Stock lengths, cutting kerf, and remainder handling.
- Roofing coverage/lapping and overhang assumptions, if implemented.
- Waste/reserve quantities.
- Catalogue labels and section weights.
- Missing or duplicated members.
- Cost values and units, if cost estimation is enabled.

### Engineering review

A qualified structural engineer must independently verify load combinations, wind uplift, seismic effects where applicable, member strength/buckling, deflection, stability, bracing, connections, welds/bolts, columns, anchors, foundations, material grade, corrosion, and local codes. The application does not guarantee that every factor is represented.

---

## 14. PDF Export and Result History

### PDF export

`main/pdf_export.py` builds a PDF from the result/design data passed by the application. The exact report content depends on the current implementation. If result keys or BOM structures change, update and test the exporter.

### Result history

`main/result_history.py` provides cache/history helpers for cache-key generation, exact-result lookup, persistence, and similar-result retrieval. History is for application continuity and comparison; it is not an engineering approval record.

### Evaluation files

```text
results/truss_hybrid_v4/
├── training_summary.json       # Training-run summary and selected-candidate information.
├── validation_metrics.json     # Per-target validation metrics.
├── validation_predictions.csv  # Validation predictions for error analysis.
├── test_metrics.json           # Per-target held-out test metrics.
└── test_predictions.csv        # Test predictions for error analysis.
```

After retraining, ensure that these files correspond to the same model artifacts and dataset splits being documented.

---

## 15. Configuration and Data Contracts

### `main/config.py`

Defines the project root and paths for:

- `MODEL_DIR` → `models/truss_hybrid_v4/`
- `RESULTS_DIR` → `results/truss_hybrid_v4/`
- `PREPROCESSED_DIR` → `preprocessed/`

### Shared schema modules

- `files/truss_common.py` — training-side features, targets, paths, and ML helpers.
- `main/truss_common.py` — separate application-side helpers; it is not automatically synchronized with the training module.

### Metadata

`pipeline_metadata.json` identifies the model bundles expected by inference. It must match the actual artifacts and schema from the same training run.

When changing a feature, target, category, or output type, trace it end-to-end:

```text
master dataset
  → preprocessing/splits
  → training
  → metadata and model bundles
  → inference/decoding
  → design generation
  → BOM calculator
  → PDF/report
```

Changing only one stage can create missing-column errors or inconsistent outputs.

---

## 16. Troubleshooting

| Symptom | Likely cause | Recommended check |
|---|---|---|
| Model directory missing | Artifacts absent from expected path | Copy the model catalog into `models/truss_hybrid_v4/` |
| Metadata missing/invalid | Incomplete model export | Restore metadata from the same training run |
| Missing `.joblib` file | Manifest and files do not match | Compare `model_files` against actual filenames |
| Model loading error | Dependency mismatch or corrupt artifact | Use compatible package versions and verify the file |
| Dataset not found | Wrong working directory or missing CSV | Run from project root and check `DATASET/` |
| Schema mismatch | CSV headers differ from shared schema | Compare headers with `files/truss_common.py` |
| Weak rare-class results | Imbalance or distribution shift | Inspect macro-F1 and per-class metrics |
| Inconsistent downstream fields | Sequential error propagation | Inspect each prediction stage and guardrail |
| Incorrect BOM | Unit, geometry, count, catalogue, or rounding issue | Trace output through `main/calculator.py` |
| PDF export failure | Missing dependency or changed result fields | Inspect traceback and exporter inputs |
| Streamlit port occupied | Another instance is running | Use `--server.port 8502` |

For reproducible debugging, record the exact input, expected/actual output, raw prediction, post-guardrail value, model version, dependency versions, and traceback.

---

## 17. Security, Reproducibility, and Engineering Limitations

### Security

- Load only trusted `.joblib` files; pickle-based deserialization can execute code.
- Do not commit credentials or private design data.
- Review what is stored in session/history files before publishing.
- Add authentication and access controls before exposing the app publicly.

### Reproducibility

Keep together for each release:

- Dataset version/checksum.
- Preprocessing code and split files.
- Python/library versions.
- Training configuration and random seed.
- Model bundles and matching metadata.
- Validation/test metrics and known limitations.
- Source code revision.

### Engineering limitations

Models learn patterns from their dataset. Out-of-range inputs may be unreliable; sequential predictions can compound errors; guardrails enforce only rules explicitly implemented in code.

High test metrics do not establish structural safety. Do not use this application as the sole basis for construction, fabrication, or certification. Obtain independent calculations and approval from a qualified structural engineer.

---

## 18. License

The supplied ZIP did not contain a `LICENSE` file. Add the appropriate license before publishing. Do not claim MIT licensing until the actual license file is included and compatible with all bundled code, datasets, drawings, and third-party assets.

---

**TRUSS AI Design Studio — ML-assisted truss configuration, candidate visualization, and preliminary material estimation.**
