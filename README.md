# 🏗️ TRUSS AI Design Studio — AI-Assisted Structural Design & Preliminary Estimation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-scikit--learn%20%7C%20CatBoost%20%7C%20XGBoost-brightgreen.svg)](https://scikit-learn.org/)
[![Model Pipeline](https://img.shields.io/badge/Pipeline-truss__hybrid__v4-blueviolet.svg)](#7-machine-learning-models-and-evaluation)
[![License](https://img.shields.io/badge/License-Add%20before%20publishing-lightgrey.svg)](#18-license)

**TRUSS AI Design Studio** is a Python-based application for exploring truss configurations, predicting selected design fields using machine learning, visualizing candidate structural layouts, and estimating preliminary material quantities.

The project combines a Streamlit interface, target-specific ML models, deterministic design/application logic, a pipe catalogue, result-history helpers, and PDF export functionality.

> **Engineering safety notice:** TRUSS AI is a design-assistance/prototyping project. Model accuracy and BOM calculations do not demonstrate structural safety or code compliance. Do not use generated output as the sole basis for fabrication or construction. A qualified structural engineer must independently verify loads, members, connections, bracing, foundations, serviceability, and applicable codes.

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Main Capabilities](#2-main-capabilities)
3. [Supported Truss Families](#3-supported-truss-families)
4. [System Architecture](#4-system-architecture)
5. [Repository Structure — Every File Explained](#5-repository-structure--every-file-explained)
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

TRUSS AI is designed to assist with the early exploration of steel-truss and building-frame configurations. A user provides supported site/building inputs. The application then predicts selected design fields, applies implemented rules, creates a candidate design representation, calculates preliminary quantities, and displays the results.

The system separates three responsibilities:

1. **Machine-learning inference** — estimates values from patterns in the training data.
2. **Deterministic application logic** — applies explicitly coded compatibility rules, geometry operations, quantities, and formatting.
3. **User interface and reporting** — presents results through Streamlit, supports result retrieval, and generates PDF output.

The system does not automatically prove that a predicted structure is safe. The dataset, model features, implemented rules, and engineering assumptions all limit the reliability of its outputs.

### Intended use

- Academic demonstrations of ML-assisted structural-design workflows.
- Comparing supported candidate truss configurations.
- Inspecting model performance against held-out data.
- Producing preliminary layouts and material estimates for professional review.

### Important scope distinction

The repository contains reference drawings and a model pipeline. A reference image does not prove that a design family is fully implemented in every code path. Actual behavior is determined by the current application source, configuration, available model artifacts, and catalogue data.

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
| Pipe catalogue | `main/pipes.csv` | Stores section/pipe reference data |

---

## 3. Supported Truss Families

The dataset and reference assets include names such as:

- **Full-A Truss**
- **Half-A Truss**
- **Box Truss**
- **Semi-Truss**
- **Plain Pipe**

The exact supported options depend on the current dataset labels and application logic. The images in `DESIGNS/` and `main/reference_designs/` are visual references; they are not structural calculations or fabrication drawings.

### Reference profile sketches

```text
      FULL-A TRUSS             HALF-A TRUSS             BOX TRUSS
           /\                       /|                 |---------|
          /  \                     / |                 |         |
         /____\                   /__|                 |---------|

      SEMI-TRUSS                PLAIN PIPE
         /====\              =================
        /      \
```

These are illustrative silhouettes only. They do not define member sizes, connection details, load capacity, or engineering approval.

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

**Path note:** `main/config.py` and `files/truss_common.py` configure model artifacts under `models/truss_hybrid_v4/`. Training and inference must use the same model directory. The supplied source ZIP did not include the model directory, although a separate screenshot shows its contents.

---

## 5. Repository Structure — Every File Explained

The structure below is based on the supplied `TRUSS - Copy(1).zip` archive and the separately supplied screenshot of `models/truss_hybrid_v4/`. Comments describe observed file roles and intended responsibilities; where a command-line interface or internal behavior is not fully established, inspect the file before relying on an assumption.

```text
TRUSS/
│
├── DATASET/
│   └── truss_ai_complete_dataset_10000_purlin_remainder_fixed.csv
│       # Preferred master CSV used by the preprocessing pipeline.
│       # Contains source design features and target/output columns.
│       # Back up this file before filtering or replacing it.
│       # The filename suggests 10,000 rows; verify the actual row count.
│
├── DESIGNS/
│   ├── BOX.png
│   │   # Reference image for the Box Truss profile.
│   ├── BOX_TRUSS_1SIDE.png
│   │   # Reference image for a one-side Box Truss variant.
│   ├── FULL A TRUSS.png
│   │   # Reference image for the Full-A Truss profile.
│   ├── HALF_A.png
│   │   # Reference image for a Half-A variant.
│   ├── HALF_A_TRUSS.png
│   │   # Additional Half-A Truss reference image.
│   ├── SEMI_TRUSS.png
│   │   # Reference image for a Semi-Truss profile.
│   ├── ZIG-ZAG-A.png
│   │   # Reference image for a Zig-Zag-A layout.
│   ├── ZIGZAG_BEAM_1SIDE.png
│   │   # Reference image for a one-side Zig-Zag beam layout.
│   ├── ZIG_ZAG_BOX.png
│   │   # Reference image for a Zig-Zag Box layout.
│   └── ZIG_ZAG_HALF_A.png
│       # Reference image for a Zig-Zag Half-A layout.
│
├── files/
│   ├── filter.py
│   │   # Dataset filtering/inspection utility.
│   │   # Read its arguments before use; do not assume every option removes
│   │   # statistical outliers or preserves every rare target category.
│   │   # Keep the source CSV unchanged until the filtered output is checked.
│   │
│   ├── preprocess_truss.py
│   │   # Loads the preferred master dataset and validates its schema/data.
│   │   # Builds train, validation, and test splits.
│   │   # Writes prepared CSVs plus schema and preprocessing-report JSON files.
│   │   # Uses the shared schema/grouping logic in truss_common.py.
│   │
│   ├── train_truss.py
│   │   # Trains target-specific ML models for the current V4 feature schema.
│   │   # Compares available candidates using validation metrics.
│   │   # Candidate libraries depend on the installed packages and script code.
│   │   # Writes model bundles and metadata to the configured model directory.
│   │   # The test split should not be used to select candidate winners.
│   │
│   ├── validate_truss.py
│   │   # Loads the saved model catalog and evaluates it on validation.csv.
│   │   # Writes per-target validation metrics and prediction outputs.
│   │   # Useful for model selection diagnostics and error analysis.
│   │
│   ├── test_truss.py
│   │   # Evaluates saved models against the held-out test.csv split.
│   │   # Writes final test metrics and prediction outputs.
│   │   # Keep the test split independent during model tuning.
│   │
│   └── truss_common.py
│       # Shared ML schema/utilities for preprocessing, training and inference.
│       # Defines base features, target fields, category/count handling,
│       # data paths, random seed, and shared model/helper functions.
│       # Changes here may require preprocessing, retraining, and re-export.
│
├── main/
│   ├── calculator.py
│   │   # Calculates preliminary material quantities from generated design data.
│   │   # Uses pipe catalogue values and application-specific quantity logic.
│   │   # Check lengths, units, rounding, cutting, and catalogue weights.
│   │   # Outputs must be checked against approved drawings before procurement.
│   │
│   ├── config.py
│   │   # Centralizes paths used by the application.
│   │   # PROJECT_ROOT is resolved relative to the main/ directory.
│   │   # MODEL_DIR points to models/truss_hybrid_v4/.
│   │   # RESULTS_DIR points to results/truss_hybrid_v4/.
│   │   # PREPROCESSED_DIR points to preprocessed/.
│   │
│   ├── design.py
│   │   # Candidate design generation and drawing/visualization helpers.
│   │   # Converts user inputs and predictions into a supported layout.
│   │   # Contains application rules/compatibility logic used by the UI.
│   │   # Test geometry boundaries whenever dimensions or rules change.
│   │
│   ├── design_options.py
│   │   # Evaluates supported alternative design/layout options.
│   │   # Keep its assumptions consistent with design.py and the model schema.
│   │   # Do not interpret a ranked option as verified structural safety.
│   │
│   ├── main.py
│   │   # Streamlit application entry point and orchestration layer.
│   │   # Collects user inputs, invokes prediction, displays the candidate
│   │   # design, and connects BOM, history, and PDF output components.
│   │   # Uses browser-session state to help restore the user's recent result.
│   │   # Launch with `python -m streamlit run main/main.py`.
│   │
│   ├── master datatset.csv
│   │   # Additional master/reference CSV included in the supplied archive.
│   │   # "datatset" is a spelling inconsistency in the filename.
│   │   # The active preprocessing script prefers the DATASET/ source path.
│   │   # Compare schema and contents before treating this as authoritative.
│   │
│   ├── model.py
│   │   # Loads pipeline_metadata.json and required .joblib model bundles.
│   │   # Runs staged prediction: family, rise, and downstream design fields.
│   │   # Applies shared decoding/guardrail helpers from files/truss_common.py.
│   │   # Contains a narrow compatibility shim for a historical scikit-learn
│   │   # helper name; it does not guarantee all versions are compatible.
│   │
│   ├── pdf_export.py
│   │   # Generates PDF output from the design/result data supplied by the UI.
│   │   # Report contents depend on the fields provided by the current code.
│   │   # Update and test this module if the result/BOM data contract changes.
│   │
│   ├── pipes.csv
│   │   # Pipe/section reference catalogue used by design/BOM logic.
│   │   # Keep the expected column names and units intact.
│   │   # Verify section dimensions, thicknesses and weights with a trusted
│   │   # supplier/manufacturer catalogue before real-world use.
│   │
│   ├── reference_designs/
│   │   ├── BOX.png
│   │   │   # Box Truss reference asset.
│   │   ├── BOX_TRUSS_1SIDE.png
│   │   │   # One-side Box Truss reference asset.
│   │   ├── FULL A TRUSS.png
│   │   │   # Full-A Truss reference asset.
│   │   ├── HALF_A.png
│   │   │   # Half-A Truss reference asset.
│   │   └── SEMI_TRUSS.png
│   │       # Semi-Truss reference asset.
│   │
│   ├── result_history.py
│   │   # Helpers for result cache keys, exact-result lookup, persistence,
│   │   # and finding similar historical inputs/results.
│   │   # History is an application convenience, not an engineering approval.
│   │
│   ├── run_regression.py
│   │   # Regression/smoke-test utility included with the application.
│   │   # Inspect its test cases and arguments before running.
│   │   # Run after changing inference, geometry, or BOM behavior.
│   │
│   └── truss_common.py
│       # Application-side shared helpers.
│       # This is distinct from files/truss_common.py; changes are not
│       # automatically synchronized between these two modules.
│
├── preprocessed/
│   ├── columns.json
│   │   # Column/schema metadata generated by preprocessing.
│   │   # Keep it aligned with the dataset schema used for the model run.
│   ├── preprocessing_report.json
│   │   # Machine-readable preprocessing and split summary.
│   │   # Review it after every data-preparation run.
│   ├── reference_master_32_columns.csv
│   │   # Reference master dataset with a 32-column schema in this archive.
│   │   # Useful for schema comparison; not automatically the active source.
│   ├── train.csv
│   │   # Training split used to fit model candidates.
│   ├── validation.csv
│   │   # Validation split used to compare/select model candidates.
│   └── test.csv
│       # Held-out test split for final model evaluation.
│
├── results/
│   └── truss_hybrid_v4/
│       ├── training_summary.json
│       │   # Training-run summary, selected candidates, and validation details.
│       ├── validation_metrics.json
│       │   # Per-target metrics calculated on the validation split.
│       ├── validation_predictions.csv
│       │   # Validation predictions and associated target values for analysis.
│       ├── test_metrics.json
│       │   # Per-target metrics on the held-out test split.
│       └── test_predictions.csv
│           # Test predictions and target values for error analysis.
│
├── models/
│   └── truss_hybrid_v4/                 # Seen in the supplied screenshot.
│       ├── beam_pipe_size.joblib
│       │   # Saved model bundle for beam pipe-size prediction.
│       ├── beam_total_count.joblib
│       │   # Saved model bundle for total beam count.
│       ├── column_pipe_size.joblib
│       │   # Saved model bundle for column pipe-size prediction.
│       ├── column_spacing_across_span_ft.joblib
│       │   # Saved model bundle for column spacing across the span.
│       ├── column_spacing_along_length_ft.joblib
│       │   # Saved model bundle for column spacing along the building length.
│       ├── column_total_count.joblib
│       │   # Saved model bundle for total column count.
│       ├── frame_chord_pipe_size.joblib
│       │   # Saved model bundle for frame-chord pipe-size prediction.
│       ├── purlin_orientation.joblib
│       │   # Saved model bundle for purlin orientation prediction.
│       ├── purlin_pipe_size.joblib
│       │   # Saved model bundle for purlin pipe-size prediction.
│       ├── purlin_spacing_ft.joblib
│       │   # Saved model bundle for purlin-spacing prediction.
│       ├── rise_height_ft.joblib
│       │   # Saved model bundle for truss rise-height prediction.
│       ├── truss_family.joblib
│       │   # Saved model bundle for truss-family classification.
│       ├── truss_spacing_ft.joblib
│       │   # Saved model bundle for spacing between trusses.
│       ├── truss_total_count.joblib
│       │   # Saved model bundle for total truss count.
│       ├── web_intermediate_member_count.joblib
│       │   # Saved model bundle for intermediate web-member count.
│       ├── web_pipe_size.joblib
│       │   # Saved model bundle for web pipe-size prediction.
│       ├── web_spacing_ft.joblib
│       │   # Saved model bundle for web-spacing prediction.
│       ├── web_total_member_count.joblib
│       │   # Saved model bundle for total web-member count.
│       └── pipeline_metadata.json
│           # Model catalog manifest listing the expected model bundles and
│           # related inference metadata. It must match the saved artifacts.
│
└── README.md
    # Project documentation: architecture, setup, file roles, ML pipeline,
    # execution commands, evaluation, limitations, and troubleshooting.
```

### Important structure notes

- The supplied ZIP includes `results/truss_hybrid_v4/` metrics and predictions, but not the `models/truss_hybrid_v4/` folder shown in the screenshot. Copy the matching model bundles and metadata into the expected directory before running inference.
- The screenshot shows 18 `.joblib` files plus `pipeline_metadata.json`. The actual manifest determines which files the loader requires.
- The archive has `.dist/` directories; they are omitted from the runtime tree because their purpose is not established as an application requirement. Do not delete them without checking whether you need them.
- Some reference images appear in both `DESIGNS/` and `main/reference_designs/`. Keep both copies if the code uses both paths.
- `main/truss_common.py` and `files/truss_common.py` are different modules with different roles.

---

## 6. End-to-End Execution Flow

### 6.1 Training flow

1. The preprocessing script loads the preferred master dataset from `DATASET/`.
2. It checks schema/data and writes train, validation, and test splits.
3. The training script builds target-specific candidate estimators.
4. Candidate winners are selected using validation metrics.
5. The selected model bundles and `pipeline_metadata.json` are saved to the configured model directory.
6. The validation and test scripts score the saved models and write metrics/prediction files under `results/truss_hybrid_v4/`.

### 6.2 Inference flow

1. The user enters supported site and roof inputs in Streamlit.
2. `main/model.py` reads model metadata and loads the required `.joblib` bundles.
3. The family model predicts `truss_family`.
4. The rise model predicts `rise_height_ft` using base inputs plus the predicted family.
5. Downstream models predict pipe sizes, spacings, orientations, and counts.
6. Shared inference helpers decode outputs and apply implemented guardrails.
7. Design code builds the candidate representation.
8. The calculator estimates preliminary BOM quantities.
9. The UI presents the result and may export a PDF or store history.

### Sequential prediction risk

Some stages use earlier predictions as input. An incorrect family or rise prediction can affect downstream predictions. Evaluate the full sequential pipeline and inspect per-target error cases, not just standalone model scores.

---

## 7. Machine-Learning Models and Evaluation

### 7.1 Feature schema

The shared V4 schema in `files/truss_common.py` defines the base inputs:

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

The current schema excludes the constant `overhang_ft` field from model features. If `roof_configuration_detail` is absent from a source dataset, the shared schema documents a fallback of `"Unknown"` rather than fabricating a detailed value.

These are ML inputs, not a complete engineering design brief. Do not infer safety from the feature list.

### 7.2 Prediction stages

| Stage | Target | Features |
|---|---|---|
| 1 | `truss_family` | Base features |
| 2 | `rise_height_ft` | Base features + predicted family |
| 3 | Design targets | Base features + predicted family + predicted rise |

The design-target list includes:

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

Together with `truss_family` and `rise_height_ft`, these correspond to 18 target-specific model artifacts.

### 7.3 Algorithms

The training code and saved summary should be treated as authoritative for the actual candidates used in a particular run. The shared code includes Random Forest support and optional CatBoost/XGBoost candidates. Do not claim LightGBM is part of the current trained pipeline unless the current training script and artifacts confirm it.

- **Classification:** categorical targets such as family, pipe-size label, and orientation.
- **Regression:** continuous targets such as rise height and other numeric fields.
- **Discrete values:** where a target has a small set of allowed numeric values, verify whether the current code treats it as classification or regression and how outputs are decoded.

### 7.4 Reported test metrics

The supplied `results/truss_hybrid_v4/test_metrics.json` reports 1,500 test rows. Selected values read from that file are:

| Target | Test metric | Reported value |
|---|---|---:|
| Truss family | Accuracy | 98.73% |
| Truss family | Weighted F1 | 0.9882 |
| Rise height | MAE | 0.00213 ft |
| Rise height | RMSE | 0.00468 ft |
| Rise height | R² | 0.999998 |
| Frame/chord pipe size | Accuracy | 98.27% |
| Web pipe size | Accuracy | 98.73% |
| Web spacing | Accuracy | 98.87% |

These are metrics in the included evaluation file, not guarantees of real-world design correctness. For example, the test report shows lower performance for the less common `Plain Pipe` family and `None (Single Rafter)` web-pipe category than for several common classes. Inspect `test_metrics.json` for every target and per-class results before describing model performance publicly.

Do not use a universal “98.73% accuracy” badge to imply that every model is 98.73% accurate. Different targets use different metrics and have different error distributions.

### 7.5 How to interpret metrics

- **Accuracy:** fraction of classification predictions that are correct.
- **Macro F1:** computes F1 per class and averages them equally; it highlights rare-class performance.
- **Weighted F1:** averages class F1 weighted by class support; common classes have greater influence.
- **MAE:** mean absolute error in the target's units.
- **RMSE:** square-root mean squared error; larger errors are penalized more strongly.
- **R²:** compares regression performance with a constant-mean baseline; it is not a safety score.

Very high scores can occur when generated examples follow deterministic rules. Check for leakage, duplicate designs, target-derived features, and realistic differences between the dataset and future inputs.

---

## 8. Model Artifacts

The supplied screenshot shows the following model bundles under `models/truss_hybrid_v4/`:

| File | Predicted field |
|---|---|
| `truss_family.joblib` | Truss family/category |
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
| `pipeline_metadata.json` | Required model catalog/manifest |

### Model-loading contract

The inference loader expects:

```text
models/truss_hybrid_v4/
├── pipeline_metadata.json
└── [all model bundles listed in metadata]
```

The metadata file must contain a non-empty `model_files` list, and every referenced file must exist. Keep the metadata and bundles from the same training run. Do not rename a bundle without updating the manifest and any related target/schema definitions.

Serialized `.joblib` files can execute code during loading. Only load trusted model artifacts.

---

## 9. Dataset and Preprocessing

### Preferred master file

```text
DATASET/truss_ai_complete_dataset_10000_purlin_remainder_fixed.csv
```

The filename suggests 10,000 rows, but the actual row count must be verified. The archive also contains `main/master datatset.csv` and `preprocessed/reference_master_32_columns.csv`; neither should automatically replace the preferred source.

### Preprocessing outputs

```text
preprocessed/
├── train.csv
├── validation.csv
├── test.csv
├── columns.json
└── preprocessing_report.json
```

- `train.csv` fits model candidates.
- `validation.csv` compares candidates and supports model selection.
- `test.csv` is held out for final evaluation.
- `columns.json` records schema information.
- `preprocessing_report.json` records preparation/split information.

### Data-quality checklist

Before training, check:

- Required columns exist and are named consistently.
- Numeric measurements parse correctly and use consistent units.
- Counts are non-negative and valid for the target.
- Pipe labels match the expected categories/catalogue.
- Categories do not differ because of spelling or capitalization.
- Repeated or related designs are grouped appropriately across splits.
- No feature leaks the target directly or indirectly.
- All target classes have adequate representation.

Do not silently replace missing target labels with invented structural values. Fix the source data or use a documented filtering strategy.

---

## 10. Installation

### Requirements

- Python version compatible with the model artifacts and dependencies.
- pip.
- Streamlit.
- pandas, NumPy, scikit-learn, and joblib.
- Optional candidate libraries such as CatBoost and XGBoost if the training script uses them.
- Any additional plotting/PDF dependencies imported by the current source code.

A virtual environment is recommended to keep dependencies isolated. It is not intrinsically required by the application.

### Windows PowerShell

From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn joblib
python -m pip install catboost xgboost
python -m pip install matplotlib plotly reportlab
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn joblib
python -m pip install catboost xgboost matplotlib plotly reportlab
```

These commands are a practical starting point, not a tested lockfile. The supplied archive does not contain a root `requirements.txt`. Verify all imports in a clean environment, then create a pinned `requirements.txt` for reproducibility.

### Dependency-version caution

Saved scikit-learn pipelines may depend on the version used during training. The compatibility shim in `main/model.py` addresses one historical helper class but cannot guarantee arbitrary cross-version compatibility. Use the training environment's dependency versions whenever possible.

---

## 11. Run the Streamlit Application

Run from the project root:

```bash
python -m streamlit run main/main.py
```

Open the local URL shown in the terminal, commonly:

```text
http://localhost:8501
```

### Before launch

Confirm the following paths exist:

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

**The model folder was not included in the supplied ZIP.** Copy the intended, trusted model artifacts from the separate model directory into the path expected by `main/config.py`.

### Different port

If port 8501 is occupied:

```bash
python -m streamlit run main/main.py --server.port 8502
```

---

## 12. Preprocess, Train, Validate, and Test

Run these commands from the project root.

### Step 1 — Preprocess the dataset

```bash
python files/preprocess_truss.py
```

Review the generated `preprocessing_report.json`. Resolve missing columns, invalid measurements, unexpected split sizes, or leakage concerns before proceeding.

### Step 2 — Train models

```bash
python files/train_truss.py
```

Check available command-line options in the exact source version:

```bash
python files/train_truss.py --help
```

The shared configuration expects trained bundles in `models/truss_hybrid_v4/`. The training and inference paths must agree.

### Step 3 — Validate

```bash
python files/validate_truss.py
```

Review per-target metrics and prediction files. Look at class-level results where available, especially for rare categories.

### Step 4 — Test

```bash
python files/test_truss.py
```

Use the test split as the final held-out evaluation. If you repeatedly tune against it, create a new untouched test split before making generalization claims.

### Step 5 — Optional filtering

```bash
python files/filter.py --help
```

Inspect the supported arguments and output behavior before filtering. Keep the original master CSV unchanged as a backup.

### Release checklist

1. Record the source dataset version.
2. Run preprocessing and inspect the report.
3. Train and preserve the selected model bundles plus metadata.
4. Run validation and held-out testing.
5. Record Python/library versions and the code revision.
6. Smoke-test model loading and inference.
7. Test geometry, BOM, and PDF output on representative boundary cases.
8. Keep the metrics associated with the exact model version they evaluate.

---

## 13. Design Generation, Pipe Catalogue, and BOM

### Design generation

`main/design.py` contains design-generation and drawing helpers. `main/design_options.py` evaluates supported alternatives. These modules operate on the inputs, predictions, and implemented rules; they do not replace structural analysis.

Test design logic using small, typical, large, and boundary-size site dimensions. Include supported roofing choices, roof-side choices, and center-support cases.

### Pipe catalogue

`main/pipes.csv` provides pipe/section reference values used by the application. Preserve the expected column names and units. Verify dimensions, thickness, weight per unit length, and material grade with a trusted supplier/manufacturer catalogue before using the values for procurement or design.

### BOM calculator

`main/calculator.py` computes preliminary material quantities. Results depend on the generated geometry, member lengths/counts, pipe catalogue, unit conversions, rounding, and any cutting/waste assumptions implemented in code.

Before using a BOM for fabrication or purchasing, independently verify:

- Member lengths and counts.
- Conversion between feet, inches, millimetres, and metres.
- Stock lengths, cutting kerf, and remainder handling.
- Lapping, overhang, and coverage assumptions where applicable.
- Reserve/waste quantities.
- Pipe labels and section weights.
- Duplicate or missing members in the generated layout.
- Cost inputs and units, if cost estimation is enabled in the current code.

### Engineering review required

A qualified structural engineer must independently check load combinations, wind uplift, seismic effects where applicable, member strength and buckling, deflection, lateral stability, bracing, joints, welds/bolts, columns, anchors, foundations, corrosion, material grade, and local building codes. The application does not guarantee that all of these are represented.

---

## 14. PDF Export and Result History

### PDF export

`main/pdf_export.py` generates a PDF from the design/result data supplied by the application. The actual report contents are determined by the current implementation and available fields. Do not promise that a report contains a particular schedule or visualization unless that output is present in the current code.

If you change result keys or BOM structure, update and test the PDF exporter.

### Result history

`main/result_history.py` contains result cache/history helpers, including cache-key creation, lookup, persistence, and similar-result retrieval. Treat history as a convenience for comparing application runs, not as an approval or certification record.

### Evaluation files

```text
results/truss_hybrid_v4/
├── training_summary.json
├── validation_metrics.json
├── validation_predictions.csv
├── test_metrics.json
└── test_predictions.csv
```

These files document the saved evaluation run. After retraining, ensure that results correspond to the same dataset splits and model artifacts you are documenting.

---

## 15. Configuration and Data Contracts

### `main/config.py`

The project root is resolved relative to `main/config.py`. It defines paths for:

- `MODEL_DIR` → `models/truss_hybrid_v4/`
- `RESULTS_DIR` → `results/truss_hybrid_v4/`
- `PREPROCESSED_DIR` → `preprocessed/`

### `files/truss_common.py`

The training-side shared schema defines features, targets, paths, target types, and common ML helpers. It is the primary place to inspect when investigating feature/target mismatches in preprocessing and training.

### `main/truss_common.py`

This is a separate application-side module. Do not assume it automatically stays synchronized with the training-side shared schema.

### `pipeline_metadata.json`

This manifest lists the model artifacts expected by inference. It must match the actual bundle names and schema from the same training run.

### Schema changes must be end-to-end

When changing a feature, target, category label, or output type, trace the change through:

```text
master dataset
   → preprocessing and split schema
   → training
   → model metadata/artifacts
   → inference features and decoding
   → design generation
   → BOM calculator
   → PDF/report output
```

Updating only one stage can lead to missing-column exceptions or incorrect-looking outputs.

---

## 16. Troubleshooting

| Symptom | Likely cause | Recommended check |
|---|---|---|
| Model directory not found | Artifacts missing from the expected path | Copy the model directory into `models/truss_hybrid_v4/` |
| Metadata missing or invalid | Incomplete model export | Restore matching `pipeline_metadata.json` |
| Missing `.joblib` bundle | Metadata and artifact files do not match | Compare `model_files` in metadata against actual filenames |
| Joblib/scikit-learn load error | Dependency-version mismatch or corrupt file | Use compatible training versions; verify artifact integrity |
| Required CSV not found | Wrong working directory or missing dataset | Run from project root and check `DATASET/` |
| Missing feature/target column | Dataset/schema mismatch | Compare CSV headers with `files/truss_common.py` |
| Weak rare-class results | Imbalance, few examples, or distribution shift | Inspect macro-F1 and per-class metrics |
| Inconsistent downstream fields | Sequential prediction error propagation | Inspect predictions stage-by-stage and review guardrails |
| Incorrect BOM quantity | Unit, geometry, count, catalogue, or rounding issue | Trace output through `main/calculator.py` |
| PDF export error | Missing dependency or result-key mismatch | Inspect traceback and expected report fields |
| Streamlit port occupied | Another server is running | Use `--server.port 8502` |

### Good debugging practice

Record the exact input, expected result, actual result, raw model prediction, post-guardrail value, model version, package versions, and relevant traceback. Fix the stage that introduces the error instead of only patching the displayed symptom.

---

## 17. Security, Reproducibility, and Engineering Limitations

### Security

- Only load trusted `.joblib` files; pickle-based deserialization can execute code.
- Do not commit credentials or private user designs.
- Review what is stored in result/session history before sharing or hosting the app.
- Add authentication and access controls before exposing the application publicly.

### Reproducibility

Keep the following together for each released model version:

- Master dataset version/checksum.
- Preprocessing code and split files.
- Python and package versions.
- Training configuration and random seed.
- Model bundles and matching metadata.
- Validation/test metrics and prediction outputs.
- Known limitations and code revision.

### Engineering limitations

The models learn patterns from the supplied dataset. Inputs outside its range or distribution may produce unreliable predictions. Sequential models can propagate earlier errors, and guardrails only enforce the rules explicitly implemented in the code.

High test metrics do not prove structural safety. The application is not a substitute for code-compliant structural analysis, professional engineering judgment, or approval by a qualified structural engineer.

---

## 18. License

The supplied ZIP did not contain a `LICENSE` file. Add a license before publishing the repository and choose terms that are compatible with your source code, datasets, reference drawings, and third-party assets. Do not claim that the project is MIT-licensed until the corresponding license file is included.

---

**TRUSS AI Design Studio — ML-assisted truss configuration, candidate visualization, and preliminary material estimation.**
