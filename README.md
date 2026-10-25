
```markdown
# TRUSS AI — AI-Powered Structural Design, 3D Modeling & Preliminary Safety Analysis

> **TRUSS AI Design Studio** is a machine-learning-assisted application for exploring truss families, predicting structural configuration fields, visualizing candidate layouts, and estimating preliminary material quantities from building/site inputs.

> ⚠️ **Important Engineering Notice:** This project is a design-assistance and prototyping tool. ML predictions, guardrails, drawings, and BOM estimates **do not certify structural safety**. Any design intended for construction or fabrication must be independently checked and approved by a qualified structural engineer under applicable codes.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Repository Structure](#repository-structure)
5. [End-to-End Execution Flow](#end-to-end-execution-flow)
6. [Machine Learning Design](#machine-learning-design)
7. [Model Artifacts](#model-artifacts)
8. [Dataset & Preprocessing](#dataset--preprocessing)
9. [Installation & Dependencies](#installation--dependencies)
10. [Quick Start](#quick-start)
11. [Training Pipeline](#training-pipeline)
12. [Design Generation & BOM](#design-generation--bom)
13. [Results & Reporting](#results--reporting)
14. [Configuration & Data Contracts](#configuration--data-contracts)
15. [Troubleshooting](#troubleshooting)
16. [Security & Reproducibility](#security--reproducibility)
17. [Development Workflow](#development-workflow)
18. [License](#license)

---

## 🎯 Project Overview

TRUSS AI connects a web-based design interface with a target-specific machine-learning pipeline and deterministic application logic. A user supplies building/site details; the application predicts supported design fields, applies compatibility checks, generates a candidate representation, calculates preliminary quantities, and provides results for review.

The project intentionally separates three responsibilities:

| Layer | Responsibility | Key Files |
|---|---|---|
| **ML Prediction** | Estimates fields from patterns in training data | `files/train_truss.py`, `main/model.py` |
| **Design & Rules** | Compatibility checks, guardrails, geometry, BOM | `main/design.py`, `main/calculator.py` |
| **Presentation** | Streamlit UI, visualization, PDF export | `main/main.py`, `main/pdf_export.py` |

> ⚠️ **A successful prediction is not proof of a safe structure.** The model learns only from the dataset and features provided; the rules can only enforce conditions explicitly implemented in code.

### Intended Users & Uses

- Students and developers exploring ML-assisted structural-design workflows
- Engineers reviewing preliminary design alternatives
- Researchers evaluating target-specific ML models against held-out data
- Users comparing candidate truss configurations and preliminary BOM quantities

**Not intended for:** Code-compliant structural analysis, connection design, material certification, or professional approval.

---

## ✨ Key Features

### 🤖 AI-Powered Design Prediction
- Predicts truss family, rise height, pipe sizes, spacings, and member counts
- Sequential inference pipeline: family → rise → downstream targets
- Supports Random Forest, XGBoost, and CatBoost estimators

### 🏗️ Design Generation & Validation
- Generates candidate structural layouts from ML predictions
- Applies compatibility checks and configurable guardrails
- Supports multiple truss families (Box, A-Frame, Zig-Zag, Semi, etc.)

### 📊 Visualization & Reporting
- Interactive Streamlit dashboard
- Truss/building visualizations
- Preliminary Bill of Materials (BOM) calculation
- PDF report generation

### 🔬 Training & Evaluation Pipeline
- Deterministic data splitting with grouped leakage prevention
- Per-target metrics for classification and regression
- Validation/test split workflow for model selection

---

## 🏗️ System Architecture

```text
                         TRUSS AI DESIGN STUDIO
                                  |
                                  v
                         Streamlit UI
                           main/main.py
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
           Input validation          Session state mgmt
                    |
                    v
             main/model.py
         Load model catalog/metadata
                    |
                    v
          Sequential ML inference
    +------------+-------------------+
    |                                |
    v                                v
Truss family                    Rise height
    |                                |
    +----------------+---------------+
                     |
                     v
          Target-specific design models
+-------+-------+-------+-------+-------+-------+
|       |       |       |       |       |       |
v       v       v       v       v       v       v
Pipe   Member  Truss  Column  Beam   Purlin  Spacing
sizes   counts spacing layout quantities /
                                    orientation
                     |
                     v
          Design compatibility/guardrails
             main/design.py
                     |
           +---------+----------+
           |                    |
           v                    v
    Design visualization   BOM calculation
                           main/calculator.py
           |                    |
           +---------+----------+
                     |
                     v
      Result history and PDF export
main/result_history.py / main/pdf_export.py
```

### Training Pipeline

```text
DATASET/truss_ai_complete_dataset_10000_purlin_remainder_fixed.csv
                              |
                              v
                    files/preprocess_truss.py
                              |
               +--------------+--------------+
               |              |              |
               v              v              v
         train.csv      validation.csv     test.csv
               |              |              |
               v              v              |
             files/train_truss.py            |
               |                             |
               v                             |
        models/truss_hybrid_v4/               |
   target-specific .joblib bundles            |
               |                             |
               +--------------+--------------+
                              |
                              v
          files/validate_truss.py & test_truss.py
                              |
                              v
                  results/truss_hybrid_v4/
             metrics, predictions, summary
```

---

## 📂 Repository Structure

```text
TRUSS/
│
├── 📁 DATASET/
│   └── truss_ai_complete_dataset_10000_purlin_remainder_fixed.csv
│       # Preferred master dataset for model preprocessing and training.
│       # Contains source design inputs and target/output fields.
│       # Preserve a backup before cleaning, filtering, or regenerating.
│
├── 📁 DESIGNS/
│   ├── BOX.png                          # Box Truss reference illustration
│   ├── BOX_TRUSS_1SIDE.png              # One-side Box Truss variant
│   ├── FULL A TRUSS.png                 # Full-A truss reference
│   ├── HALF_A.png                       # Half-A truss variant
│   ├── HALF_A_TRUSS.png                 # Additional Half-A truss reference
│   ├── SEMI_TRUSS.png                   # Semi Truss configuration
│   ├── ZIG-ZAG-A.png                    # Zig-Zag-A layout
│   ├── ZIGZAG_BEAM_1SIDE.png            # One-side Zig-Zag beam layout
│   ├── ZIG_ZAG_BOX.png                  # Zig-Zag Box configuration
│   └── ZIG_ZAG_HALF_A.png               # Zig-Zag Half-A configuration
│
├── 📁 files/
│   ├── filter.py
│   │   # Dataset filtering/inspection utility.
│   │   # Check CLI arguments before running; preserve original dataset.
│   │
│   ├── preprocess_truss.py
│   │   # Validates master CSV and creates deterministic data splits.
│   │   # Checks required fields, numeric values, counts, and target data.
│   │   # Writes train.csv, validation.csv, test.csv, columns.json, and
│   │   # preprocessing_report.json into preprocessed/.
│   │   # Uses grouped splitting to reduce related-record leakage.
│   │
│   ├── train_truss.py
│   │   # Trains family, rise, and downstream design-target models.
│   │   # Compares candidate estimators using validation data.
│   │   # Supports Random Forest, XGBoost, CatBoost, and constant baselines.
│   │   # Saves target-specific model bundles and pipeline metadata.
│   │
│   ├── validate_truss.py
│   │   # Evaluates saved model catalog on validation.csv.
│   │   # Produces per-target metrics and prediction output files.
│   │
│   ├── test_truss.py
│   │   # Evaluates saved model catalog against held-out test.csv.
│   │   # Produces final metrics; avoid tuning against test results.
│   │
│   └── truss_common.py
│       # Shared ML schema and helpers for preprocessing/training.
│       # Defines paths, random seed, base features, targets,
│       # categorical targets, split/model columns, and utilities.
│
├── 📁 main/
│   ├── calculator.py
│   │   # Calculates preliminary BOM/material quantities.
│   │   # Uses generated design info and pipe catalogue.
│   │
│   ├── config.py
│   │   # Central path configuration: PROJECT_ROOT, MODEL_DIR,
│   │   # RESULTS_DIR, PREPROCESSED_DIR.
│   │
│   ├── design.py
│   │   # Design generation, compatibility checks, and drawing helpers.
│   │   # Where model outputs become candidate layouts.
│   │
│   ├── design_options.py
│   │   # Evaluates supported design alternatives for supplied inputs.
│   │
│   ├── main.py
│   │   # Streamlit application entry point and UI orchestration.
│   │   # Collects inputs, loads predictions, generates designs,
│   │   # calculates BOM, presents visualizations, exports results.
│   │
│   ├── master datatset.csv
│       # Master/reference CSV (note spelling inconsistency in filename).
│       # Compare schemas with DATASET/ before using for training.
│   │
│   ├── model.py
│   │   # Loads saved target-specific model bundles.
│   │   # Sequential inference: family → rise → downstream fields.
│   │
│   ├── pdf_export.py
│   │   # Builds PDF report from generated design and BOM data.
│   │
│   ├── pipes.csv
│   │   # Pipe/section catalogue used by BOM and design logic.
│   │
│   ├── result_history.py
│   │   # Cache-key, exact-match, and similar-result helpers.
│   │
│   ├── run_regression.py
│       # Regression/smoke-test utility for the application pipeline.
│   │
│   ├── truss_common.py
│       # Application-side shared utilities for design/inference.
│       # Separate from files/truss_common.py — keep both in sync.
│   │
│   └── reference_designs/
│       ├── BOX.png
│       ├── BOX_TRUSS_1SIDE.png
│       ├── FULL A TRUSS.png
│       ├── HALF_A.png
│       └── SEMI_TRUSS.png
│
├── 📁 preprocessed/
│   ├── columns.json                    # Schema/column metadata
│   ├── preprocessing_report.json       # Preprocessing summary
│   ├── reference_master_32_columns.csv  # 32-column reference schema
│   ├── train.csv                       # Training split
│   ├── validation.csv                  # Validation split
│   └── test.csv                        # Held-out test split
│
├── 📁 results/
│   └── truss_hybrid_v4/
│       ├── test_metrics.json            # Per-target test metrics
│       ├── test_predictions.csv         # Test predictions
│       ├── training_summary.json        # Training run summary
│       ├── validation_metrics.json      # Validation metrics
│       └── validation_predictions.csv   # Validation predictions
│
├── 📁 models/
│   └── truss_hybrid_v4/                 # Model artifacts (not in source ZIP)
│       ├── beam_pipe_size.joblib
│       ├── beam_total_count.joblib
│       ├── column_pipe_size.joblib
│       ├── column_spacing_across_span_ft.joblib
│       ├── column_spacing_along_length_ft.joblib
│       ├── column_total_count.joblib
│       ├── frame_chord_pipe_size.joblib
│       ├── purlin_orientation.joblib
│       ├── purlin_pipe_size.joblib
│       ├── purlin_spacing_ft.joblib
│       ├── rise_height_ft.joblib
│       ├── truss_family.joblib
│       ├── truss_spacing_ft.joblib
│       ├── truss_total_count.joblib
│       ├── web_intermediate_member_count.joblib
│       ├── web_pipe_size.joblib
│       ├── web_spacing_ft.joblib
│       ├── web_total_member_count.joblib
│       └── pipeline_metadata.json        # Artifact manifest
│
└── README.md                           # This file
```

> **Note:** `models/truss_hybrid_v4/` appears in the supplied screenshot but was **not present in the source ZIP**. Copy the correct, trusted model artifacts to the expected path before running inference.

---

## 🔁 End-to-End Execution Flow

### Preprocessing & Training

| Step | Command | Key Outputs |
|---|---|---|
| 1. Preprocess | `python files/preprocess_truss.py` | `train.csv`, `validation.csv`, `test.csv`, `columns.json`, `preprocessing_report.json` |
| 2. Train | `python files/train_truss.py` | Model bundles + `pipeline_metadata.json` |
| 3. Validate | `python files/validate_truss.py` | Validation metrics + predictions |
| 4. Test | `python files/test_truss.py` | Test metrics + predictions |

### User Inference Flow

1. User opens Streamlit app (`python main/main.py`)
2. UI gathers site dimensions and design inputs
3. `main/model.py` loads model catalog and validates artifact list
4. **Stage 1:** Truss-family model predicts `truss_family`
5. **Stage 2:** Rise-height model uses base inputs + predicted family → `rise_height_ft`
6. **Stage 3:** Downstream models predict pipe sizes, spacings, counts
7. Design-generation and compatibility logic creates candidate layout (`main/design.py`)
8. Calculator estimates preliminary material quantities (`main/calculator.py`)
9. UI displays output; PDF report can be generated (`main/pdf_export.py`)
10. Result-history helpers support retrieval of matching/similar previous outputs

> ⚠️ **Error Propagation Warning:** If the family prediction is wrong, downstream predictions can also be affected. Validate the complete pipeline, not just each estimator in isolation.

---

## 🤖 Machine Learning Design

### Base Features

The shared training schema in `files/truss_common.py` defines:

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

> **Note:** `roof_configuration_detail` may be represented as `"Unknown"` if absent from the source CSV. The `overhang_ft` field is deliberately excluded from model features in this schema.

### Prediction Stages

| Stage | Target | Feature Dependency |
|---|---|---|
| 1 | `truss_family` | Base features only |
| 2 | `rise_height_ft` | Base features + predicted family |
| 3 | Design targets | Base features + predicted family + predicted rise |

### Downstream Design Targets

| Category | Targets |
|---|---|
| **Pipe Sizes** | `frame_chord_pipe_size`, `web_pipe_size`, `column_pipe_size`, `beam_pipe_size`, `purlin_pipe_size` |
| **Spacings** | `web_spacing_ft`, `truss_spacing_ft`, `column_spacing_along_length_ft`, `column_spacing_across_span_ft`, `purlin_spacing_ft` |
| **Counts** | `web_intermediate_member_count`, `web_total_member_count`, `truss_total_count`, `column_total_count`, `beam_total_count` |
| **Other** | `purlin_orientation` |

### Candidate Estimators

The training code supports:
- **Random Forest** — primary candidate for classification and regression
- **XGBoost** — optional when package is installed
- **CatBoost** — optional when package is installed
- **Constant Baseline** — for comparison

> **Classification** is suitable for categorical outputs (truss family, pipe-size labels). **Regression** is suitable for continuous numeric outputs (heights, spacings). Discrete numeric choices may require classification or explicit rounding/guardrails.

### Evaluation Metrics

| Target Type | Metrics |
|---|---|
| Classification | Accuracy, Precision, Recall, Macro-F1, Weighted-F1 |
| Regression | MAE, RMSE |

> ⚠️ An aggregate score can hide weak performance on rare truss families or pipe-size categories. Always inspect per-class and per-target results.

---

## 📦 Model Artifacts

The `models/truss_hybrid_v4/` directory contains 18 `.joblib` files and one metadata file:

| Artifact | Predicts |
|---|---|
| `truss_family.joblib` | Truss family category |
| `rise_height_ft.joblib` | Rise height in feet |
| `frame_chord_pipe_size.joblib` | Frame-chord pipe size |
| `web_pipe_size.joblib` | Web-member pipe size |
| `web_spacing_ft.joblib` | Web spacing |
| `web_intermediate_member_count.joblib` | Intermediate web-member count |
| `web_total_member_count.joblib` | Total web-member count |
| `truss_spacing_ft.joblib` | Spacing between trusses |
| `truss_total_count.joblib` | Total truss count |
| `column_pipe_size.joblib` | Column pipe size |
| `column_total_count.joblib` | Total column count |
| `column_spacing_along_length_ft.joblib` | Column spacing along length |
| `column_spacing_across_span_ft.joblib` | Column spacing across span |
| `beam_pipe_size.joblib` | Beam pipe size |
| `beam_total_count.joblib` | Total beam count |
| `purlin_pipe_size.joblib` | Purlin pipe size |
| `purlin_orientation.joblib` | Purlin orientation |
| `purlin_spacing_ft.joblib` | Purlin spacing |
| `pipeline_metadata.json` | Artifact manifest & inference metadata |

### Loading Requirements

- Directory path must match `main/config.py`
- Metadata must reference model files that actually exist
- All bundles must come from the same compatible training run
- Python/scikit-learn versions should match the training environment
- Only load trusted `.joblib` files (pickle-based serialization can execute code)

---

## 📊 Dataset & Preprocessing

### Master Dataset

**Preferred source:**
```
DATASET/truss_ai_complete_dataset_10000_purlin_remainder_fixed.csv
```

The filename suggests 10,000 rows — confirm the actual count from the file. The preprocessing script can look for alternative matching CSVs if the preferred file is absent.

### Preprocessing Checks

- Required feature and target columns are present
- Numeric columns parse correctly
- Dimensions and spacing values use consistent units
- Counts are valid and non-negative
- Pipe-size labels match expected catalogue categories
- Categorical labels are normalized consistently
- Related/duplicate designs are not split in a way that leaks records across train/test
- No input feature accidentally encodes the target value

### Grouped Splitting

Grouped splitting keeps related examples together to prevent near-duplicate designs from appearing in both training and testing. Inspect the implementation and preprocessing report to understand the grouping key used for the current dataset.

### Never Silently Fabricate Missing Labels

A missing target should be fixed at source or handled through a documented filtering/imputation strategy. Filling a missing target with an invented structural value can teach the model incorrect relationships.

---

## 🚀 Installation & Dependencies

### Recommended Environment

```bash
# Python 3.8+ required
python --version
```

### Windows PowerShell

```powershell
# Create and activate virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install core dependencies
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn joblib

# Optional: for XGBoost/CatBoost candidate estimators
python -m pip install xgboost catboost
```

### Linux / macOS

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install core dependencies
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn joblib

# Optional: for XGBoost/CatBoost candidate estimators
python -m pip install xgboost catboost
```

### Dependency Reproducibility

After confirming the project works in a clean environment, create a `requirements.txt` with exact tested package versions. This is especially important for scikit-learn model artifacts — serialized pipelines depend on internal class names and version-specific behavior.

---

## 🎯 Quick Start

### Prerequisites

Before launching, verify these paths exist:

```text
main/config.py
main/model.py
main/design.py
main/calculator.py
main/pipes.csv
files/truss_common.py
models/truss_hybrid_v4/pipeline_metadata.json
models/truss_hybrid_v4/*.joblib
```

### Launch the Design Studio

From the project root:

```bash
# Method 1: Direct execution
python main/main.py

# Method 2: Via Streamlit CLI
python -m streamlit run main/main.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

### Browser Session Files

The UI creates `results/browser_sessions/` to preserve a browser's latest design state across refreshes. These are runtime files, not source files — do not publish session data containing private designs or user-supplied information.

---

## 🔄 Training Pipeline

### Step 1 — Preprocess

```bash
python files/preprocess_truss.py
```

**Expected outputs:**
```text
preprocessed/train.csv
preprocessed/validation.csv
preprocessed/test.csv
preprocessed/columns.json
preprocessed/preprocessing_report.json
```

Review errors and warnings before training. Confirm split sizes, target coverage, and data quality.

### Step 2 — Train

```bash
python files/train_truss.py
```

Inspect available options:
```bash
python files/train_truss.py --help
```

Training fits candidate estimators and selects winners using validation data. The output path is controlled by shared configuration — confirm it agrees with `main/config.py`.

### Step 3 — Validate

```bash
python files/validate_truss.py
```

Review per-target validation metrics and prediction output. Use this stage to identify poor classes, high-error targets, and schema problems.

### Step 4 — Test

```bash
python files/test_truss.py
```

The test script evaluates the saved model catalog on the held-out test split. **Do not repeatedly adjust models against test metrics and then continue describing that same set as an untouched test set.**

### Step 5 — Optional Filtering

```bash
python files/filter.py --help
```

Read the script's supported arguments before using it. Keep the original dataset and document any filters applied.

### Recommended Release Procedure

1. Back up the master dataset
2. Record code revision and dataset version
3. Run preprocessing and inspect its report
4. Train the model catalog
5. Validate per-target performance
6. Run held-out testing
7. Preserve exact model artifacts and metadata from the same run
8. Smoke-test inference through `main/model.py` and the Streamlit UI
9. Test representative design and BOM boundary cases
10. Document known limitations before sharing the release

---

## 🏗️ Design Generation & BOM

### Design Generation

`main/design.py` turns input fields and predictions into a candidate design representation. It contains supported-family compatibility checks and drawing helpers. `main/design_options.py` evaluates available alternatives.

> Geometry changes should be tested on small, typical, large, and boundary-size sites. Include cases with/without center support and each supported roofing/truss configuration.

### Pipe Catalogue

`main/pipes.csv` is the pipe catalogue used by the application. Preserve column names expected by the calculator and design code. Confirm dimensions, wall thicknesses, section labels, and weights against reliable supplier/manufacturer data before using catalogue values for fabrication decisions.

### BOM Calculation

`main/calculator.py` calculates preliminary material quantities. Verify all of the following before using a BOM commercially:

- Member lengths and counts
- Unit conversions (feet, inches, millimetres)
- Cutting allowance and kerf assumptions
- Stock-length selection and remainder handling
- Waste/reserve allowances
- Pipe/section labels and catalogue weights
- Duplicate or missing members in the generated layout

### PDF Report

`main/pdf_export.py` builds a PDF from the data passed by the UI. If a design/result dictionary changes, update and test the report fields as well.

> ⚠️ **Structural verification is separate.** ML accuracy and BOM arithmetic do not establish structural adequacy. A real design must consider loads, buckling, deflection, stability, bracing, member slenderness, connections, welds/bolts, columns, anchors, foundations, material grade, corrosion, wind uplift, seismic conditions, and local codes.

---

## 📊 Results & Reporting

### Model Evaluation Results

```text
results/truss_hybrid_v4/
├── training_summary.json      # Training-run summary
├── validation_metrics.json    # Validation performance by target
├── validation_predictions.csv # Validation predictions for error inspection
├── test_metrics.json          # Test performance by target
└── test_predictions.csv       # Held-out predictions
```

These are run outputs. If you retrain, confirm these files correspond to the same model version and dataset split you intend to report.

### Application History

`main/result_history.py` contains cache-key, exact-match, and similar-result helpers. The application also writes browser-session state under `results/browser_sessions/`.

> History is an application convenience, not a design approval record.

---

## ⚙️ Configuration & Data Contracts

### `main/config.py`

| Setting | Value |
|---|---|
| `PROJECT_ROOT` | Repository root (resolved from `main/config.py`) |
| `MODEL_DIR` | `models/truss_hybrid_v4` |
| `RESULTS_DIR` | `results/truss_hybrid_v4` |
| `PREPROCESSED_DIR` | `preprocessed` |

### `files/truss_common.py` (Training Side)

Defines base features, target fields, categorical target handling, split/model columns, shared preprocessing/estimator helpers, paths, and random seed.

### `main/truss_common.py` (Application Side)

Distinct from `files/truss_common.py`. Do not assume interchangeability. When changing feature names or inference helpers, inspect both files and update the appropriate contracts.

### `pipeline_metadata.json`

Manifest used by the inference loader to identify expected model bundles. Must match actual files and schema for the same training run.

### Version Changes

When changing a target name, category encoding, feature schema, or model output format, update and test the **complete path**:

```text
dataset → preprocessing → training → metadata → model loader
        → inference → design generation → BOM → PDF report
```

Changing only one layer can produce runtime errors or plausible-looking but incorrect outputs.

---

## 🔍 Troubleshooting

| Problem | Likely Cause | Check |
|---|---|---|
| Model directory missing | Artifacts not copied into project | Create `models/truss_hybrid_v4/` with matching trusted bundles |
| `pipeline_metadata.json` missing | Incomplete model export | Copy metadata from same training run as bundles |
| `.joblib` fails to load | Version mismatch or corrupted artifact | Use compatible training-environment versions; verify file integrity |
| Missing feature/target columns | Dataset schema differs | Compare CSV headers with `files/truss_common.py` and preprocessing report |
| Wrong master CSV selected | Multiple candidate CSVs present | Keep intended master in `DATASET/`; verify preprocessing log |
| Poor family predictions | Imbalanced classes, weak features, or dataset shift | Inspect per-class metrics, confusion matrix, and validation/test distribution |
| Poor spacing predictions | Discrete target treated as continuous, noisy labels, or inconsistent units | Inspect target values and categorical/regression logic in shared schema |
| Downstream output inconsistent | Sequential prediction error propagation | Inspect raw predictions at each stage and check guardrails |
| BOM quantities look wrong | Unit conversion, geometry, catalogue, rounding, or count issue | Trace member lengths/counts through `main/calculator.py` |
| PDF export fails | Missing dependency or changed result fields | Inspect traceback and compare exporter expectations with design object |
| Port already in use | Another Streamlit instance running | Use `--server.port 8502` |
| Training is slow | Large estimators or many target models | Inspect `--help`, training config, CPU/RAM use, and optional estimator availability |

### Debugging Approach

For a reproducible bug, capture: exact input, target output, raw model prediction, post-guardrail value, relevant log, model version, and dependency versions. Avoid fixing a visible output symptom without finding which stage introduced it.

---

## 🔒 Security & Reproducibility

### Model & Data Security

- Treat `.joblib` artifacts as trusted executable serialization; never load unknown files
- Do not commit secrets, private user data, or confidential design files
- Review session/history persistence before public deployment
- Add authentication and access control if exposed beyond a trusted local environment

### Reproducibility Checklist

Keep together:

- [ ] Dataset version/checksum
- [ ] Preprocessing code revision and split outputs
- [ ] Python and dependency versions
- [ ] Training configuration and random seed
- [ ] Model artifacts and matching metadata
- [ ] Validation/test metrics and known limitations

### Engineering Limitations

The model learns relationships represented in the training dataset. Inputs outside the training distribution can produce unreliable outputs. Sequential inference can compound errors, and deterministic guardrails only enforce rules that have actually been implemented.

> ⚠️ **This software must not be used as the sole basis for construction, structural certification, or safety-critical decisions. A qualified structural engineer must verify the design and all assumptions before fabrication or construction.**

---

## 🛠️ Development Workflow

### When Modifying the Project

1. **Understand current behavior:** inspect relevant source file and its callers
2. **Define the change:** specify input/output contract and affected modules
3. **Preserve a baseline:** save current metrics and representative input/output cases
4. **Update the full pipeline:** schema, preprocessing, training, inference, UI, calculator, and report where applicable
5. **Test boundaries:** min/max dimensions, unsupported categories, missing values, rounding remainders
6. **Retrain when required:** changes to model features or target definitions require compatible model artifacts
7. **Run validation and testing:** inspect each target rather than relying on a single score
8. **Test complete UI workflow:** input → prediction → design → BOM → PDF/history
9. **Document the release:** list dataset/model versions, dependency versions, metrics, and known limitations

### Suggested Future Improvements

- Add a tested, pinned `requirements.txt`
- Add automated tests for data schema, geometry, BOM, and PDF export
- Add explicit model/dataset version information to each generated report
- Add out-of-distribution input warnings
- Add confidence/calibration estimates only where statistically justified
- Add engineering assumption and review-sign-off sections to reports
- Add secure authentication and deployment hardening if hosted publicly

---

## 📝 License

No license file was present in the supplied ZIP. Before publishing the repository, add a `LICENSE` file and choose terms appropriate for the project, datasets, drawings, and third-party assets. Do not claim a license badge or license grant until the corresponding license file is actually included.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Open an issue to discuss proposed changes
2. Fork the repository and create a feature branch
3. Follow the [development workflow](#development-workflow) when making changes
4. Ensure the full pipeline is tested (preprocessing → training → inference → design → BOM)
5. Update documentation to reflect any schema or behavior changes
6. Submit a pull request with a clear description of changes

---

**TRUSS AI — AI-assisted truss configuration, design visualization, and preliminary material estimation.**

```text
© 2026 TRUSS AI Design Studio. Not a substitute for professional structural engineering.
```
```

This is the complete, enhanced `README.md` with:

- ✅ Professional emoji-accented formatting
- ✅ Detailed table of contents with anchor links
- ✅ Comprehensive file-by-file repository structure
- ✅ Full system architecture diagrams
- ✅ Complete ML pipeline explanation
- ✅ All 18 model artifacts documented
- ✅ Step-by-step training and inference flows
- ✅ Troubleshooting table with specific solutions
- ✅ Security and reproducibility checklists
- ✅ Professional development workflow
- ✅ Clear engineering disclaimers throughout

Simply copy this entire block into your `README.md` file and it's ready for GitHub.
