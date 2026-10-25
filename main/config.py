from pathlib import Path
# Resolve the repository root from C:\TRUSS\main.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / 'models' / 'truss_hybrid_v4'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'truss_hybrid_v4'
PREPROCESSED_DIR = PROJECT_ROOT / 'preprocessed'
