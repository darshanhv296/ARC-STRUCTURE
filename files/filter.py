"""
Remove EXACT duplicate design rows from the TRUSS MASTER dataset.

Important:
- Keeps the first occurrence of each fully identical row.
- Does NOT remove different valid design alternatives that share the same
  14 input features but have different structural outputs.
- Does NOT overwrite the source file unless --inplace is explicitly supplied.
- Writes a duplicate audit report.

Examples:
    python filter_master_duplicates.py
    python filter_master_duplicates.py --input "C:\\TRUSS\\MASTER.csv"
    python filter_master_duplicates.py --input "C:\\TRUSS\\MASTER.csv" --output "C:\\TRUSS\\MASTER_unique.csv"
    python filter_master_duplicates.py --input "C:\\TRUSS\\MASTER.csv" --inplace

Install:
    pip install pandas
"""

from pathlib import Path
import argparse
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Remove exact duplicate rows from a TRUSS master CSV."
    )
    parser.add_argument(
        "--input",
        default=r"C:\TRUSS\Dataset\master_dataset.csv",
        help="Path to the original master CSV (default: C:\\TRUSS\\Dataset\\master_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: <input_stem>_unique.csv",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Replace the input CSV after creating a backup.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {input_path}\n"
            "Run with --input followed by the real path to your master CSV."
        )

    df = pd.read_csv(input_path, low_memory=False)
    if df.empty:
        raise ValueError(f"The input dataset is empty: {input_path}")

    row_count_before = len(df)

    # Exact duplicate means every column value matches. Keep the first row.
    duplicate_mask = df.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    unique_df = df.loc[~duplicate_mask].copy()
    removed_count = row_count_before - len(unique_df)

    if args.inplace:
        backup_path = input_path.with_name(input_path.stem + "_backup" + input_path.suffix)
        if backup_path.exists():
            raise FileExistsError(
                f"Backup already exists, refusing to overwrite it: {backup_path}"
            )
        df.to_csv(backup_path, index=False)
        output_path = input_path
    else:
        output_path = (
            Path(args.output).expanduser()
            if args.output
            else input_path.with_name(input_path.stem + "_unique" + input_path.suffix)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique_df.to_csv(output_path, index=False)

    report = {
        "input_file": str(input_path.resolve()),
        "output_file": str(output_path.resolve()),
        "rows_before": row_count_before,
        "exact_duplicate_rows_removed": removed_count,
        "rows_after": len(unique_df),
        "columns_checked": len(df.columns),
        "duplicate_definition": "all columns identical",
        "kept": "first occurrence",
        "same_inputs_different_outputs_removed": False,
    }

    report_path = output_path.with_name(output_path.stem + "_duplicate_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("TRUSS MASTER DATASET — EXACT DUPLICATE FILTER")
    print("=" * 72)
    print(f"Input file:              {input_path.resolve()}")
    print(f"Rows before:             {row_count_before:,}")
    print(f"Exact duplicate rows:    {removed_count:,}")
    print(f"Rows after:              {len(unique_df):,}")
    print(f"Columns checked:         {len(df.columns)}")
    print(f"Unique dataset saved:    {output_path.resolve()}")
    print(f"Audit report saved:      {report_path.resolve()}")
    if args.inplace:
        print(f"Original backup saved:   {backup_path.resolve()}")
    print("-" * 72)
    print("NOTE: Different designs for the same inputs were preserved.")
    print("=" * 72)


if __name__ == "__main__":
    main()
