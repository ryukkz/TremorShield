"""
TREMORSHIELD — validation stage entry point.

    python run_validation.py \
        --train data/final/train_combined.csv \
        --test data/final/test_combined.csv \
        --metadata data/final/tremor_metadata.csv \
        --outdir data/final/validation \
        --robustpdx path/to/robustpdx.csv   (optional)

Runs, in order:
    1. Part 1 — mathematical/signal validation      (src/validation_signal.py)
    2. Part 2a — spatial validation (own data)       (src/validation_spatial.py)
    3. Part 2b — RobustPDX spatial reference         (src/validation_robustpdx.py)
    4. Statistical comparison                        (src/validation_statistics.py)
    5. validation_report.md                          (src/validation_report.py)

Deliberately does NOT touch or re-run the injection pipeline (run_pipeline.py)
— it only reads the CSVs that pipeline already produced.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config import TremorConfig
from src.validation_signal import run_signal_validation
from src.validation_spatial import run_spatial_validation
from src.validation_robustpdx import RobustPDXColumnMap, run_robustpdx_validation
from src.validation_statistics import run_statistical_comparison
from src.validation_report import generate_report


def main():
    parser = argparse.ArgumentParser(description="TREMORSHIELD synthetic-tremor validation")
    parser.add_argument("--train", default="data/final/train_combined.csv")
    parser.add_argument("--test", default="data/final/test_combined.csv")
    parser.add_argument("--metadata", default="data/final/tremor_metadata.csv")
    parser.add_argument("--outdir", default="data/final/validation")
    parser.add_argument("--seed", type=int, default=42)

    # RobustPDX is entirely optional — see src/validation_robustpdx.py for
    # why we never guess at its column layout.
    parser.add_argument("--robustpdx", default=None, help="Path to RobustPDX CSV (optional)")
    parser.add_argument("--robustpdx-x-col", default="x")
    parser.add_argument("--robustpdx-y-col", default="y")
    parser.add_argument("--robustpdx-trace-id-col", default="trace_id")
    parser.add_argument("--robustpdx-group-col", default="group")
    parser.add_argument("--robustpdx-reference-x-col", default=None)
    parser.add_argument("--robustpdx-reference-y-col", default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = TremorConfig()  # default frequencies/amplitudes; only used for reporting/plots

    print("=" * 70)
    print("TREMORSHIELD VALIDATION")
    print("=" * 70)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    meta_df = pd.read_csv(args.metadata)
    rows_df = pd.concat([train_df, test_df], ignore_index=True)

    dataset_summary = {
        "train rows": f"{len(train_df):,}",
        "test rows": f"{len(test_df):,}",
        "total rows validated": f"{len(rows_df):,}",
        "trials in metadata": len(meta_df),
        "tremor trials in metadata": int((meta_df.tremor_status == 1).sum()),
        "clean trials in metadata": int((meta_df.tremor_status == 0).sum()),
    }
    print(f"\n[run_validation] {dataset_summary}\n")

    # 1. Mathematical / signal validation ------------------------------------
    signal_summary = run_signal_validation(
        rows_df, meta_df, cfg.frequencies_hz, cfg.amplitudes_px, outdir, seed=args.seed
    )

    # 2a. Spatial validation (own data) --------------------------------------
    spatial_df = run_spatial_validation(rows_df, meta_df, outdir)

    # 2b. RobustPDX spatial reference (optional) -----------------------------
    colmap = RobustPDXColumnMap(
        path=args.robustpdx,
        x_col=args.robustpdx_x_col,
        y_col=args.robustpdx_y_col,
        trace_id_col=args.robustpdx_trace_id_col,
        group_col=args.robustpdx_group_col,
        reference_x_col=args.robustpdx_reference_x_col,
        reference_y_col=args.robustpdx_reference_y_col,
    )
    robustpdx_df = run_robustpdx_validation(colmap, outdir)

    # 3. Statistical comparison ------------------------------------------------
    comparison_df = run_statistical_comparison(spatial_df, robustpdx_df, outdir)

    # 4. Report -----------------------------------------------------------------
    generate_report(outdir, dataset_summary, signal_summary, spatial_df,
                     robustpdx_df, comparison_df, cfg)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
