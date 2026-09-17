"""
diagnose_quality_flags.py — run this against train_combined.csv / test_combined.csv
to see exactly WHY validate_dataset() flagged missing values and duplicate rows.

Usage:
    python diagnose_quality_flags.py data/final/train_combined.csv
"""

import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "data/final/train_combined.csv"
df = pd.read_csv(path)
print(f"Loaded {path}: {len(df):,} rows, {df.shape[1]} columns\n")

# --- 1. Missing values, broken down by column -----------------------------
print("=" * 70)
print("1. MISSING VALUES BY COLUMN (only columns with >0 missing)")
print("=" * 70)
na = df.isna().sum()
na = na[na > 0].sort_values(ascending=False)
print(na.to_string())
print(f"\nTotal missing across these columns: {na.sum():,}")

# --- 2. Where those missing values live (event type / tremor_status) ------
print("\n" + "=" * 70)
print("2. WHICH ROWS ARE MISSING SOMETHING (by event, tremor_status)")
print("=" * 70)
if "event" in df.columns and "tremor_status" in df.columns:
    missing_rows = df[df.isna().any(axis=1)]
    print(missing_rows.groupby(["event", "tremor_status"]).size().to_string())

# --- 2b. Is it just "first row of every trial"? ----------------------------
if {"user_id", "session_id", "trial_id"}.issubset(df.columns):
    first_rows = df.groupby(["user_id", "session_id", "trial_id"]).head(1)
    first_row_idx = set(first_rows.index)
    non_first_missing = df[df.isna().any(axis=1)].index.difference(first_row_idx)
    print(f"\nRows missing something that are NOT a trial's first row: "
          f"{len(non_first_missing):,}")
    print("(if this is ~0, your missing values are just the expected "
          "first-row-per-trial + clean-trial tremor-metadata pattern)")

# --- 3. Duplicate rows: what are they? -------------------------------------
print("\n" + "=" * 70)
print("3. DUPLICATE ROWS (same user_id/session_id/trial_id/elapsed_sec/tremor_status)")
print("=" * 70)
key = ["user_id", "session_id", "trial_id", "elapsed_sec", "tremor_status"]
key = [k for k in key if k in df.columns]
dup_mask = df.duplicated(subset=key, keep=False)
dups = df[dup_mask].sort_values(key)
print(f"Total duplicate rows: {dup_mask.sum():,}\n")

if "event" in df.columns:
    print("Duplicates by event type:")
    print(dups["event"].value_counts().to_string())

print("\nFirst 10 duplicate rows (to eyeball what's colliding):")
show_cols = key + (["event"] if "event" in df.columns else [])
print(dups[show_cols].head(10).to_string())