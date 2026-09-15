"""
TREMORSHIELD — pipeline entry point.

    python run_pipeline.py --input data/raw/all_cleaned_data.csv --outdir data/final

Orchestrates, in order:
    1. load_data              (src/data_loader.py)
    2. rename_participants     (src/anonymize.py)
    3. split_participants      (src/splitting.py)   -- 80/20 by participant
    4. split_training_trials   (src/splitting.py)   -- 70/30 tremor/clean, by trial, within train
    5. build_condition x3      (src/pipeline.py)     -- train, TEST-A clean, TEST-B tremor
    6. validate_dataset        (src/validation.py)   -- leakage asserts + PSD sanity figure
    7. save_outputs            (src/io_writer.py)

All randomness is seeded (--seed) for reproducibility.
"""

import argparse
import os

import pandas as pd

from src.config import TremorConfig, TRIAL_KEYS
from src.data_loader import load_data
from src.anonymize import rename_participants
from src.splitting import split_participants, split_training_trials, assign_test_trials, _trial_table
from src.pipeline import build_condition
from src.validation import validate_dataset
from src.io_writer import save_outputs


def main():
    parser = argparse.ArgumentParser(description="TREMORSHIELD synthetic tremor injection")
    parser.add_argument("--input", default="data/raw/all_participants_cleaned.csv")
    parser.add_argument("--outdir", default="data/final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--train-tremor-frac", type=float, default=0.70)
    parser.add_argument("--test-tremor-frac", type=float, default=1.00)
    args = parser.parse_args()

    cfg = TremorConfig(
        seed=args.seed,
        train_frac=args.train_frac,
        train_tremor_frac=args.train_tremor_frac,
        test_tremor_frac=args.test_tremor_frac,
    )

    # 1-2. load + anonymize
    df_raw = load_data(args.input)
    df, _mapping = rename_participants(df_raw, cfg.seed)

    # 3. participant-level split
    train_users, test_users, participant_split = split_participants(
        df, cfg.seed, cfg.train_frac
    )
    df_train = df[df.participant_id.isin(train_users)]
    df_test = df[df.participant_id.isin(test_users)]

    # 4-5a. TRAIN: 70/30 tremor/clean split at trial level
    train_assignment = split_training_trials(df_train, cfg.seed, cfg.train_tremor_frac)
    train_combined, train_meta = build_condition(
        df_train, train_assignment, cfg, seed_offset=cfg.seed + 100
    )
    train_clean = train_combined[train_combined.tremor_status == 0].reset_index(drop=True)
    train_tremor = train_combined[train_combined.tremor_status == 1].reset_index(drop=True)

    # 5b. TEST: paired clean (TEST-A) + tremor (TEST-B) renderings
    test_trials_all = _trial_table(df_test)
    test_trials_all["tremor_status"] = 0
    test_clean, test_clean_meta = build_condition(
        df_test, test_trials_all, cfg, seed_offset=cfg.seed + 200
    )

    tremor_selection = assign_test_trials(df_test, cfg.test_tremor_frac)
    tremor_selection = tremor_selection.copy()
    tremor_selection["tremor_status"] = 1
    test_tremor, test_tremor_meta = build_condition(
        df_test, tremor_selection, cfg, seed_offset=cfg.seed + 300
    )

    test_combined = pd.concat([test_clean, test_tremor], ignore_index=True)
    test_meta = pd.concat([test_clean_meta, test_tremor_meta], ignore_index=True)

    # 6. validation
    validate_dataset(train_users, test_users, train_combined, test_combined,
                      train_meta, test_meta, cfg, args.outdir)

    tremor_metadata_all = pd.concat([train_meta, test_meta], ignore_index=True)

    # 7. save
    save_outputs(
        args.outdir,
        train_clean, train_tremor, train_combined,
        test_clean, test_tremor, test_combined,
        participant_split, tremor_metadata_all, cfg,
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Participants: {len(train_users)} train / {len(test_users)} test "
          f"(of {len(train_users)+len(test_users)} total)")
    print(f"Training trials: {len(train_meta)} "
          f"({(train_meta.tremor_status==1).sum()} tremor, "
          f"{(train_meta.tremor_status==0).sum()} clean)")
    print(f"Test trials (distinct): {test_meta[TRIAL_KEYS].drop_duplicates().shape[0]}, "
          f"rendered as {len(test_meta)} rows in metadata "
          f"(TEST-A clean + TEST-B tremor)")
    print(f"train_combined: {len(train_combined):,} rows")
    print(f"test_combined:  {len(test_combined):,} rows")
    print(f"Outputs written to: {os.path.abspath(args.outdir)}")
    print("Source file all_cleaned_data.csv was not modified.")


if __name__ == "__main__":
    main()
