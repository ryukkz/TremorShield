"""
Convert trial-level mouse trajectory data into fixed-duration windows.

Important:
- Windows never cross participant/session/trial boundaries.
- Uses elapsed_sec for time-based windowing.
- Keeps task/context labels.
- Keeps tremor metadata for later analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SEC = 0.50       # 500 ms
STEP_SEC = 0.25         # 250 ms = 50% overlap

GROUP_COLS = [
    "participant_id",
    "session_id",
    "trial_id",
]

# These are candidate ML features from the observed trajectory.
# We will verify the final feature list separately.
FEATURE_COLS = [
    "dx_observed",
    "dy_observed",
    "vx_observed",
    "vy_observed",
    "velocity_observed",
    "acceleration_observed",
    "direction_change_observed",
]


# ============================================================
# CREATE WINDOWS FOR ONE TRIAL
# ============================================================

def make_trial_windows(
    trial_df,
    window_sec=WINDOW_SEC,
    step_sec=STEP_SEC,
):
    """
    Convert one trial into overlapping time windows.

    Returns one row per window.
    """

    trial_df = trial_df.sort_values("elapsed_sec").reset_index(drop=True)

    if trial_df.empty:
        return []

    t = trial_df["elapsed_sec"].to_numpy(dtype=float)

    start_time = np.nanmin(t)
    end_time = np.nanmax(t)

    windows = []

    current_start = start_time

    while current_start + window_sec <= end_time + 1e-9:

        current_end = current_start + window_sec

        mask = (
            (t >= current_start)
            & (t < current_end)
        )

        window = trial_df.loc[mask].copy()

        # Require enough samples.
        if len(window) >= 3:

            windows.append(
                {
                    "window_start_sec": current_start,
                    "window_end_sec": current_end,
                    "window_df": window,
                }
            )

        current_start += step_sec

    return windows


# ============================================================
# CONVERT A WINDOW INTO ONE FEATURE ROW
# ============================================================

def summarize_window(window_df):
    """
    Convert all samples inside one window into one ML feature row.
    """

    row = {}

    # --------------------------------------------------------
    # Basic window metadata
    # --------------------------------------------------------

    row["participant_id"] = window_df["participant_id"].iloc[0]
    row["session_id"] = window_df["session_id"].iloc[0]
    row["trial_id"] = window_df["trial_id"].iloc[0]

    row["task"] = window_df["task"].iloc[0]

    # Keep these for analysis/validation.
    if "tremor_status" in window_df.columns:
        row["tremor_status"] = window_df["tremor_status"].iloc[0]

    if "tremor_frequency_hz" in window_df.columns:
        row["tremor_frequency_hz"] = window_df[
            "tremor_frequency_hz"
        ].iloc[0]

    if "tremor_amplitude_px" in window_df.columns:
        row["tremor_amplitude_px"] = window_df[
            "tremor_amplitude_px"
        ].iloc[0]

    # --------------------------------------------------------
    # Time information
    # --------------------------------------------------------

    row["window_start_sec"] = window_df["elapsed_sec"].iloc[0]
    row["window_end_sec"] = window_df["elapsed_sec"].iloc[-1]

    row["n_samples"] = len(window_df)

    # --------------------------------------------------------
    # Statistical features
    # --------------------------------------------------------

    for col in FEATURE_COLS:

        if col not in window_df.columns:
            continue

        values = window_df[col].to_numpy(dtype=float)

        values = values[np.isfinite(values)]

        if len(values) == 0:
            row[f"{col}_mean"] = np.nan
            row[f"{col}_std"] = np.nan
            row[f"{col}_min"] = np.nan
            row[f"{col}_max"] = np.nan
            continue

        row[f"{col}_mean"] = np.mean(values)
        row[f"{col}_std"] = np.std(values)
        row[f"{col}_min"] = np.min(values)
        row[f"{col}_max"] = np.max(values)

    return row


# ============================================================
# CONVERT COMPLETE DATASET INTO WINDOWS
# ============================================================

def create_windows(
    df,
    window_sec=WINDOW_SEC,
    step_sec=STEP_SEC,
):
    """
    Convert a complete trajectory dataset into window-level data.

    Windows are generated independently for every
    participant/session/trial.
    """

    results = []

    grouped = df.groupby(
        GROUP_COLS,
        sort=False,
        dropna=False,
    )

    total_trials = 0

    for _, trial_df in grouped:

        total_trials += 1

        windows = make_trial_windows(
            trial_df,
            window_sec=window_sec,
            step_sec=step_sec,
        )

        for item in windows:

            row = summarize_window(
                item["window_df"]
            )

            row["window_start_sec"] = item[
                "window_start_sec"
            ]

            row["window_end_sec"] = item[
                "window_end_sec"
            ]

            results.append(row)

    result = pd.DataFrame(results)

    print(
        f"[create_windows] "
        f"{total_trials} trials -> "
        f"{len(result)} windows"
    )

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    input_file = Path(
        "data/final/train_combined.csv"
    )

    output_file = Path(
        "data/final/train_windows.csv"
    )

    print(f"Reading: {input_file}")

    df = pd.read_csv(input_file)

    print(
        f"Loaded {len(df):,} rows "
        f"and {len(df.columns)} columns"
    )

    windows = create_windows(df)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows.to_csv(
        output_file,
        index=False,
    )

    print(
        f"[OK] Wrote {len(windows):,} windows "
        f"to {output_file}"
    )