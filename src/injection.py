"""inject_tremor_into_trial(): apply the tremor model to one trial.

Tremor is added ONLY to rows where event == 'move'. press/release/
double_click rows keep observed == ground_truth exactly, so discrete
action locations are never perturbed.
"""

import numpy as np
import pandas as pd

from .tremor_model import generate_tremor, _unit_normals


def inject_tremor_into_trial(trial_df: pd.DataFrame, freq, amplitude_px, phase,
                              rng: np.random.Generator,
                              tremor_status: int) -> pd.DataFrame:
    """Add ground_truth_x/y and observed_x/y to one trial's rows.

    freq / amplitude_px / phase are None when tremor_status == 0.
    """
    trial_df = trial_df.sort_values("elapsed_sec").reset_index(drop=True)

    gt_x = trial_df["x"].to_numpy(dtype=float)
    gt_y = trial_df["y"].to_numpy(dtype=float)
    obs_x = gt_x.copy()
    obs_y = gt_y.copy()

    trial_df["ground_truth_x"] = gt_x
    trial_df["ground_truth_y"] = gt_y
    trial_df["tremor_status"] = tremor_status

    if tremor_status == 1:
        dt = trial_df["dt"].fillna(0.0).to_numpy(dtype=float)
        t_local = trial_df["elapsed_sec"].to_numpy(dtype=float)

        nx, ny = _unit_normals(gt_x, gt_y, dt)
        s = generate_tremor(dt, freq, amplitude_px, phase, t_local, rng)

        move_mask = (trial_df["event"] == "move").to_numpy()
        obs_x[move_mask] = gt_x[move_mask] + s[move_mask] * nx[move_mask]
        obs_y[move_mask] = gt_y[move_mask] + s[move_mask] * ny[move_mask]

        trial_df["tremor_frequency_hz"] = freq
        trial_df["tremor_amplitude_px"] = amplitude_px
        trial_df["tremor_phase_rad"] = phase
    else:
        trial_df["tremor_frequency_hz"] = np.nan
        trial_df["tremor_amplitude_px"] = np.nan
        trial_df["tremor_phase_rad"] = np.nan

    trial_df["observed_x"] = obs_x
    trial_df["observed_y"] = obs_y
    return trial_df
