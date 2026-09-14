"""
PART 2 — SPATIAL VALIDATION (RobustPDX half): external reference data.

RobustPDX is used ONLY as a spatial/distributional reference — it is NOT
used for 4/6/8/10 Hz frequency validation, because its point sequences do
not provide reliable per-point timestamps.

We don't have Anthropic-side access to your actual RobustPDX file, and its
exact column names/layout can vary by release, so this module never
guesses at column names. Instead you supply a RobustPDXColumnMap telling
it what your file actually calls things. If no path is given (or the file
isn't found), this module skips cleanly and says so — it never fabricates
data or column structure.

Required columns (must exist under whatever names you map them to):
    x_col, y_col            - point coordinates
    trace_id_col            - identifies which points belong to one trace/trial
    group_col               - the PD / non-PD (/ suspected) label

Optional:
    reference_x_col, reference_y_col
        - if your RobustPDX release includes the INTENDED path (e.g. the
          straight-line or spiral centerline), supply these to unlock
          "deviation from reference geometry" features. Without them, that
          feature is skipped and logged as skipped (not invented).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RobustPDXColumnMap:
    path: Optional[str] = None
    x_col: str = "x"
    y_col: str = "y"
    trace_id_col: str = "trace_id"
    group_col: str = "group"          # expected values like 'PD', 'non-PD' (case-insensitive)
    reference_x_col: Optional[str] = None
    reference_y_col: Optional[str] = None


def load_robustpdx(colmap: RobustPDXColumnMap) -> Optional[pd.DataFrame]:
    """Load the RobustPDX CSV if a path is configured and the file exists
    and has the required columns. Returns None (and prints why) otherwise
    — callers must handle None gracefully rather than assuming data.
    """
    if not colmap.path:
        print("[validation_robustpdx] No RobustPDX path configured — "
              "skipping RobustPDX comparison. Pass --robustpdx PATH to "
              "run_validation.py to enable it.")
        return None
    if not Path(colmap.path).exists():
        print(f"[validation_robustpdx] RobustPDX file not found at "
              f"{colmap.path} — skipping RobustPDX comparison.")
        return None

    df = pd.read_csv(colmap.path)
    required = [colmap.x_col, colmap.y_col, colmap.trace_id_col, colmap.group_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[validation_robustpdx] RobustPDX file is missing required "
              f"column(s) {missing} under the configured column mapping "
              f"{colmap}. Update RobustPDXColumnMap to match your file's "
              f"actual column names. Skipping RobustPDX comparison.")
        return None

    print(f"[validation_robustpdx] Loaded {len(df):,} points, "
          f"{df[colmap.trace_id_col].nunique()} traces, "
          f"groups={sorted(df[colmap.group_col].astype(str).unique())}")
    return df


def _path_length(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)))


def compute_robustpdx_features(df: pd.DataFrame, colmap: RobustPDXColumnMap) -> pd.DataFrame:
    """Compute per-trace spatial features that are always safely
    computable from raw (x, y) points: path length, bounding-box area,
    x/y standard deviation. Deviation-from-reference-geometry is only
    computed if reference_x_col/reference_y_col are supplied; otherwise
    it's skipped (never invented) and that is logged once.
    """
    have_reference = colmap.reference_x_col is not None and colmap.reference_y_col is not None
    if not have_reference:
        print("[validation_robustpdx] No reference_x_col/reference_y_col configured "
              "— skipping 'deviation from intended path' feature for RobustPDX "
              "(only generically computable features will be reported).")

    rows = []
    for trace_id, g in df.groupby(colmap.trace_id_col):
        x = g[colmap.x_col].to_numpy(dtype=float)
        y = g[colmap.y_col].to_numpy(dtype=float)
        if len(x) < 3:
            continue
        group_label = str(g[colmap.group_col].iloc[0])

        feats = {
            "trace_id": trace_id,
            "group": group_label,
            "n_points": len(x),
            "path_length_px": _path_length(x, y),
            "x_std_px": float(x.std()),
            "y_std_px": float(y.std()),
            "bbox_area_px2": float((x.max() - x.min()) * (y.max() - y.min())),
        }
        if have_reference:
            rx = g[colmap.reference_x_col].to_numpy(dtype=float)
            ry = g[colmap.reference_y_col].to_numpy(dtype=float)
            dev = np.sqrt((x - rx) ** 2 + (y - ry) ** 2)
            feats.update({
                "deviation_mean_px": float(dev.mean()),
                "deviation_rms_px": float(np.sqrt(np.mean(dev ** 2))),
                "deviation_p95_px": float(np.percentile(dev, 95)),
                "deviation_max_px": float(dev.max()),
            })
        rows.append(feats)

    return pd.DataFrame(rows)


def run_robustpdx_validation(colmap: RobustPDXColumnMap, outdir: str) -> Optional[pd.DataFrame]:
    """Load RobustPDX (if configured), compute per-trace features, and
    write robustpdx_feature_summary.csv. Returns the per-trace feature
    DataFrame, or None if RobustPDX wasn't available (callers/statistics
    module must handle this by skipping the comparison, not by failing).
    """
    from pathlib import Path as _Path
    outdir = _Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_robustpdx(colmap)
    if df is None:
        pd.DataFrame(columns=["note"]).assign(
            note=["RobustPDX not available for this run — see log for reason."]
        ).to_csv(outdir / "robustpdx_feature_summary.csv", index=False)
        return None

    features = compute_robustpdx_features(df, colmap)
    features.to_csv(outdir / "robustpdx_feature_summary.csv", index=False)
    print(f"[validation_robustpdx] wrote {outdir/'robustpdx_feature_summary.csv'} "
          f"({len(features)} traces)")
    return features
