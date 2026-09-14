
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def rename_participants(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Replace real user_id values with anonymous IDs U1..U66.

    Assignment order is shuffled (not alphabetical) so the anonymous ID
    reveals nothing about the original name ordering. The mapping is kept
    ONLY in memory / returned to the caller — it is intentionally never
    written to any output file, to avoid re-identifying participants from
    the delivered dataset.
    """
    rng = np.random.default_rng(seed)
    original_ids = sorted(df["user_id"].unique())
    shuffled = rng.permutation(original_ids)
    mapping = {orig: f"U{i+1}" for i, orig in enumerate(shuffled)}

    df = df.copy()
    df["user_id"] = df["user_id"].map(mapping)

    n = len(mapping)
    print(f"[rename_participants] Anonymized {n} participants -> U1..U{n} "
          f"(seed={seed}); mapping kept in memory only, not written to disk.")
    return df, mapping
