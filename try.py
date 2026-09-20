import pandas as pd

df = pd.read_csv("data/final/validation/trial_level_validation.csv")

bad = df[
    df["detected_frequency_hz"].notna() &
    (df["frequency_error_hz"].abs() > 1.0)
]

print(bad[
    [
        "participant_id",
        "trial_id",
        "task",
        "injected_frequency_hz",
        "configured_amplitude_px",
        "detected_frequency_hz",
        "frequency_error_hz",
        "frequency_error_percent",
        "native_fs_hz",
        "rms_residual_px",
        "peak_residual_px"
    ]
].to_string(index=False))