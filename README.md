# TREMORSHIELD — Synthetic Tremor Injection + Validation Pipeline

Two stages:
1. **Injection** — participant-level train/test split + trial-level
   synthetic tremor injection, producing paired ground-truth/observed
   trajectories.
2. **Validation** — proves the injected tremor actually has the
   frequency/amplitude/spatial properties it was designed to have,
   *before* that data is used to train a Random Forest classifier or
   evaluate an Adaptive Kalman Filter.

## Folder structure

```
tremorshield/
├── README.md
├── requirements.txt
├── run_pipeline.py             # STAGE 1 entry point: injection
├── run_validation.py           # STAGE 2 entry point: validation
├── src/
│   ├── __init__.py
│   │
│   │  ── Stage 1: injection ──
│   ├── config.py                # constants, TremorConfig
│   ├── data_loader.py           # load_data() — schema validation
│   ├── anonymize.py             # rename_participants()
│   ├── splitting.py             # split_participants(), split_training_trials(), assign_test_trials()
│   ├── tremor_model.py          # generate_tremor(), _unit_normals() — the Kulkarni et al. (2024) model
│   ├── injection.py             # inject_tremor_into_trial()
│   ├── features.py              # recalculate_features()
│   ├── pipeline.py              # build_condition() — orchestrates one split
│   ├── validation.py            # generation-stage sanity check: leakage asserts + PSD figure
│   ├── io_writer.py             # save_outputs()
│   │
│   │  ── Stage 2: validation ──
│   ├── validation_common.py     # shared: unit normals, uniform resampling, Welch PSD, Nyquist check
│   ├── validation_signal.py     # Part 1: frequency / amplitude / envelope / clean & event preservation
│   ├── validation_spatial.py    # Part 2a: paired clean-vs-tremor spatial deviation, path length, direction change
│   ├── validation_robustpdx.py  # Part 2b: optional external spatial reference (configurable column mapping)
│   ├── validation_statistics.py # distribution comparison: KS, Wasserstein, Cohen's d
│   └── validation_report.py     # validation_report.md generator with PASS/FAIL criteria
└── data/
    ├── raw/                     # put all_cleaned_data.csv here (read-only, never modified)
    └── final/                   # both stages write here (git-ignored; regenerate via the scripts)
        └── validation/          # run_validation.py output (CSVs, validation_plots/, validation_report.md)
```

## Usage

**Stage 1 — injection:**
```bash
pip install -r requirements.txt
cp /path/to/all_cleaned_data.csv data/raw/
python run_pipeline.py --input data/raw/all_cleaned_data.csv --outdir data/final --seed 42
```
Outputs: `train_clean.csv`, `train_synthetic_tremor.csv`, `train_combined.csv`,
`test_clean.csv`, `test_synthetic_tremor.csv`, `test_combined.csv`,
`participant_split.csv`, `tremor_metadata.csv`, `run_config.json`,
`plots/tremor_frequency_validation.png`.

**Stage 2 — validation** (run after Stage 1):
```bash
python run_validation.py \
    --train data/final/train_combined.csv \
    --test data/final/test_combined.csv \
    --metadata data/final/tremor_metadata.csv \
    --outdir data/final/validation \
    --robustpdx path/to/robustpdx.csv    # optional — omit to skip that comparison
```
Outputs (in `data/final/validation/`):
`trial_level_validation.csv`, `frequency_validation.csv`,
`amplitude_validation.csv`, `signal_validation_summary.csv`,
`spatial_validation.csv`, `robustpdx_feature_summary.csv`,
`robustpdx_comparison.csv`, `validation_report.md`, and
`validation_plots/` (PSD-by-frequency, detected-vs-injected frequency,
amplitude-vs-RMS, spatial boxplots, RobustPDX comparison ECDFs/boxplots).

If `--robustpdx` isn't supplied, or the file/columns don't match
`RobustPDXColumnMap` in `src/validation_robustpdx.py`, that comparison is
skipped and clearly logged — never fabricated.

## Design rules enforced by the code

**Injection (Stage 1):**
1. **Split by participant, never by row** (`splitting.split_participants`).
2. **tremor_status is metadata, not the RF target** — the RF target stays
   task/context (`normal, fast, slow, precision, click, double_click, drag,
   target_selection, idle`).
3. **Tremor perturbs `move` events only** — `press`/`release`/`double_click`
   rows keep their exact original coordinates.
4. **~70/30 tremor/clean is a trial-level, task-stratified split within
   train** — never row-level, never applied to test-for-training.
5. **Test participants are never used to build training data.** Each test
   trial is rendered twice (TEST-A clean, TEST-B tremor) for paired eval.

**Validation (Stage 2):**
6. **Frequency-domain checks use the SIGNED transverse residual**, never
   `sqrt(res_x²+res_y²)` — the magnitude is a rectified sinusoid and
   rectification doubles the apparent frequency.
7. **A trial's frequency is only validated if its native (raw) mouse-event
   rate clears a 3× Nyquist margin over the injected frequency** —
   verified against the actual pipeline output: linear interpolation
   cannot manufacture frequency content that sparse idle/slow segments
   never sampled in the first place. Trials that fail this are skipped
   and logged, not silently misreported.
8. **RobustPDX is a spatial reference only**, never used for 4/6/8/10 Hz
   temporal validation (its point sequences lack reliable per-point
   timestamps), and every comparison reports a distance/significance
   measure (KS, Wasserstein) *together with* an effect size (Cohen's d) —
   never significance alone.
9. **PASS/FAIL thresholds are stated up front** in `validation_report.py`
   (not chosen after seeing results), and the report distinguishes
   mathematical validation, spatial similarity to RobustPDX, and clinical
   validity as three separate, non-equivalent claims.

See `src/tremor_model.py` for the tremor model derivation and full
citations (Kulkarni et al., 2024; Rocon et al., 2004; Randall, 1973;
Gantert et al., 1992).
