# CCPP Case-Study Reproducibility Project

This directory is a self-contained reproduction package for the two CCPP
main-text figures used by `../../main.tex`:

- `fig_ccpp_case_study_main.pdf/png`
- `fig_ccpp_radius_mismatch.pdf/png`

It covers only the CCPP case study. The nested-linear and teacher--student
simulation figures are intentionally out of scope.

## What Is Archived

The reference run is stored under:

```text
reference/run_20260522_004859/
```

It contains the CSV outputs from the full CCPP run with:

- CCPP data file: `data/Folds5x2_pp.xlsx`
- radius grid:
  `0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.17, 0.18, 0.19, 0.195, 0.2, 0.205, 0.21, 0.22, 0.24, 0.27, 0.32, 0.4, 0.5`
- outer replications: `25`
- inner replications: `15`
- base seed: `20260521`
- validation perturbation draws: `1`
- test perturbation draws: `1`
- candidate order:
  `OLS, RidgeCV, LassoCV, KNN, RandomForest, HistGB`

The archived manuscript figures are stored under:

```text
reference/figures/
```

## Dependencies

Install the Python dependencies from this directory:

```bash
pip install -r requirements.txt
```

No system-level dependency is required beyond a working Python scientific stack
and an Excel reader supported by `openpyxl`.

## Quick Plot-Only Reproduction

This is the fastest path for a reproducibility reviewer. It starts from the
archived CSV outputs and regenerates the two manuscript figures:

```bash
python -m ccpp_repro plot-only ^
  --config configs/ccpp_full.json ^
  --run-dir reference/run_20260522_004859 ^
  --figure-dir outputs/plot_only_figures
```

Verify the archived CSV signatures and regenerated figures:

```bash
python -m ccpp_repro verify ^
  --config configs/ccpp_full.json ^
  --run-dir reference/run_20260522_004859 ^
  --figure-dir outputs/plot_only_figures ^
  --reference-figure-dir reference/figures
```

The verification checks CSV schemas through row counts, selected numerical
signatures, figure existence, and PNG-level comparison when reference PNG files
are available.

## Smoke Test From Raw Data

The smoke test uses the same code path as the full experiment but a tiny grid:

```bash
python -m ccpp_repro run ^
  --config configs/ccpp_smoke.json ^
  --output-dir outputs/smoke_run ^
  --figure-dir outputs/smoke_figures ^
  --no-progress
```

Verify the smoke outputs:

```bash
python -m ccpp_repro verify ^
  --config configs/ccpp_smoke.json ^
  --run-dir outputs/smoke_run ^
  --figure-dir outputs/smoke_figures
```

This confirms that data loading, candidate fitting, robust scoring, aggregation,
radius-mismatch post-processing, and figure generation all run end to end.

## Full Reproduction From Raw Data

The full run refits all candidates and regenerates the CSV outputs and figures:

```bash
python -m ccpp_repro run ^
  --config configs/ccpp_full.json ^
  --output-dir outputs/full_run ^
  --figure-dir outputs/full_figures
```

Then verify:

```bash
python -m ccpp_repro verify ^
  --config configs/ccpp_full.json ^
  --run-dir outputs/full_run ^
  --figure-dir outputs/full_figures ^
  --reference-figure-dir reference/figures
```

Expected runtime is several hours on a typical workstation because the full run
fits `20 * 25 * 15 * 6 = 45,000` candidate models. If the full rerun differs
from the archived signatures under the same dependency versions, report the
difference rather than tuning parameters to force agreement.

## Configuration

The full configuration is `configs/ccpp_full.json`. It records:

- data path and SHA256 checksum;
- radius grid and repeat counts;
- random seed and perturbation draw counts;
- candidate order and tree-model sizes;
- mismatch evaluation radii;
- manuscript figure stems and formats;
- reference row counts and numerical signatures.

All relative paths are resolved against this reproducibility directory. There
are no hard-coded external drive paths.

## Output Files

A successful full run writes:

```text
inner_validation_scores.csv
inner_test_scores.csv
inner_selections.csv
outer_summary.csv
summary_by_radius.csv
selection_frequencies.csv
candidate_test_risk_by_radius.csv
radius_mismatch_posthoc/
```

The two manuscript figures are written to the requested figure directory:

```text
fig_ccpp_case_study_main.pdf
fig_ccpp_case_study_main.png
fig_ccpp_radius_mismatch.pdf
fig_ccpp_radius_mismatch.png
```

## Notes for Reviewers

The radius-mismatch figure is a post-processing analysis of the same full-run
CSV outputs. It does not refit models. It recombines the selected model at
`r_val` with outer-test robust scores at `r_eval`, then reports regret,
Monte Carlo standard errors, oracle-match rates, selection frequencies, and
stability summaries.
