# CCPP Case-Study Reproducibility

This directory contains two independent model-selection experiments on the
UCI Combined Cycle Power Plant (CCPP) data:

- stochastic robust cross-validation (SRCV), the original case-study pipeline
  used by the current manuscript;
- aligned adversarial cross-validation (AACV), a separate real-data plug-in
  experiment based on the Gaussian-Hermite correction in Section 4.4.1 of
  `../main.tex`.

The two pipelines have separate commands, configurations, output directories,
figures, and verification rules. Running AACV does not alter SRCV results.

## Data and candidate models

The checked input is `data/Folds5x2_pp.xlsx`. It contains 9568 observations,
with `AT`, `V`, `AP`, and `RH` as covariates and `PE` as the response. The
configuration records the required columns, row count, source URL, and SHA256.

Both pipelines use this fixed candidate order:

```text
OLS, RidgeCV, LassoCV, KNN, RandomForest, HistGB
```

Exact ties are resolved by this order. Preprocessing is fitted only on the
current training observations.

## Installation and tests

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

The direct dependencies are NumPy, pandas, scikit-learn, SciPy, Matplotlib,
openpyxl, tqdm, and pytest.

## SRCV pipeline

The existing SRCV entry points and configurations are unchanged.

```powershell
python -m ccpp_repro run `
  --config configs/ccpp_smoke.json `
  --output-dir outputs/srcv_smoke `
  --figure-dir outputs/srcv_smoke_figures `
  --no-progress

python -m ccpp_repro verify `
  --config configs/ccpp_smoke.json `
  --run-dir outputs/srcv_smoke `
  --figure-dir outputs/srcv_smoke_figures
```

The SRCV full configuration uses 20 radii, 25 outer replications, 15 inner
replications, and six candidates.

```powershell
python -m ccpp_repro run `
  --config configs/ccpp_full.json `
  --output-dir outputs/srcv_full `
  --figure-dir outputs/srcv_full_figures

python -m ccpp_repro verify `
  --config configs/ccpp_full.json `
  --run-dir outputs/srcv_full `
  --figure-dir outputs/srcv_full_figures

python -m ccpp_repro plot-only `
  --config configs/ccpp_full.json `
  --run-dir outputs/srcv_full `
  --figure-dir outputs/srcv_plot_only_figures

python -m ccpp_repro verify `
  --config configs/ccpp_full.json `
  --run-dir outputs/srcv_full `
  --figure-dir outputs/srcv_plot_only_figures
```

A full SRCV rerun fits 45,000 candidate models and can take several hours on a
typical workstation. Existing CSV outputs can be redrawn with `plot-only`

## AACV working-error design

The complete specification is in `AACV_WORKING_ERROR_GUIDELINE.md`. This is a
working-model analysis: it does not identify an unavailable physical
regression function or physical random-error law.

### Frozen global reference selection

AACV fixes one primary working reference algorithm before inspecting AACV
held-out results. The frozen selection file is
`configs/ccpp_aacv_reference.json`. It is generated from the radius-zero clean
validation rows in the canonical SRCV full run:

```text
25 outer repeats x 15 inner repeats = 375 clean-risk splits
criterion                          = mean validation squared error
aggregation                        = mean over the 375 splits
tie-breaking                       = fixed candidate order
selected working reference         = HistGB
runner-up                           = RandomForest
```

The selection file records every candidate's mean clean MSE, Monte Carlo
standard error, paired difference from the winner, and split-win frequency.
It also fingerprints the source configuration and normalized source table.

To regenerate it from an available canonical SRCV run:

```powershell
python -m ccpp_repro freeze-aacv-reference `
  --srcv-config configs/ccpp_full.json `
  --srcv-run-dir outputs/full_run `
  --output configs/ccpp_aacv_reference.json
```

The canonical SRCV analysis and AACV reuse the same CCPP dataset. The frozen
selection is therefore a case-study design decision, not independent external
validation and not evidence that HistGB equals the unknown conditional mean.

### Honest cross-fitted residuals

Within every AACV inner training split, the code performs five-fold
cross-fitting. Each fold fits its own scaler and working reference model using
the other four folds, then predicts only the held-out fold. Inner validation
and outer-test responses never enter this construction.

The primary plug-in scale is

```text
sigma2_hat = sample_variance(Y - HistGB_OOF_prediction, ddof=1)
sigma_hat  = sqrt(sigma2_hat)
```

The primary working law is homoskedastic Gaussian. The numerical variance
floor is fixed at `1e-8 * sample_variance(Y_train)` and its use is recorded.
Residual means, MSEs, skewness, excess kurtosis, Jarque-Bera statistics, and
residual/covariate relationships are diagnostics only; they do not change the
predeclared calculation.

### Gaussian-Hermite correction

For adversarial center `c`, the correction is

```text
psi_hat = sum from m=0 to J of
          g_(2m) B^(1-2m) sigma_hat^(2m)
          H_(2m)((Y-c)/sigma_hat)
```

The primary setting is `J=2`, `B=6*sigma_hat`, and no clipping. For prediction
extrema `M` and `m`, AACV uses

```text
c     = (M + m) / 2
rho   = (M - m) / 2
score = (Y-c)^2 + rho^2 + 2*rho*psi_hat
```

Every candidate within a split uses the same scale for a given analysis
variant. Candidate-specific scale estimates are not used.

### Predeclared sensitivities

There are 12 normalized `analysis_variant` values. Each sensitivity changes
one nuisance component relative to the primary analysis:

```text
primary_histgb_oof_j2_b6

reference_ols_oof_j2_b6
reference_ridgecv_oof_j2_b6
reference_lassocv_oof_j2_b6
reference_knn_oof_j2_b6
reference_randomforest_oof_j2_b6

scale_tong_wang_j2_b6
scale_fan_yao_j2_b6

hermite_histgb_oof_j1_b6
hermite_histgb_oof_j3_b6
hermite_histgb_oof_j2_b4
hermite_histgb_oof_j2_b8
```

All six reference algorithms use honest five-fold OOF residuals and the same
fold partition. The Tong-Wang-style multivariate difference estimator and the
Fan-Yao-style multivariate local residual-variance estimator are structural
sensitivity analyses; neither is labeled as a ground-truth variance estimate.
Hermite sensitivities reuse the same fitted candidates and adversarial
extrema.

### Continuous adversarial optimization

The perturbation set is the standardized continuous box
`delta in [-r,r]^4`. OLS, RidgeCV, and LassoCV extrema are solved exactly.
KNN, RandomForest, and HistGB use deterministic batched multi-start coordinate
pattern search. Starts include the center, all 16 corners, eight signed axis
endpoints, and nested scrambled Sobol points.

For each observation and direction, the primary search retains `K` independent
paths, refines every path separately, and takes their strict envelope. Every
path keeps its historical best value, so later step sizes cannot degrade it.
The stronger audit inherits all primary paths and extrema, adds the next Sobol
points from the same scrambled sequence, continues to step `1/128`, and takes
an explicit envelope with the primary result. Therefore a stronger maximum
cannot decrease and a stronger minimum cannot increase.

The full primary budgets are:

```text
validation: 8 Sobol starts; steps 1/2, 1/4, 1/8, 1/16
evaluation: 32 Sobol starts; steps 1/2, 1/4, 1/8, 1/16, 1/32, 1/64
```

The Smoke run derives nested `K=1,2,3,4` envelopes from one four-path search.
It retains `K=3` only when all predeclared selection, score, and extrema gates
pass; otherwise the checked-in Medium and Full protocols must use `K=4`. The recorded Smoke gate failed the extrema thresholds, so all three checked-in configurations are frozen at `K=4`.

## AACV configurations

`configs/ccpp_aacv_smoke.json` uses radii 0, 0.05, 0.15, 0.5, and 1.0,
one outer repeat, one inner repeat, and reduced candidate/optimizer budgets.
It writes the path-calibration detail and decision tables. The working-reference
RF and HistGB protocols remain fixed at 300 trees/iterations.

Medium and Full share this 15-point radius grid. It is concentrated around the
pilot switching regions near 0, 0.1, and 0.5, with sparse anchors elsewhere:

```text
0, 0.0025, 0.005, 0.01, 0.05,
0.08, 0.10, 0.125, 0.15,
0.30,
0.45, 0.475, 0.50, 0.525, 0.60
```

`configs/ccpp_aacv_medium.json` uses one outer and one inner split while
retaining the Full candidate models and primary attack budgets. It is intended
for a lower-cost exploratory run and does not provide stable Monte Carlo
standard errors. `configs/ccpp_aacv_full.json` uses the robust design of 20
outer by 3 inner repeats for the final repeated experiment. The 20 outer
replications give selected-model frequencies a 5-percentage-point resolution
and a worst-case binomial Monte Carlo standard error of about 0.112.

## AACV commands

Smoke run and verification:

```powershell
python -m ccpp_repro run-aacv `
  --config configs/ccpp_aacv_smoke.json `
  --output-dir outputs/aacv_smoke `
  --figure-dir outputs/aacv_smoke_figures `
  --no-progress

python -m ccpp_repro verify-aacv `
  --config configs/ccpp_aacv_smoke.json `
  --run-dir outputs/aacv_smoke `
  --figure-dir outputs/aacv_smoke_figures
```

Medium run:

```powershell
python -m ccpp_repro run-aacv `
  --config configs/ccpp_aacv_medium.json `
  --output-dir outputs/aacv_medium `
  --figure-dir outputs/aacv_medium_figures

python -m ccpp_repro verify-aacv `
  --config configs/ccpp_aacv_medium.json `
  --run-dir outputs/aacv_medium `
  --figure-dir outputs/aacv_medium_figures
```

Full run:

```powershell
python -m ccpp_repro run-aacv `
  --config configs/ccpp_aacv_full.json `
  --output-dir outputs/aacv_full `
  --figure-dir outputs/aacv_full_figures
```

Each completed outer replication is written atomically under `partials/`.
Resume an interrupted run with the same command plus `--resume`. A nonempty
output directory fails without `--resume`; resume also validates the reference,
configuration, scientific implementation version, attack algorithm, and shard
schema. Shards produced by the former single-path optimizer cannot be resumed.

Plot and verify existing outputs:

```powershell
python -m ccpp_repro plot-aacv `
  --config configs/ccpp_aacv_full.json `
  --run-dir outputs/aacv_full `
  --figure-dir outputs/aacv_full_figures

python -m ccpp_repro verify-aacv `
  --config configs/ccpp_aacv_full.json `
  --run-dir outputs/aacv_full `
  --figure-dir outputs/aacv_full_figures
```

`run-aacv` also supports `--skip-figures` and `--no-progress`.

## AACV outputs

The principal generated tables are:

```text
reference_selection.csv
reference_selection.json
working_reference_diagnostics.csv
scale_estimator_diagnostics.csv
scale_estimator_summary.csv
residual_covariate_diagnostics.csv
residual_covariate_bins.csv
inner_aacv_scores.csv
inner_test_aacv_scores.csv
inner_aacv_selections.csv
outer_aacv_summary.csv
outer_candidate_test_scores.csv
aacv_summary_by_radius.csv
aacv_selection_frequencies.csv
aacv_candidate_test_scores.csv
working_error_selection_switches.csv
working_error_sensitivity_summary.csv
attack_diagnostics.csv
attack_path_calibration.csv
attack_path_calibration_summary.csv
aacv_radius_mismatch/
partials/
run_manifest.json
```

The selector is either `clean` or `aacv`. Primary matched and mismatched radii
use qualified held-out benchmark terminology. Sensitivity tables report model
switching and agreement without using a held-out result to redefine the
primary variant.

The four figure families deliberately parallel the SRCV naming and
information hierarchy while retaining the AACV prefix:

```text
fig_ccpp_aacv_case_study_main.pdf/png
fig_ccpp_aacv_radius_mismatch.pdf/png
fig_ccpp_aacv_working_error_diagnostics.pdf/png
fig_ccpp_aacv_sensitivity.pdf/png
```

`run_manifest.json` records resolved settings, the reference specification and
fingerprint, data/source hashes, split and optimizer seeds, fixed path count,
calibration decision, attack/scientific schema versions, package versions,
available Git commit/status, and scientific-status qualifications.

## Git policy

Everything under `ccpp_case_study/outputs/` is generated and ignored by Git.
Commit source code, configurations, tests, documentation, the frozen
lightweight reference specification, and required input data. Do not commit
rerun CSVs, figures, shards, caches, or virtual environments.
