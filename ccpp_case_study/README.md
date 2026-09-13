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

## Official manuscript and supplementary figures

After the two full experiments have produced their CSV outputs, run the
standalone presentation script from this directory:

~~~powershell
python plot_manuscript_figures.py `
  --srcv-run-dir outputs/srcv_full `
  --aacv-run-dir outputs/aacv_full `
  --output-dir outputs/manuscript_figures
~~~

**These are the figures formally displayed in the manuscript and its
supplement.** The script generates the final layouts directly: no manual
cropping, panel assembly, model refitting, rescoring, or re-aggregation is
required. The original full-diagnostic plotting commands remain available
and intentionally retain their original four-panel layouts.

There are two workflows:

- **Existing CSVs:** run the command above, substituting the locations of your
  completed SRCV and AACV results. No training is run.
- **From scratch:** complete the full SRCV and full AACV commands documented
  below, then run the same command above. The full configurations, frozen
  reference, seeds, and primary AACV variant are unchanged.

The three directory options can be relative to the current working directory
or absolute. Their defaults are the paths shown above, resolved relative to
this script. No machine-specific server path or automatic "latest run"
selection is used. Keep the output directory separate from the input CSV
directories.

### Inputs and official panel mapping

The SRCV input directory must contain:

~~~text
candidate_test_risk_by_radius.csv
summary_by_radius.csv
radius_mismatch_posthoc/radius_mismatch_summary.csv
radius_mismatch_posthoc/radius_mismatch_selection_frequencies.csv
~~~

The AACV input directory must contain:

~~~text
aacv_candidate_test_scores.csv
aacv_summary_by_radius.csv
aacv_radius_mismatch/radius_mismatch_summary.csv
aacv_radius_mismatch/radius_mismatch_selection_frequencies.csv
~~~

The script checks the full configured grids (20 SRCV radii with 25 outer
replications; 37 AACV radii with 30 outer replications) and uses only the
predeclared AACV variant `primary_histgb_oof_j2_b6`, never a sensitivity
variant selected by held-out performance. Missing cells, duplicate cells,
nonfinite plotted values, incompatible replication counts, or invalid
selection frequencies cause an error.

Every output below is written as both vector PDF and 400-dpi PNG:

| Output stem | Paper location and layout (original panel letters) |
| --- | --- |
| `fig_ccpp_manuscript_heldout` | Main text, Power-plant held-out analyses: 3 rows by 2 columns; left SRCV, right AACV; rows A (full range), B (transitions), C (held-out-best match). AACV B retains both Early and Late windows. |
| `fig_ccpp_manuscript_mismatch` | Main text, Power-plant radius-mismatch diagnostics: 2 rows by 2 columns; top SRCV, bottom AACV; columns B (match heatmap), D (selected-model frequencies). |
| `fig_ccpp_supp_heldout_excess` | Supplement, Detailed results and radius guidance: 1 row by 2 columns; SRCV D (regret), AACV D (score excess). |
| `fig_ccpp_supp_mismatch_excess` | Same supplementary subsection: 2 rows by 2 columns; top SRCV, bottom AACV; columns A (regret/excess heatmap), C (representative slices). |

The six removed main-text panels are also exported individually, using the
same drawing functions as the composite supplementary figures:

~~~text
fig_ccpp_srcv_heldout_regret
fig_ccpp_aacv_heldout_excess
fig_ccpp_srcv_mismatch_regret
fig_ccpp_srcv_mismatch_slices
fig_ccpp_aacv_mismatch_excess
fig_ccpp_aacv_mismatch_slices
~~~

Numerical conventions match the original plotting routines: candidate and
matched-radius performance bands are one standard error; SRCV mismatch
slices retain their 1.96-MCSE bands and positive clean-CV reference lines;
AACV mismatch slices retain clean-CV reference lines without adding bands.
Heatmaps preserve every original cell and radius-grid ordering. All model
colors, line styles, and markers are retained. Only layout, labels, and the
portable Matplotlib-bundled DejaVu Sans font are standardized.

`figure_manifest.json` records input/configuration hashes, plotting-code
hashes, package versions, panel mappings, and output hashes. The script also
checks that no input changed during rendering. For identical rendering,
use the same CSVs, code version, and package versions; retraining or changing
the numerical environment need not produce byte-identical files.

### Figure regression tests

~~~powershell
python -m pytest tests/test_manuscript_figures.py -q
~~~

The default tests use deterministic synthetic CSVs and compare every new
panel with the original plotting routine, including line coordinates,
uncertainty-band vertices, heatmap cells, and both AACV inset windows.
To perform those comparisons on completed experimental outputs instead:

~~~powershell
$env:CCPP_MANUSCRIPT_SRCV_RUN_DIR = "outputs/srcv_full"
$env:CCPP_MANUSCRIPT_AACV_RUN_DIR = "outputs/aacv_full"
python -m pytest tests/test_manuscript_figures.py -q
~~~

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

### Finite radius-specific attack sets

AACV no longer optimizes over the standardized continuous box. It evaluates a
deterministic finite set. At radius zero,

$$
A(0)=\{0\}.
$$

For every configured $r>0$,

$$
A(r)=\{0\}\cup\{r s:s\in\{-1,+1\}^4\}.
$$

The first point is the unperturbed observation. The remaining 16 points are
the lexicographically ordered corners of the current radius-$r$ cube. All six
candidate models use this same definition. Validation and evaluation use
identical attack sets and exact finite enumeration; there are no Sobol starts,
coordinate-refinement paths, model-specific budgets, or stronger-search
audits.

The sets are radius-specific rather than cumulative. Adding an intermediate
radius does not alter extrema or AACV scores at any existing radius. The result
should therefore be described as exact robustness over the declared 17-point
set, not robustness over the continuous cube $[-r,r]^4$.

## AACV configurations

`configs/ccpp_aacv_smoke.json` uses radii 0, 0.05, 0.15, 0.5, and 1.0,
one outer repeat, one inner repeat, and reduced candidate fitting budgets.
The working-reference RF and HistGB protocols remain fixed at 300
trees/iterations.

`configs/ccpp_aacv_medium.json` uses one outer and one inner split, full
candidate fitting settings, and 54 radii:

~~~text
0, 0.0025, 0.005, 0.01,
0.02, 0.04, ..., 0.98, 1.00
~~~

It is a dense exploratory scan for model-selection tradeoff and switching
radii. It does not provide stable Monte Carlo standard errors.

`configs/ccpp_aacv_full.json` uses a 37-point grid that keeps the three
observed switching regions dense, adds moderate coverage between them, and
ends at 0.90:

~~~text
0, 0.0025, 0.005, 0.0075, 0.0085, 0.01, 0.02,
0.04, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.10,
0.15, 0.22, 0.30, 0.40, 0.50, 0.60, 0.68,
0.70, 0.705, 0.71, 0.715, 0.72, 0.725, 0.73, 0.74,
0.75, 0.76, 0.78, 0.82, 0.86, 0.90
~~~

Full uses the strengthened protocol of 30 outer by 3 inner repeats. The 30
outer replications give selected-model frequencies about a 3.33-percentage-
point resolution and a worst-case binomial Monte Carlo standard error of about
0.091. Scaling the measured 30-radius local run to 36 radii and 30 outer
replications gives an estimated runtime of roughly 8-10 hours on the same
workstation.

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
schema. Schema-v2 continuous-optimizer shards cannot be resumed under schema v3.

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
attack_set_diagnostics.csv
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
fingerprint, data/source hashes, split seeds, the exact finite attack-set
definition and fingerprint, attack/scientific schema versions, package
versions, available Git commit/status, and scientific-status qualifications.

## Git policy

Everything under `ccpp_case_study/outputs/` is generated and ignored by Git.
Commit source code, configurations, tests, documentation, the frozen
lightweight reference specification, and required input data. Do not commit
rerun CSVs, figures, shards, caches, or virtual environments.
