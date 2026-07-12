# Reproducibility Package

This repository is the top-level reproducibility package for three experiments.
The current snapshot contains both simulation experiments and the CCPP case
study.

## Experiment Status

| Order | Experiment | Status | Directory | Main artifacts |
| --- | --- | --- | --- | --- |
| 1 | Nested-linear simulation | Available | `nested_linear_example/` | raw CSVs, summary CSVs, threshold summaries, PNG plots |
| 2 | Neural-network teacher-student simulation | Available | `neaural_network_experiment/` | raw metrics, aggregated metrics, selection frequencies, tradeoff tables, plots |
| 3 | CCPP case study | Available | `ccpp_case_study/` | `fig_ccpp_case_study_main`, `fig_ccpp_radius_mismatch` |

## Repository Contents

This repository tracks source code, experiment configurations, tests,
documentation, and required input data. Generated outputs, local caches,
virtual environments, and model checkpoints are excluded through
`.gitignore`.

Contributors should review [`CONTRIBUTING.md`](CONTRIBUTING.md) before staging
changes or opening a pull request.

## 1. Nested-Linear Simulation

The first simulation lives in:

```text
nested_linear_example/
```

It studies a two-dimensional linear regression example:

```text
y = beta1 * x1 + beta2 * x2 + eps
```

The code compares two candidate models:

- `A1`: uses only `x1`;
- `A2`: uses both `x1` and `x2`.

For each Monte Carlo replication, the package fits both candidates, evaluates
cross-validation scores, computes closed-form robust risks, records the model
selected by CV, records the robust-optimal model, and reports whether the
selected model is optimal.

### Implemented Criteria

The package implements three validation criteria:

- `adversarial`: pointwise worst-case validation perturbation;
- `aligned_adversarial`: aligned adversarial score using a localized
  Gaussian-Hermite absolute-mean estimator;
- `stochastic`: validation-time uniform perturbation on `x2`.

### Internal Simulation Sweeps

The package has two internal sweeps, exposed as `experiment_1` and
`experiment_2` in the runner:

- `experiment_1`: fixed `beta1=1.0`, `beta2=1.0`, `sigma=0.5`; varies sample
  size, train/validation ratio, and robust radius near the adversarial and
  stochastic switching thresholds.
- `experiment_2`: fixed `beta1=1.0`, `beta2=1.0`, `sigma=0.5`; varies sample
  size from 100 to 5000 to study whether CV selection converges to the
  robust-optimal model.

### Setup

```bash
cd nested_linear_example
python -m pip install -e .
python -m pip install numpy pandas matplotlib seaborn rich pytest
```

### Tests

```bash
python -m pytest
```

### Smoke Run

```bash
python -m robust_cv_sim.runner --experiments experiment_2 --reps 10 --output-dir smoke_outputs --skip-plots
```

### Full Run

```bash
python -m robust_cv_sim.runner --experiments experiment_1 experiment_2 --reps 1000 --seed 20260319 --output-dir outputs_rerun
```

Useful runner options:

- `--skip-plots`: write raw and summary CSVs only;
- `--keep-chunks`: keep per-chunk CSV files after merging;
- `--no-resume`: ignore existing completed chunks and rerun everything.

### Outputs

Each internal sweep writes an experiment subdirectory with raw CSVs, summary
CSVs, threshold summaries, and plots.

## 2. Neural-Network Teacher-Student Simulation

The second simulation lives in:

```text
neaural_network_experiment/
```

It studies robust procedure selection in a synthetic teacher-student regression
problem. The feature vector is split into core, fragile, and noise blocks:

```text
X = (X_core, X_frag, X_noise)
Y = g_*(X_core) + lambda h_*(X_frag) + epsilon
```

The teacher functions `g_*` and `h_*` are two-layer ReLU networks. The
simulation compares four training procedures under attacked covariates:

- `erm`: standard MSE training on all features;
- `at`: adversarial training under the training threat model;
- `sr`: stochastic robust training with random perturbations;
- `core_only`: an MLP restricted to the core feature block.

The threat model perturbs fragile features only and supports restricted `L_inf`
and restricted `L2` attacks.

### Execution Order

The main command runs the experiment first and then makes plots.

Concretely, `scripts/runner.py main` calls
`src.experiments.pipeline.run_experiment_grid`. That function:

1. expands the training grid over `(n_total, lambda, r_train, threat_model)`;
2. runs all replications and SR-CV procedure selection;
3. writes `raw_metrics.csv`, `aggregated_metrics.csv`,
   `selection_frequencies.csv`, `tradeoff_raw.csv`, `tradeoff_summary.csv`,
   `winner_switch_points.csv`, and `manifests/run_config.yaml`;
4. calls `make_all_plots(...)` to write figures under `plots/`.

The separate `plots` command is a plot-only path for existing CSV outputs; it
reads `aggregated_metrics.csv`, `selection_frequencies.csv`, and
`raw_metrics.csv` from a completed run directory and redraws the figures.

### Setup

Python 3.11 is the target environment.

```bash
cd neaural_network_experiment
python -m pip install -r requirements.txt
```

The declared dependencies are `torch`, `numpy`, `pandas`, `matplotlib`,
`PyYAML`, `tqdm`, and `pytest`.

### Smoke Run

```bash
python scripts/runner.py smoke
```

The smoke configuration uses `n_total=240`, two `lambda` values, two evaluation
radii, one training radius, one threat model, and two replications.

### Main Runs

```bash
python scripts/runner.py main --scale lite --workers 2
python scripts/runner.py main --scale medium --workers 4
python scripts/runner.py main --scale full --workers 6
```

The default main scale is `medium`. Use `lite` for quick visual validation
before running `medium` or `full`.

### Plot-Only Reproduction

```bash
python scripts/runner.py plots --input-dir outputs/main_medium_YYYYMMDD_HHMMSS
```

This command does not train models. It only redraws plots from existing CSVs.

### Convenience Wrappers

```bash
python scripts/run_smoke_test.py
python scripts/run_main_experiment.py
python scripts/make_all_plots.py --input-dir outputs/main_medium_YYYYMMDD_HHMMSS
```

`run_main_experiment.py` calls the default `main` command, which currently means
`main --scale medium` unless overridden through `runner.py` directly.

### Outputs

A normal run writes a timestamped directory under `outputs/`, unless
`--output-dir` is supplied:

```text
outputs/<run_name>_YYYYMMDD_HHMMSS/
  raw_metrics.csv
  aggregated_metrics.csv
  selection_frequencies.csv
  tradeoff_raw.csv
  tradeoff_summary.csv
  winner_switch_points.csv
  manifests/run_config.yaml
  plots/*.png
  plots/*.pdf
```

The raw table is per replication, procedure, and `r_eval`. The aggregated tables
summarize stochastic robust MSE, SR-CV scores, training time, best epochs, and
SR-CV/oracle agreement. The plot set includes robust-risk curves, selected
procedure frequencies, phase diagrams, agreement heatmaps, and clean-endpoint
versus robust-risk tradeoff plots.

### Multi-GPU Runs

On a multi-GPU server, the same runner can split grid points across devices:

```bash
python scripts/runner.py multi-gpu smoke --devices 0,1 --workers-per-gpu 1
python scripts/runner.py multi-gpu main --scale lite --devices 0,1 --workers-per-gpu 1
python scripts/runner.py multi-gpu main --scale medium --devices 0,1 --workers-per-gpu 1
```

The final output directory has the same public CSV and plot files as a normal
run, with additional worker diagnostics under `partials/` and
`manifests/multi_gpu.yaml`.

## 3. CCPP Case Study

The CCPP directory contains two independent Combined Cycle Power Plant
model-selection pipelines. SRCV is the stochastic robust case study currently
used by the manuscript. AACV is a separate aligned-adversarial plug-in
experiment with its own configurations, outputs, figures, and verification.

### Data

The data file is stored at:

```text
ccpp_case_study/data/Folds5x2_pp.xlsx
```

The configuration records the expected checksum, source URL, required columns,
and row count. The experiment uses the feature columns `AT`, `V`, `AP`, and
`RH`, with `PE` as the response.

### Candidate Models

The implemented candidate order is:

```text
OLS, RidgeCV, LassoCV, KNN, RandomForest, HistGB
```

Ties are resolved by this fixed order, which is recorded in the configuration
files.

### Setup

```bash
cd ccpp_case_study
python -m pip install -r requirements.txt
```

The required Python packages are `matplotlib`, `numpy`, `openpyxl`, `pandas`,
`scikit-learn`, `scipy`, `tqdm`, and `pytest`.

### Quick Verification of Existing Outputs

```bash
python -m ccpp_repro verify --config configs/ccpp_full.json --run-dir outputs/full_run --figure-dir outputs/full_figures
```

### Plot-Only Reproduction

```bash
python -m ccpp_repro plot-only --config configs/ccpp_full.json --run-dir outputs/full_run --figure-dir outputs/plot_only_figures
python -m ccpp_repro verify --config configs/ccpp_full.json --run-dir outputs/full_run --figure-dir outputs/plot_only_figures
```

This path recomputes the radius-mismatch post-processing from the CSV files and
then redraws:

```text
fig_ccpp_case_study_main.pdf
fig_ccpp_case_study_main.png
fig_ccpp_radius_mismatch.pdf
fig_ccpp_radius_mismatch.png
```

### Smoke Test From Raw Data

```bash
python -m ccpp_repro run --config configs/ccpp_smoke.json --output-dir outputs/smoke_run --figure-dir outputs/smoke_figures --no-progress
python -m ccpp_repro verify --config configs/ccpp_smoke.json --run-dir outputs/smoke_run --figure-dir outputs/smoke_figures
```

### Full Reproduction From Raw Data

The full configuration uses 20 perturbation radii, 25 outer repeats, 15 inner
repeats, and 6 candidate models. It fits:

```text
20 * 25 * 15 * 6 = 45,000 candidate models
```

Run it in a fresh output directory if you want to preserve the included
outputs:

```bash
python -m ccpp_repro run --config configs/ccpp_full.json --output-dir outputs/full_run_rerun --figure-dir outputs/full_figures_rerun
python -m ccpp_repro verify --config configs/ccpp_full.json --run-dir outputs/full_run_rerun --figure-dir outputs/full_figures_rerun
```

The full run can take several hours on a typical workstation.

### CCPP Output Files

A successful CCPP run writes:

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

The `radius_mismatch_posthoc/` directory contains the post-processing tables
used for the validation-radius mismatch figure. It does not refit models; it
recombines selected models at validation radius `r_val` with outer-test robust
scores at evaluation radius `r_eval`.

### Independent AACV Pipeline

AACV freezes HistGB as its global working reference learner from the canonical
SRCV radius-zero clean-risk analysis (25 outer by 15 inner splits). The frozen
selection and source fingerprints are stored in
`configs/ccpp_aacv_reference.json`. Because selection and AACV reuse the same
CCPP dataset, this is a case-study design decision rather than independent
external validation.

Within each AACV inner training split, five-fold fold-local cross-fitting
constructs honest OOF residuals. The primary Gaussian-Hermite correction uses
the HistGB OOF residual scale, J=2, B=6 times that scale, and no clipping.
Predeclared sensitivities cover all six working reference learners, two
structural scale estimators, and four Hermite settings. Linear-model box
extrema are exact; KNN, RandomForest, and HistGB use deterministic multipath
continuous optimization with a strict inherited stronger envelope. The predeclared Smoke gate selected and froze `K=4`.

The held-out AACV best candidate is a qualified diagnostic benchmark; it does
not use the unknown regression function. See `ccpp_case_study/README.md`,
`ccpp_case_study/AACV_WORKING_ERROR_GUIDELINE.md`, and
`ccpp_case_study/AACV_IMPLEMENTATION_PLAN.md`, and
`ccpp_case_study/AACV_MULTIPATH_MEDIUM_PLAN.md` for the complete assumptions.

AACV smoke run:

~~~bash
cd ccpp_case_study
python -m ccpp_repro run-aacv --config configs/ccpp_aacv_smoke.json --output-dir outputs/aacv_smoke --figure-dir outputs/aacv_smoke_figures --no-progress
python -m ccpp_repro verify-aacv --config configs/ccpp_aacv_smoke.json --run-dir outputs/aacv_smoke --figure-dir outputs/aacv_smoke_figures
~~~

AACV Medium run (31 radii, one outer and one inner split, Full attack budget):

~~~bash
python -m ccpp_repro run-aacv --config configs/ccpp_aacv_medium.json --output-dir outputs/aacv_medium --figure-dir outputs/aacv_medium_figures
python -m ccpp_repro verify-aacv --config configs/ccpp_aacv_medium.json --run-dir outputs/aacv_medium --figure-dir outputs/aacv_medium_figures
~~~

The Medium and Full configurations share 31 radii from 0 to 1. Full uses 10
outer repeats and 5 inner repeats. Interrupted runs can be resumed with the
same command plus --resume.
The separate plot-aacv and verify-aacv commands regenerate and validate its
four figure families without invoking SRCV.

## Notes

Generated experiment outputs are intentionally excluded from the source tree.
If exact reference outputs are needed for publication review, archive them
outside Git, for example in a release artifact or data repository, and keep this
repository focused on code, configuration, tests, and lightweight documentation.
