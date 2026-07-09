# Reproducibility Package

This repository is the top-level reproducibility package for three experiments.
The current snapshot contains the first simulation experiment and the third
case-study experiment. The second simulation experiment is still reserved.

## Experiment Status

| Order | Experiment | Status | Directory | Main artifacts |
| --- | --- | --- | --- | --- |
| 1 | Nested-linear simulation | Available | `nested_linear_example/` | raw CSVs, summary CSVs, threshold summaries, PNG plots |
| 2 | Teacher-student simulation | Interface reserved | `teacher_student_simulation/` | To be added |
| 3 | CCPP case study | Available | `ccpp_case_study/` | `fig_ccpp_case_study_main`, `fig_ccpp_radius_mismatch` |

## GitHub Upload Policy

For `nested_linear_example/`, keep the reproducible source package and tests in
Git:

```text
nested_linear_example/README.md
nested_linear_example/pyproject.toml
nested_linear_example/robust_cv_sim/*.py
nested_linear_example/tests/*.py
```

Do not upload local caches or generated experiment outputs:

```text
nested_linear_example/.agents/
nested_linear_example/.pytest_cache/
nested_linear_example/**/__pycache__/
nested_linear_example/outputs/
nested_linear_example/aligned_smoke/
nested_linear_example/smoke_outputs/
nested_linear_example/outputs_rerun/
```

The reason is practical: `nested_linear_example/outputs/` currently contains
404 generated files and is about 390 MB. Its largest raw files are
`experiment_3_raw.csv.gz` at about 185 MB, `experiment_2_raw.csv.gz` at about
84 MB, and `experiment_1_raw.csv.gz` at about 31 MB. These files are
reconstructible from the source code and fixed seeds, so they should stay out
of the Git repository. If a frozen reference run is needed for a paper artifact,
put it in a separate release or archival deposit rather than in the source
tree.

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

The package has three internal sweeps, exposed as `experiment_1`,
`experiment_2`, and `experiment_3` in the runner:

- `experiment_1`: fixed `beta1=1.0`, `beta2=1.0`, `sigma=0.5`; varies sample
  size, train/validation ratio, and robust radius near the adversarial and
  stochastic switching thresholds.
- `experiment_2`: fixed `beta1=1.0`, `beta2=1.0`, `sigma=0.5`; varies sample
  size from 100 to 5000 to study whether CV selection converges to the
  robust-optimal model.
- `experiment_3`: varies signal strength `beta2` and noise level `sigma` to
  study sensitivity of model selection.

### Setup

From the repository root:

```bash
cd nested_linear_example
python -m pip install -e .
python -m pip install numpy pandas matplotlib seaborn rich pytest
```

`pyproject.toml` currently declares package metadata and pytest settings. The
runtime dependencies should be installed with the second command above.

### Tests

Run the unit tests:

```bash
python -m pytest
```

The current suite covers formulas for adversarial and stochastic CV, the
Gaussian-Hermite estimators, closed-form robust risks, chunk enumeration,
resume/merge behavior, summary generation, and plot file patterns.

### Smoke Run

Use a small number of replications for a quick end-to-end check:

```bash
python -m robust_cv_sim.runner --experiments experiment_2 --reps 10 --output-dir smoke_outputs --skip-plots
```

This generates raw and summary CSVs without producing the full plot set.

### Full Run

The default runner executes all three internal sweeps with 1000 Monte Carlo
replications per configuration and seed `20260319`:

```bash
python -m robust_cv_sim.runner --experiments experiment_1 experiment_2 experiment_3 --reps 1000 --seed 20260319 --output-dir outputs_rerun
```

Useful runner options:

- `--skip-plots`: write raw and summary CSVs only;
- `--keep-chunks`: keep per-chunk CSV files after merging;
- `--no-resume`: ignore existing completed chunks and rerun everything.

### Outputs

Each internal sweep writes an experiment subdirectory:

```text
outputs_rerun/
  experiment_1/
    experiment_1_raw.csv.gz
    experiment_1_summary.csv
    experiment_1_threshold_summary.csv
    plots/
  experiment_2/
    experiment_2_raw.csv.gz
    experiment_2_summary.csv
    experiment_2_threshold_summary.csv
    plots/
  experiment_3/
    experiment_3_raw.csv.gz
    experiment_3_summary.csv
    experiment_3_threshold_summary.csv
    plots/
```

The raw files contain one row per replication, configuration, criterion, and
radius. The summary files aggregate selection probabilities, robust losses,
confidence intervals, and threshold summaries. Plots are generated from the raw
CSV files by `robust_cv_sim.reporting`.

## 2. Teacher-Student Simulation

Status: code not yet included.

Reserved directory:

```text
teacher_student_simulation/
```

Reserved package entry point:

```text
python -m teacher_student_repro
```

Reserved command interface:

```bash
cd teacher_student_simulation
python -m pip install -r requirements.txt
python -m teacher_student_repro run --config configs/smoke.json --output-dir outputs/smoke_run --figure-dir outputs/smoke_figures --no-progress
python -m teacher_student_repro verify --config configs/smoke.json --run-dir outputs/smoke_run --figure-dir outputs/smoke_figures
python -m teacher_student_repro run --config configs/full.json --output-dir outputs/full_run --figure-dir outputs/full_figures
python -m teacher_student_repro verify --config configs/full.json --run-dir outputs/full_run --figure-dir outputs/full_figures
```

When this simulation is added, this section should document the teacher model,
student candidates, sample-size and perturbation settings, random seeds,
expected figures, and verification signatures.

## 3. CCPP Case Study

The CCPP case study is a self-contained reproduction package for the Combined
Cycle Power Plant data experiment. It evaluates clean cross-validation and
stable/robust cross-validation across perturbation radii, compares selected
models against the empirical oracle, and generates the two case-study figures.

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

From the repository root:

```bash
cd ccpp_case_study
python -m pip install -r requirements.txt
```

The required Python packages are `matplotlib`, `numpy`, `openpyxl`, `pandas`,
`scikit-learn`, and `tqdm`.

### Quick Verification of Existing Outputs

The repository currently includes generated full-run CSV outputs and figures
under `ccpp_case_study/outputs/`. To verify them without refitting models:

```bash
python -m ccpp_repro verify --config configs/ccpp_full.json --run-dir outputs/full_run --figure-dir outputs/full_figures
```

### Plot-Only Reproduction

To regenerate the manuscript figures from existing full-run CSV outputs:

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

The smoke configuration runs the same pipeline on a tiny radius grid with one
outer repeat and two inner repeats:

```bash
python -m ccpp_repro run --config configs/ccpp_smoke.json --output-dir outputs/smoke_run --figure-dir outputs/smoke_figures --no-progress
python -m ccpp_repro verify --config configs/ccpp_smoke.json --run-dir outputs/smoke_run --figure-dir outputs/smoke_figures
```

This checks data loading, model fitting, validation scoring, test scoring,
aggregation, radius-mismatch post-processing, plotting, and verification.

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

## Notes for Adding the Remaining Simulation

To keep the final package consistent, the remaining simulation should include:

- `README.md` with experiment-specific details;
- dependency metadata or `requirements.txt`;
- a smoke command for a quick end-to-end test;
- a full-run command for the manuscript run;
- optional archived artifacts outside the Git source tree if plot-only
  reproduction is required;
- deterministic seeds and explicit verification checks for row counts,
  numerical signatures, and generated figures.