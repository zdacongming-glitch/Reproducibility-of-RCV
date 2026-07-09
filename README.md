# Reproducibility Package

This repository is the top-level reproducibility package for three experiments.
At the current snapshot, only the CCPP case study is included. The two
simulation experiments are reserved below with stable directory and command
interfaces so that the final README can grow into a single entry point for all
three experiments.

## Experiment Status

| Experiment | Status | Directory | Main artifacts |
| --- | --- | --- | --- |
| CCPP case study | Available | `ccpp_case_study/` | `fig_ccpp_case_study_main`, `fig_ccpp_radius_mismatch` |
| Nested-linear simulation | Interface reserved | `nested_linear_simulation/` | To be added |
| Teacher-student simulation | Interface reserved | `teacher_student_simulation/` | To be added |

## Common Interface

Each experiment is expected to expose the same three-stage interface:

```text
python -m <experiment_package> run --config <config> --output-dir <run_dir> --figure-dir <figure_dir>
python -m <experiment_package> plot-only --config <config> --run-dir <run_dir> --figure-dir <figure_dir>
python -m <experiment_package> verify --config <config> --run-dir <run_dir> --figure-dir <figure_dir>
```

The CCPP case study already implements this interface through
`python -m ccpp_repro`. The two simulation sections below reserve the same
shape for the code that will be added later.

## 1. CCPP Case Study

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

## 2. Nested-Linear Simulation

Status: code not yet included.

Reserved directory:

```text
nested_linear_simulation/
```

Reserved package entry point:

```text
python -m nested_linear_repro
```

Reserved command interface:

```bash
cd nested_linear_simulation
python -m pip install -r requirements.txt
python -m nested_linear_repro run --config configs/smoke.json --output-dir outputs/smoke_run --figure-dir outputs/smoke_figures --no-progress
python -m nested_linear_repro verify --config configs/smoke.json --run-dir outputs/smoke_run --figure-dir outputs/smoke_figures
python -m nested_linear_repro run --config configs/full.json --output-dir outputs/full_run --figure-dir outputs/full_figures
python -m nested_linear_repro verify --config configs/full.json --run-dir outputs/full_run --figure-dir outputs/full_figures
```

When this simulation is added, this section should document the data-generating
process, candidate model classes, validation procedure, random seeds, expected
figures, and verification signatures.

## 3. Teacher-Student Simulation

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

## Notes for Adding the Remaining Experiments

To keep the final package consistent, each remaining experiment should include:

- `README.md` with experiment-specific details;
- `requirements.txt`;
- `configs/smoke.json` for a quick end-to-end test;
- `configs/full.json` for the manuscript run;
- `outputs/` or `reference/` artifacts needed for plot-only reproduction;
- a Python package exposing `run`, `plot-only`, and `verify` subcommands;
- deterministic seeds and explicit verification checks for row counts,
  numerical signatures, and generated figures.
