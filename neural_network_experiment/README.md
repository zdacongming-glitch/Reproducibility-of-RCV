# Robust Cross-Validation Under Attacked Covariates

PyTorch codebase for teacher-student regression experiments that compare training procedures under attacked covariates and select procedures with stochastic robust cross-validation (SR-CV).

## Features

- Synthetic teacher-student regression data with `core`, `fragile`, and `noise` feature blocks
- One shared MLP architecture for `erm`, `at`, and `sr`
- `core_only` baseline that only uses `X_core`
- Restricted `L_inf` and restricted `L2` perturbations on fragile features only
- SR-CV procedure selection with common random perturbations across procedures
- Main experiment produces both robust-risk curves and phase diagrams
- CSV summaries, selection frequencies, switch points, and matplotlib plots
- Unified CLI runner with timestamped output directories and progress bars

## Setup

Python `3.11` is the target environment.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Conda example:

```bash
conda activate robustcv311
python scripts/runner.py smoke
```

## Reproducibility And Stochastic Variation

The configured base seed fixes the synthetic data, teacher network, CV splits,
and common validation/test perturbations. It does not currently reseed the
global PyTorch random-number generator before every model fit. Consequently,
independent reruns can differ in MLP initialization, DataLoader shuffling, SR
training perturbations, PGD random starts, validation/early-stopping paths, and
some GPU numerical operations. Changing the number of GPUs or workers can also
change the order in which these random numbers are consumed.

These effects are not expected to materially change the main numerical patterns
or scientific conclusions. In repeated `lite` runs with the same resolved
configuration, the median relative difference in MSE was approximately `2%` to
`3%`; larger local differences and procedure-label changes occurred mainly near
ties. The `full` experiment averages `16` replications, so its robust-risk curves
are expected to be more stable. Nevertheless, exact pointwise MSE values, phase
diagram boundaries, agreement-heatmap cells, CSV hashes, and PDF bytes are not
guaranteed to match across independent reruns.

The code should therefore be interpreted as a statistical and qualitative
replication package rather than a bit-for-bit artifact reproduction. For the
execution topology used by the paper's archived full run, use:

```bash
python scripts/runner.py multi-gpu main --scale full --devices 0,1 --workers-per-gpu 1
```

## Commands

```bash
python scripts/runner.py smoke
python scripts/runner.py main --scale lite
python scripts/runner.py main --scale medium --workers 4
python scripts/runner.py main --scale full --workers 6
python scripts/runner.py plots --input-dir outputs/main_medium_YYYYMMDD_HHMMSS
```

Convenience wrappers:

```bash
python scripts/run_smoke_test.py
python scripts/run_main_experiment.py
python scripts/make_all_plots.py --input-dir outputs/main_medium_YYYYMMDD_HHMMSS
```

The `main` command defaults to `--scale medium`.

## Procedures

- `erm`: standard MSE on all features
- `at`: adversarial training against the training threat model at radius `r_train`
- `sr`: stochastic robust training with random perturbations supported on the training threat set
- `core_only`: same MLP shape but only sees the core block

The selection target is the training procedure, not the architecture.

## DGP And Experiment Logic

Common data-generating process:

- `X = (X_core, X_frag, X_noise)` with `d_core = 4`, `d_frag = 4`, `d_noise = 8`
- Blocks are sampled from independent standard Gaussians and block-standardized
- `Y = g_*(X_core) + lambda h_*(X_frag) + epsilon`
- `epsilon ~ N(0, sigma^2)` with `sigma = 0.25`
- `g_*` and `h_*` are two-layer ReLU teacher networks with `m_core = 16`, `m_frag = 16`
- The teacher is fixed across replications by default

Main experiment logic:

- A training grid point is `(n_total, lambda, r_train, threat_model)`.
- For each replication of a training grid point, the code generates one `D_train_cv` of size `n_total` and one independent `D_test` of size `n_total`.
- `D_train_cv` is randomly split into 5 mutually disjoint folds.
- CV split `k` uses fold `k` as the inner training set, so each split trains on about `20%` of `D_train_cv`.
- The other 4 folds form the validation set, so each split validates on about `80%` of `D_train_cv`.
- Each split trains `erm`, `at`, `sr`, and `core_only` once under the fixed training radius `r_train`.
- The same split-trained models are used for both validation SR-CV scoring and `D_test` stochastic MSE evaluation.
- There is no full-`D_train_cv` final retraining step.
- `r_eval` is not used to regenerate data or retrain models.
- SR-CV uses selection radius equal to evaluation radius: for each `r_eval`, the code computes SR-CV scores under that same radius.
- For each `r_eval`, `cv_srcv_score` is the mean validation SR-CV score over the 5 splits.
- For each `r_eval`, `stochastic_robust_mse` is the mean `D_test` stochastic MSE over the same 5 split-trained models.
- `selected_procedure_srcv` is the procedure with the lowest split-averaged SR-CV score.
- `oracle_best_procedure` is the procedure with the lowest split-averaged `D_test` stochastic MSE.
- `r_eval < r_train` is allowed and is not filtered.

Clean risk convention:

- There is no separate `clean_mse` output column.
- `stochastic_robust_mse` at `r_eval = 0` is the clean-risk endpoint.
- Tradeoff plots use this `r_eval = 0` endpoint on the x-axis.

## Main Scales

All scales use `threat_model = {linf, l2}`, `epochs = 50`, `lr = 0.1`, `patience = 10`, `cv.repeats = 5`, `cv.holdout_ratio = 0.8`, and candidate procedures `{erm, at, sr, core_only}`.

### `lite`

- Training grid points: `16`
- `n_total = {1000, 2000}`
- `lambda = {0.0, 1.0}`
- `r_train = {0.1, 1.0}`
- `r_eval = {0.0, 0.05, 0.1, 0.2}`
- `num_replications = 3`
- Default workers: `2`
- Output directory pattern: `outputs/main_lite_YYYYMMDD_HHMMSS`

### `medium`

- Training grid points: `48`
- `n_total = {1000, 2000}`
- `lambda = {0.0, 0.3, 0.6, 0.9, 1.2, 1.5}`
- `r_train = {0.05, 0.1}`
- `r_eval = {0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0}`
- `num_replications = 8`
- Default workers: `4`
- Output directory pattern: `outputs/main_medium_YYYYMMDD_HHMMSS`

### `full`

- Training grid points: `600`
- `n_total = {1000, 2000}`
- `lambda = {0.0, 0.05, 0.1, 0.15, 0.2, 0.23, 0.25, 0.27, 0.3, 0.33, 0.36, 0.4, 0.45, 0.5, 0.6, 0.75, 0.9, 1.05, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.55, 2.7, 2.8, 2.9, 3.0}`
- `r_train = {0.05, 0.1, 0.2, 0.5, 1.0}`
- `r_eval = {0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5, 1.65, 1.8, 1.95, 2.1, 2.25, 2.4, 2.55, 2.7, 2.85, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0}`
- `num_replications = 16`
- Default workers: `6`
- Output directory pattern: `outputs/main_full_YYYYMMDD_HHMMSS`

## Progress Bars

The runner shows two progress bars:

- `Training grid points`: outer progress over `(n_total, lambda, r_train, threat_model)`
- `Replications[i/N|w=workers]`: inner progress over replications for the current training grid point

There is no per-`r_eval` progress bar because all `r_eval` values are evaluated inside each replication and per-radius bars would be noisy under multithreading.

## Outputs

Each run writes to:

```text
outputs/<run_name>_YYYYMMDD_HHMMSS
```

If `--output-dir` is provided, that exact directory is used.

Files:

- `raw_metrics.csv`: per replication, procedure, and `r_eval`; stochastic MSE is averaged over the 5 split-trained models
- `aggregated_metrics.csv`: means and standard errors by procedure/configuration
- `selection_frequencies.csv`: SR-CV selected procedure frequencies
- `tradeoff_raw.csv`: raw robust-risk tradeoff table using the `r_eval=0` endpoint
- `tradeoff_summary.csv`: aggregated robust-risk tradeoff table
- `winner_switch_points.csv`: selected-procedure switch points over `r_eval`
- `manifests/run_config.yaml`: fully resolved config
- `plots/*.png` and `plots/*.pdf`

Main raw columns include:

- `replication_id`
- `seed`
- `teacher_seed`
- `procedure`
- `selected_procedure_srcv`
- `cv_srcv_score`
- `cv_srcv_score_std`
- `n_total`
- `lambda_value`
- `r_train`
- `r_eval`
- `threat_model`
- `stochastic_robust_mse`
- `training_time_sec`
- `best_epoch`
- `best_metric`
- `oracle_best_procedure`
- `selected_srcv_matches_oracle`

`training_time_sec` is the total training time across the 5 split-trained models for that procedure in one replication. `best_epoch` and `best_metric` are averages over those 5 split-trained models.

Plots:

- stochastic robust risk vs `r_eval`
- SR-CV selected procedure frequency vs `r_eval`
- phase diagram over `(r_eval, lambda)`
- clean-endpoint vs stochastic robust risk tradeoff

## Suggested Run Order

```bash
conda run --no-capture-output -n robustcv311 python -u scripts/runner.py smoke
conda run --no-capture-output -n robustcv311 python -u scripts/runner.py main --scale lite --workers 2
conda run --no-capture-output -n robustcv311 python -u scripts/runner.py main --scale medium --workers 4
```

Use `main --scale full` only after `lite` or `medium` results look sensible.

## Multi-GPU Runs

The standard `smoke`, `main`, and `plots` commands are unchanged. On a server with
multiple visible CUDA devices, use the `multi-gpu` entrypoint to split training
grid points across independent worker processes and merge the final outputs:

```bash
python scripts/runner.py multi-gpu smoke --devices 0,1 --workers-per-gpu 1
python scripts/runner.py multi-gpu main --scale lite --devices 0,1 --workers-per-gpu 1
python scripts/runner.py multi-gpu main --scale medium --devices 0,1 --workers-per-gpu 1
```

Each worker sets `CUDA_VISIBLE_DEVICES` before importing PyTorch, so the existing
`device: auto` setting maps each process to its assigned GPU. The final output
directory contains the same public CSV and plot files as a normal run, plus
`partials/` and `manifests/multi_gpu.yaml` for worker diagnostics.
During the run, each GPU worker prints its own grid-point progress bar and a
per-point replication progress bar.

## Runtime Notes

Runtime depends on CPU speed, memory bandwidth, worker count, and attack/training settings.

- `smoke`: usually a short end-to-end check
- `main --scale lite`: useful for quick visual validation
- `main --scale medium`: default working experiment
- `main --scale full`: dense experiment; expect substantially longer runtime

The largest cost is training, not plotting. Each replication trains `5 splits * 4 procedures` models for one training grid point, then reuses those models across all `r_eval` values. Dense radius grids are therefore much cheaper than retraining once per radius.
