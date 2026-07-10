# Robust Cross Validation Simulation

This project reproduces an experimental numerical simulation: in a two-dimensional linear regression example, it compares the model-selection performance of standard cross-validation under robust perturbations and generates raw results, summary tables, and figures.

The two candidate models being compared are:

- `A1`: Uses only the first feature, `x1`
- `A2`: Uses both features, `x1` and `x2`

The data-generating model is:

```text
y = beta1 * x1 + beta2 * x2 + eps
```

Here, `x1, x2` are standard normal features, and `eps` is Gaussian noise with a mean of 0. Across different sample sizes, training/validation split ratios, perturbation strengths `r`, and parameter settings, the code estimates whether the model selected by cross-validation is the same as the optimal model under the true robust risk.

## Implemented Features

The project currently implements three types of CV criteria:

- `adversarial`: Uses a pointwise worst-case perturbation score for the validation residuals.
- `aligned_adversarial`: Uses the aligned adversarial structure and estimates the cross-term with a localized Gaussian-Hermite absolute-mean estimator.
- `stochastic`: Adds a uniform random perturbation to `x2` during validation.

Each simulation run records:

- The CV score for each candidate model: `cv_a1`, `cv_a2`
- The closed-form robust risk for each candidate model: `loss_a1`, `loss_a2`
- The model selected by CV and the true robust-optimal model
- Whether the optimal model was selected
- The finite-sample switching thresholds: `threshold_adv`, `threshold_sto`
- `delta_cv = cv_a2 - cv_a1`

## Experiment Design

### Experiment 1

With `beta1=1.0`, `beta2=1.0`, and `sigma=0.5` fixed, this experiment examines model-selection behavior across different training ratios and perturbation radii.

Default settings:

- Sample sizes: `n = 1000, 2000`
- Training ratios: `0.1, 0.2, ..., 0.9`
- Adversarial radii: `0.0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.5, 1.75, 2.0`
- Stochastic radii: A dense set of values around `sqrt(3)`

### Experiment 2

With `beta1=1.0`, `beta2=1.0`, and `sigma=0.5` fixed, this experiment examines whether the model selected by CV converges to the robust-optimal model as the sample size increases.

Default settings:

- Sample sizes: `100, 300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000`
- Training ratios: `0.1, 0.2, ..., 0.9`
- Adversarial radii: `0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.2`
- Stochastic radii: `0.5, 0.8, 1.0, 1.5, sqrt(3), 2.0, 2.2`

## Project Structure

```text
robust_cv_sim/
  simulation.py   # Data generation, model fitting, CV criteria, closed-form risks, and experiment-row generation
  reporting.py    # Writing raw results, generating summary tables, and plotting
  runner.py       # Entry point for chunked runs, with resume support and a progress bar
  cli.py          # Simple entry point for non-chunked runs
tests/
  test_simulation.py
  test_runner.py
outputs/          # Default experiment output directory
```

## Installing Dependencies

Python 3.11 or later is recommended.

```bash
python -m pip install -e .
python -m pip install numpy pandas matplotlib seaborn rich pytest
```

The current `pyproject.toml` declares only package metadata and pytest configuration. Runtime dependencies must be installed using the commands above.

## Running Experiments

Using the chunked runner is recommended. It supports a progress bar, resuming interrupted runs, and merging chunks, making it suitable for full experiments.

Run all experiments:

```bash
python -m robust_cv_sim.runner
```

Run only Experiment 2:

```bash
python -m robust_cv_sim.runner --experiments experiment_2
```

Reduce the number of repetitions for a quick smoke test:

```bash
python -m robust_cv_sim.runner --experiments experiment_2 --reps 10 --output-dir smoke_outputs
```

Skip plotting and generate only the raw results and summary:

```bash
python -m robust_cv_sim.runner --experiments experiment_2 --skip-plots
```

Default arguments:

- `--reps 1000`
- `--seed 20260319`
- `--output-dir outputs`
- `--resume` is enabled by default

To ignore existing chunks and rerun the experiment:

```bash
python -m robust_cv_sim.runner --experiments experiment_2 --no-resume
```

## Output Files

Each experiment creates a separate subdirectory under the output directory. For example:

```text
outputs/
  experiment_2/
    experiment_2_raw.csv.gz
    experiment_2_summary.csv
    experiment_2_threshold_summary.csv
    plots/
      exp2_adversarial_trainratio_0.1_consistency.png
      exp2_adversarial_trainratio_0.1_delta_cv_boxplots.png
      ...
```

The main files are:

- `*_raw.csv.gz`: Complete results for every repetition, configuration, and `r` value.
- `*_summary.csv`: Means, selection probabilities, and confidence intervals aggregated by experiment configuration.
- `*_threshold_summary.csv`: A distributional summary of the finite-sample switching thresholds.
- `plots/`: Automatically generated visualizations.

The plots for Experiment 2 include:

- Consistency plots: The horizontal axis is `n`, the vertical axis is `P(select robust-optimal)`, and different curves correspond to different values of `r`.
- `delta_cv` boxplot: Shows the distribution of `cv_a2 - cv_a1` across different sample sizes and perturbation strengths.

## Using the Python API

Small-scale experiments can be run directly through the Python API:

```python
from robust_cv_sim import SimulationConfig, run_experiment_2

config = SimulationConfig(
    reps=10,
    split_grid=(0.5,),
    exp2_n_grid=(100, 500, 1000),
)

df = run_experiment_2(config)
print(df.head())
```

The perturbation radii for Experiment 2 can also be overridden:

```python
config = SimulationConfig(
    exp2_radius_grid_adv=(0.5, 0.7, 0.8, 1.0),
)
```

## Tests

Run all tests:

```bash
python -m pytest
```

The current tests cover:

- The adversarial and stochastic CV formulas
- Basic numerical properties of the Gaussian-Hermite estimator
- The closed-form robust-risk formulas
- Chunk enumeration for both experiments
- Logic for writing and merging chunks and resuming runs
- Summary-generation and plotting helper functions
- The default set of adversarial radii for Experiment 2
