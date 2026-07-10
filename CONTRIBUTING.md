# Contributing

This repository is a reproducibility package for three experiments. Keep
changes focused on reproducible source material and avoid committing artifacts
that can be regenerated locally.

## Repository Policy

Commit source code, experiment configurations, tests, documentation, dependency
metadata, and required input data. Do not commit local caches, virtual
environments, generated experiment outputs, rerun directories, or model
checkpoints.

The repository and experiment-specific `.gitignore` files are the primary
safeguards. Do not use `git add -f` to bypass them unless the collaborators
have explicitly agreed to archive a particular artifact in Git.

## Nested-Linear Simulation

Commit:

```text
nested_linear_example/README.md
nested_linear_example/pyproject.toml
nested_linear_example/robust_cv_sim/*.py
nested_linear_example/tests/*.py
```

Do not commit:

```text
nested_linear_example/.agents/
nested_linear_example/.pytest_cache/
nested_linear_example/**/__pycache__/
nested_linear_example/outputs/
nested_linear_example/aligned_smoke/
nested_linear_example/smoke_outputs/
nested_linear_example/outputs_rerun/
```

## Neural-Network Teacher-Student Simulation

Commit:

```text
neaural_network_experiment/README.md
neaural_network_experiment/requirements.txt
neaural_network_experiment/实验方案.md
neaural_network_experiment/configs/**/*.yaml
neaural_network_experiment/scripts/*.py
neaural_network_experiment/src/**/*.py
neaural_network_experiment/tests/*.py
```

Do not commit:

```text
neaural_network_experiment/.pytest_cache/
neaural_network_experiment/**/__pycache__/
neaural_network_experiment/.venv/
neaural_network_experiment/outputs/
neaural_network_experiment/smoke_outputs/
neaural_network_experiment/outputs_rerun/
neaural_network_experiment/**/*.pt
neaural_network_experiment/**/*.pth
neaural_network_experiment/**/*.ckpt
```

## CCPP Case Study

Commit:

```text
ccpp_case_study/.gitignore
ccpp_case_study/README.md
ccpp_case_study/requirements.txt
ccpp_case_study/configs/*.json
ccpp_case_study/ccpp_repro/*.py
ccpp_case_study/data/Folds5x2_pp.xlsx
```

Do not commit:

```text
ccpp_case_study/.pytest_cache/
ccpp_case_study/**/__pycache__/
ccpp_case_study/outputs/
```

## Generated Artifacts

Generated CSV files, figures, manifests, worker diagnostics, and model
checkpoints should remain outside the Git source tree. If exact reference
outputs are needed for publication review, place them in a GitHub release or an
external archival repository and document the corresponding code version,
configuration, and random seed.

## Before Opening a Pull Request

1. Inspect `git status` and confirm that only intended source files are staged.
2. Run the tests or smoke command for every experiment affected by the change.
3. Confirm that no generated outputs, caches, environments, or checkpoints are
   staged.
4. Preserve deterministic seeds and record any intentional configuration
   changes.
5. Update the relevant README when commands, dependencies, outputs, or expected
   behavior change.
