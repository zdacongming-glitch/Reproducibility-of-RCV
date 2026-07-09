# CCPP Radius-Mismatch Post-Processing Report

Generated: 2026-07-04T17:08:22

## Inputs

- Run directory: `D:\博士学习\科研\git_overleaf\RCV\Case Study\results\run_20260522_004859`
- Required file: `outer_summary.csv`
- Optional file used when available: `inner_validation_scores.csv`

## Design

- Candidate models: OLS, RidgeCV, LassoCV, KNN, RandomForest, HistGB
- Validation radii: 0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.17, 0.18, 0.19, 0.195, 0.2, 0.205, 0.21, 0.22, 0.24, 0.27, 0.32, 0.4, 0.5
- Evaluation radii: 0, 0.025, 0.1, 0.15, 0.19, 0.2, 0.21, 0.24, 0.32, 0.4, 0.5
- Outer replications: 25
- Mismatch regret is computed by selecting the model at `r_val` and evaluating its outer-test stochastic robust score at `r_eval`, then subtracting the `r_eval` diagnostic oracle score.

## Feasibility Checks

- Complete paired radius/outer grid: passed (500 rows).
- Clean-CV radius-invariance check: max clean_cv span across radii = 5.86198e-14; tolerance = 1e-10.
- Clean final-choice radius-invariance check: passed (0 unstable outers).

## Main SRCV Mismatch Summary

|   r_eval |   best_r_val |   best_regret_mean |   best_regret_mcse |   low_regret_r_val_min |   low_regret_r_val_max | low_regret_r_val_values                                |
|---------:|-------------:|-------------------:|-------------------:|-----------------------:|-----------------------:|:-------------------------------------------------------|
|    0     |         0    |         0          |         0          |                   0    |                  0.15  | 0,0.025,0.05,0.075,0.1,0.125,0.15                      |
|    0.025 |         0    |         0          |         0          |                   0    |                  0.15  | 0,0.025,0.05,0.075,0.1,0.125,0.15                      |
|    0.1   |         0    |         0          |         0          |                   0    |                  0.15  | 0,0.025,0.05,0.075,0.1,0.125,0.15                      |
|    0.15  |         0    |         0.00912924 |         0.00534141 |                   0    |                  0.15  | 0,0.025,0.05,0.075,0.1,0.125,0.15                      |
|    0.19  |         0    |         0.031245   |         0.0240155  |                   0    |                  0.195 | 0,0.025,0.05,0.075,0.1,0.125,0.15,0.17,0.18,0.19,0.195 |
|    0.2   |         0.19 |         0.0507961  |         0.0312848  |                   0    |                  0.195 | 0,0.025,0.05,0.075,0.1,0.125,0.15,0.17,0.18,0.19,0.195 |
|    0.21  |         0.19 |         0.0955941  |         0.0407472  |                   0    |                  0.195 | 0,0.025,0.05,0.075,0.1,0.125,0.15,0.17,0.18,0.19,0.195 |
|    0.24  |         0.24 |         0.0634808  |         0.0296532  |                   0.22 |                  0.5   | 0.22,0.24,0.27,0.32,0.4,0.5                            |
|    0.32  |         0.27 |         0          |         0          |                   0.27 |                  0.5   | 0.27,0.32,0.4,0.5                                      |
|    0.4   |         0.27 |         0          |         0          |                   0.27 |                  0.5   | 0.27,0.32,0.4,0.5                                      |
|    0.5   |         0.27 |         0          |         0          |                   0.27 |                  0.5   | 0.27,0.32,0.4,0.5                                      |

## Selected Matched and Fixed-Radius Diagnostics

| selector   |   r_eval |   regret_mean |   regret_mcse |   oracle_match_rate |   oracle_match_mcse | selected_model_mode   |
|:-----------|---------:|--------------:|--------------:|--------------------:|--------------------:|:----------------------|
| SRCV       |    0     |    0          |    0          |                1    |           0         | HistGB                |
| clean      |    0     |    0          |    0          |                1    |           0         | HistGB                |
| SRCV       |    0.025 |    0          |    0          |                1    |           0         | HistGB                |
| clean      |    0.025 |    0          |    0          |                1    |           0         | HistGB                |
| SRCV       |    0.1   |    0          |    0          |                1    |           0         | HistGB                |
| clean      |    0.1   |    0          |    0          |                1    |           0         | HistGB                |
| SRCV       |    0.15  |    0.00912924 |    0.00534141 |                0.88 |           0.0649923 | HistGB                |
| clean      |    0.15  |    0.00912924 |    0.00534141 |                0.88 |           0.0649923 | HistGB                |
| SRCV       |    0.19  |    0.044419   |    0.0260244  |                0.76 |           0.0854166 | HistGB                |
| clean      |    0.19  |    0.031245   |    0.0240155  |                0.84 |           0.0733212 | HistGB                |
| SRCV       |    0.2   |    0.144678   |    0.0580148  |                0.6  |           0.0979796 | HistGB                |
| clean      |    0.2   |    0.0515862  |    0.0314018  |                0.72 |           0.0897998 | HistGB                |
| SRCV       |    0.21  |    0.138945   |    0.0522335  |                0.52 |           0.09992   | HistGB                |
| clean      |    0.21  |    0.101943   |    0.041372   |                0.56 |           0.0992774 | HistGB                |
| SRCV       |    0.24  |    0.0634808  |    0.0296532  |                0.8  |           0.08      | KNN                   |
| clean      |    0.24  |    0.322818   |    0.0672941  |                0.2  |           0.08      | HistGB                |
| SRCV       |    0.32  |    0          |    0          |                1    |           0         | KNN                   |
| clean      |    0.32  |    1.50102    |    0.0822418  |                0    |           0         | HistGB                |
| SRCV       |    0.4   |    0          |    0          |                1    |           0         | KNN                   |
| clean      |    0.4   |    2.82634    |    0.0880687  |                0    |           0         | HistGB                |
| SRCV       |    0.5   |    0          |    0          |                1    |           0         | KNN                   |
| clean      |    0.5   |    4.70692    |    0.0950058  |                0    |           0         | HistGB                |

## Reproducibility

- Python: 3.11.7
- Platform: Windows-10-10.0.26200-SP0
- pandas: 2.1.4
- numpy: 1.26.4
- matplotlib: 3.8.0
- No model refitting was performed.
- No interpolation was used for absent radii.
