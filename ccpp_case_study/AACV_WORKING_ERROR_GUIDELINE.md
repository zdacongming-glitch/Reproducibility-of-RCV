# Guideline for a Working Error Law in AACV

## 1. Purpose and scope

This document specifies how to construct a working random-error law for
aligned adversarial cross-validation (AACV) when the true regression function
and the error distribution are unknown. It applies to real-data case studies
such as the Combined Cycle Power Plant (CCPP) experiment.

The procedure is a plug-in analysis. It does not identify an unavailable true
regression function or prove that the fitted error law is the physical noise
law. All primary assumptions, model-selection rules, diagnostics, and
sensitivity analyses must be fixed before inspecting AACV held-out results.

Assume

```text
Y = f0(X) + xi,    E[xi | X] = 0.
```

The statistical conditional mean `f0(x) = E[Y | X=x]` may exist even when it
has no known formula. A data-driven learner is therefore called a **working
reference learner**, not the true regression model.

## 2. Hierarchy for obtaining the error law

Use the first available source in this hierarchy:

1. In a simulation with known `f0` and noise, use the known error law.
2. With genuine independent repeated responses at the same covariate values,
   estimate the error law from within-covariate variation.
3. With an independently calibrated physical model or external pilot data,
   use that source without consulting AACV validation or test outcomes.
4. Otherwise, globally preselect a working reference learner using the
   clean-risk protocol below and construct honest cross-fitted residuals.

The CCPP experiment uses item 4.

## 3. Global selection of the working reference learner

### 3.1 Predeclare the selection protocol

Before running AACV, fix:

- the nuisance candidate library;
- all preprocessing rules;
- the hyperparameter protocol;
- the repeated split seeds;
- clean out-of-sample squared error as the selection criterion;
- candidate ordering and deterministic tie-breaking;
- the rule for aggregating split-level scores.

For candidate algorithm `a` and clean validation split `s`, let

```text
R[a,s] = mean((Y_i - fhat[a,s](X_i))^2 over i in validation split s).
Rbar[a] = mean(R[a,s] over the predeclared repeated splits).
```

Choose one global algorithm

```text
a_star = argmin_a Rbar[a],
```

using the fixed candidate order to resolve an exact tie. Report every
candidate's `Rbar`, Monte Carlo standard error, difference from the winner,
and split-level win frequency.

The selected object is an algorithm and fitting protocol, not one model fitted
once to the complete dataset. The algorithm is refitted whenever the AACV
training data change.

### 3.2 Interpretation

The global winner means only:

> the learner with the smallest repeated clean prediction risk within the
> predeclared candidate family.

It is not evidence that the learner equals `f0`. If the winner and runner-up
differ by less than their Monte Carlo uncertainty, retain the deterministic
winner for the primary analysis and include the runner-up in reference-model
sensitivity analysis.

Do not reselect the working reference learner separately inside every AACV
inner split. Per-split reselection would make the residual sample a mixture of
different adaptive learners and would weaken the interpretation of a common
working error law.

### 3.3 Data-reuse disclosure

Independent pilot data are preferable. If global preselection and AACV use the
same case-study dataset, state this explicitly. Such preselection is a frozen
design decision relative to AACV, but it is not independent external
validation and must not be described as proof of a true model.

No AACV robust score, attack result, outer-test response, or final held-out
metric may influence global reference-learner selection.

## 4. Honest residual construction inside AACV

For every AACV outer/inner split:

1. Restrict the procedure to the current inner training observations.
2. Keep the globally selected reference algorithm fixed.
3. Partition the inner training observations into predeclared cross-fitting
   folds.
4. For each fold, fit preprocessing and the reference learner using only the
   other folds.
5. Predict the held-out fold and store

   ```text
   xi_hat_i = Y_i - fhat_ref^(-fold(i))(X_i).
   ```

6. Pool the held-out residuals only after every training observation has one
   honest prediction.

Neither inner validation responses nor outer-test responses may enter the
preprocessing, model fitting, residual construction, error-law estimation, or
diagnostic tuning.

## 5. Working Gaussian scale model

For the primary Gaussian-homoskedastic analysis, assume the working law

```text
xi | X ~ N(0, sigma^2).
```

Estimate the common scale from the cross-fitted residuals:

```text
sigma2_hat = sample_variance(xi_hat, ddof=1)
sigma_hat  = sqrt(sigma2_hat).
```

Apply only a predeclared numerical floor, record whether it was used, and never
select a floor from AACV outcomes. Report at least:

- residual mean and mean squared residual;
- `sigma2_hat` and `sigma_hat`;
- skewness and excess kurtosis;
- Jarque-Bera statistic and p-value;
- residual and squared-residual relationships with each covariate;
- the variance-floor flag.

Diagnostics assess the plausibility of the working law. They do not prove
Gaussianity, and they must not silently trigger a distributional change after
AACV results are observed. A material diagnostic failure changes the strength
of the scientific interpretation, not the preregistered primary calculation.

Cross-fitted residual variance obeys the conceptual decomposition

```text
E[(Y - fhat_ref(X))^2 | fhat_ref]
    = sigma^2 + E[(f0(X) - fhat_ref(X))^2 | fhat_ref]
```

under the corresponding independent-evaluation and homoskedastic assumptions.
It can therefore contain reference-model approximation error and should be
called a plug-in scale estimate.

## 6. Distribution-compatible AACV correction

The correction must match the predeclared working error family. Under the
Gaussian working law, use the Gaussian-Hermite construction

```text
psi_hat = sum_{m=0}^J g_{2m} B^(1-2m) sigma_hat^(2m)
                    H_{2m}((Y-c)/sigma_hat).
```

Fix `J`, `B/sigma_hat`, clipping, and numerical behavior before examining
results. Estimating a non-Gaussian empirical residual distribution does not by
itself justify replacing this formula; a distribution-specific correction
requires its own derivation.

Within one inner split, every AACV candidate must use the same reference-model
scale estimate. Candidate-specific noise estimates would change the scoring
rule across candidates and confound model comparison.

## 7. AACV selection and held-out evaluation

After the working law has been estimated from inner training data:

1. fit every AACV candidate on the prescribed inner training data;
2. compute candidate-specific adversarial extrema on inner validation data;
3. score and select candidates using the frozen AACV rule;
4. aggregate inner selections using the predeclared voting and tie-breaking
   rule;
5. use the outer test set only for final held-out evaluation.

Outer-test information must never feed back into the reference learner, error
law, Hermite tuning, finite attack-set rule, or candidate-selection rule.

### 7.1 Fixed finite attack-set rule

The attack set is part of the frozen scientific criterion. In standardized
four-dimensional feature coordinates, use $A(0)=\{0\}$ and, for $r>0$,

$$
A(r)=\{0\}\cup\{r s:s\in\{-1,+1\}^4\}.
$$

Validation and held-out evaluation must use the same origin-plus-16-corners
definition. Extrema are exact only over this finite set and must not be
described as continuous-cube extrema. The set for one radius does not include
corners from other radii, so changing the configured grid does not redefine an
existing radius.

## 8. Predeclared sensitivity analyses

Keep the primary analysis fixed and vary one nuisance component at a time.

### 8.1 Reference-learner sensitivity

Use globally fixed alternatives, such as:

- the primary global clean-risk winner;
- the global runner-up;
- a predeclared ensemble or Super Learner.

Do not choose a different alternative from each AACV inner split.

### 8.2 Scale-estimator sensitivity

Examples include:

- cross-fitted residual variance from the primary reference learner;
- a predeclared multivariate Tong-Wang-style difference estimator;
- a predeclared multivariate Fan-Yao-style residual variance estimator.

The original Tong-Wang and Fan-Yao guarantees do not directly cover every
four-dimensional CCPP implementation. These estimators are therefore
structural sensitivity analyses, not ground-truth variance estimators.

### 8.3 Gaussian-Hermite tuning sensitivity

Vary the predeclared polynomial degree and envelope multiplier while holding
the reference learner, residuals, fitted candidates, and adversarial extrema
fixed.

Do not select the primary reference learner, scale estimator, or Hermite tuning
according to the most favorable held-out AACV result.

## 9. Required reporting

Separate primary and sensitivity results. Report:

- the global reference-selection table and split-level stability;
- the reference learner and frozen fitting protocol;
- cross-fitting folds and seeds;
- residual-law diagnostics for every AACV training split;
- scale-estimator ratios and reference-model sensitivity;
- selected-model agreement and switching across sensitivities;
- held-out score excess and held-out-best match using qualified held-out
  terminology;
- all violated or weakly supported assumptions.

Use language such as **working reference learner**, **cross-fitted Gaussian
scale plug-in**, and **held-out benchmark**. Avoid presenting a learned model,
an estimated scale, or a held-out candidate as unknown physical truth.

## 10. CCPP instantiation

For the current CCPP AACV experiment:

```text
global reference learner:
    HistGradientBoostingRegressor (HistGB)

selection rationale:
    lowest aggregated clean MSE in the preceding repeated SRCV analysis at r=0

primary error law:
    homoskedastic Gaussian working law

primary scale estimate:
    five-fold, fold-local-preprocessing, OOF HistGB residual variance computed
    separately inside every AACV inner training split

primary Gaussian-Hermite tuning:
    J=2, B=6*sigma_hat, no clipping

finite attack set:
    r=0 uses the origin; each r>0 uses the origin plus the 16 current-radius
    cube corners in standardized four-dimensional feature coordinates
```

HistGB remains fixed across AACV inner splits, but a new HistGB model is fitted
within each cross-fitting training fold. The current dataset-level preselection
uses prior analysis of the same CCPP case-study data and is therefore a frozen
case-study design decision, not independent external validation.

Primary outputs must be accompanied by the predeclared reference-model,
scale-estimator, and Hermite-tuning sensitivities described above. The primary
analysis remains valid as a reproducible working-model experiment even when a
diagnostic is unfavorable, but its interpretation must then be explicitly
qualified.

## 11. Reproducibility checklist

Before accepting an AACV case-study run, verify that:

- the global reference learner was fixed before AACV held-out inspection;
- every error-law observation is an honest OOF residual;
- preprocessing is fold-local;
- validation and test responses never enter error-law estimation;
- all candidates in a split share one scale estimate per sensitivity variant;
- primary and sensitivity analyses are clearly separated;
- no sensitivity result is used to replace the primary result post hoc;
- validation and evaluation use the same declared finite attack set;
- attack-set diagnostics verify one point at zero and 17 at positive radii;
- manifests record data hashes, split seeds, package versions, reference-model
  settings, error-law settings, and correction tuning;
- all limitations are stated without claiming an unavailable true regression
  function or true random-error distribution.
