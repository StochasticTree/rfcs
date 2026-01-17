# Supporting cloglog Link in BART Interface

PR link: [TBD]

Tracking issue: [TBD]

# Overview

Adds a new link function for modeling unbalanced ordinal responses, which users can request directly from the `stochtree::bart()` interface in R and the `BARTModel()` interface in Python

# Motivation

`stochtree` currently supports two “link functions,”

1. identity $f(x) = x$  and
2. probit $f(x) = \Phi^{-1}(x)$,

which work in concert with two different outcome models. The identity link is applied to models in which the outcome variable is continuous / real valued (modeled by a Gaussian error). The probit link transforms real-valued forest predictions to $(0,1)$ scale, modeling the conditional probability that a binary outcome equals 1. The probit sampler augments the outcome variable, treating truncated Gaussian latent variables as "outcome data" for a real-valued forest (Albert and Chib (1993)).

For datasets with ordinal outcomes (i.e. $K$ integer-valued outcomes for which $y_1 < y_2 < \dots < y_K$), there are several options:

1. Probit $f(x) = \Phi^{-1}(x)$
2. Logit $f(x) = \log\left(x / (1-x)\right)$
3. Complementary log-log (cloglog) $f(x) = \log\left(-\log(1-x)\right)$

While each of these options have been implemented for BART models, there is already [an experimental release candidate](https://github.com/StochasticTree/stochtree/tree/cloglog-bart-rc) that implements the cloglog model of Alam and Linero (2025) in `stochtree` in the form of a `cloglog_ordinal_bart()`R function. This RFC closes the loop and proposes to fold the cloglog model directly into the BART interface in R and Python.

# Proposed Implementation

## Specifying the Model

### R

Users pass an `outcome_model` type to the `general_params` list. outcome_model is constructed with two arguments:

- outcome: continuous, binary, ordinal
- link: identity, probit, cloglog

with some input checking for possible configurations (i.e. we cannot run probit on continuous outcomes).

We currently support a `probit_outcome_model` binary flag in the `general_params` list and we will pre-empt the new `outcome_model` argument and deprecate this argument gradually with deprecation warnings.

To summarize, syntax for BART models will look like

| **Model type** | **Syntax** |
| --- | --- |
| Continuous outcome, identity link | `bart(X,y,general_params = list(outcome_model = outcome_model(outcome = 'continuous', link = 'identity')))` |
| Binary outcome, probit link | `bart(X,y,general_params = list(outcome_model = outcome_model(outcome = 'binary', link = 'probit')))` |
| Ordinal outcome, cloglog link | `bart(X,y,general_params = list(outcome_model = outcome_model(outcome = 'ordinal', link = 'cloglog')))` |

This separation of outcome type from link opens the door for future implementations of ordinal probit / logit implementation, binary logit, etc…

### Python

Similar to R, we define an `OutcomeModel` class which takes `outcome` and `link` as arguments, with default links for each outcome for users who only wish to specify an outcome type.

## Model Fit

The model fitting code will need additional input validation when the cloglog link is used:

- Outcome are integers starting with either 1 or 0 (re-mapped internally)
- Prior parameters passed or correctly calibrated internally for the cloglog model

Sampling from a cloglog model requires tracking a several more variables through the C++ codebase, so we propose to attach a generic “auxiliary data” container to the `ForestDataset` object (which can be left empty for stochtree’s other models and can be repurposed for other models we support in the future). For cloglog, we need to keep track of

- Augmented / latent variables for all but the last category of an ordinal outcome
- Cached forest predictions
- Log-scale cutpoints
- Exponentiated cumulative cutpoints (transformation of above)

## Returned Model Object

The returned model object will store all / most of what is stored in a “standard” continuous outcome BART model (forests, sampler metadata, etc…), along with a matrix of draws of log-scale cutpoint parameters ($\gamma_1, \gamma_2, \dots, \gamma_K$ in the notation of Alam and Linero (2025))

What we will **not** track and return:

- Augmented / latent variables: users wishing to sample another run of a cloglog bart model from a given trace will have to “burn-in” these latent variables again

## Prediction

Right now, the `predict` function / method in both R and Python contains a `scale` argument whose valid options are `'linear'` and `'probability'`. We will modify this to more closely match the spirit / style of existing GLM implementations. For quick overview:

- [the `glm` function in base R](https://stat.ethz.ch/R-manual/R-devel/RHOME/library/stats/html/predict.glm.html) uses `type = 'response'` for inverse-link scale and `type = 'link'` for linear scale
- [the `GLM` class in statsmodels (Python)](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.predict.html#statsmodels.genmod.generalized_linear_model.GLM.predict.which) uses `which = 'mean'` for inverse-link scale and `which = 'linear'` for raw linear scale

We will accept either `'response'` or `'mean'` for inverse-link scale and `'link'` or `'linear'` for linear scale. We will also continue to accept `'probability'` for probit models, with a deprecation warning that it will be phased out in future versions.

## Serialization / De-Serialization

As the returned model object does not differ substantially from the model object currently returned by `stochtree::bart()` and `BARTModel.sample()`, we will simply modify the JSON serialization procedure to include:

1. The `outcome_model` data
2. The cutpoint parameter trace ($\gamma_1, \gamma_2, \dots, \gamma_K$)

# Value

This model is of direct interest to current stakeholders (as evidenced by PR [#196](https://github.com/StochasticTree/stochtree/pull/196)) and it also unlocks a potential user base with unbalanced ordinal outcome data.

# Risks and Drawbacks

There are no major implementation risks as the release candidate works well, but making `stochtree::bart()` and `BARTModel()` function more like a GLM does add more maintenance overhead. We believe that this is largely addressable with robust input validation and unit tests.

# Alternatives

The most “natural” alternative is what is currently done in the [release candidate](https://github.com/StochasticTree/stochtree/tree/cloglog-bart-rc) branch: maintaining a separate `cloglog_ordinal_bart()` function (and a similar Python interface) which leaves the existing BART interfaces untouched. While this makes for a slightly cleaner implementation and eases maintenance, it is less ideal from a user interface perspective. With separate functions, we require users to think about “which BART function to call” and, furthermore, we require users to correctly pinpoint the cloglog model as their desired link function if they have ordinal data. The proposed design will default to a cloglog link if users specify an ordinal outcome.

# Open Questions

What “post-processing” options should we offer for cloglog / ordinal models? Some options:

1. Helper function for estimating each observation’s probability of being in category $k$
2. Helper function for estimating each observation’s probability of being in its observed outcome category

# References

Albert, James H., and Siddhartha Chib. “Bayesian Analysis of Binary and Polychotomous Response Data.” Journal of the American Statistical Association 88, no. 422 (1993): 669–79. [https://doi.org/10.2307/2290350](https://doi.org/10.2307/2290350).

Alam, Entejar, and Antonio R. Linero. "A Unified Bayesian Nonparametric Framework for Ordinal, Survival, and Density Regression Using the Complementary Log-Log Link." *arXiv preprint arXiv:2502.00606* (2025).