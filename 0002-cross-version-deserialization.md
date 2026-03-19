# Cross-Version Deserialization for BART and BCF Models

PR link: [TBD]

Tracking issue: [TBD]

# Overview

Add version metadata to stochtree's JSON serialization format and implement
robust deserialization that gracefully handles JSONs produced by older versions
of stochtree — filling in safe defaults for fields that did not yet exist when
the model was serialized. This covers intra-platform compatibility (R→R and
Python→Python) for BART and BCF models. Cross-platform deserialization (R→Python
and Python→R) is a related but separate effort tracked as future work.

# Motivation

stochtree serializes fitted BART and BCF models to JSON so users can save and
reload them. As the library evolves, new fields are added to the JSON schema
(e.g., `multivariate_treatment`, `outcome`/`link` in `outcome_model`,
`rfx_model_spec`, `internal_propensity_model`). Because older JSON files do not
contain these fields, attempting to deserialize a model that was fit under a
prior version of stochtree fails — either with a hard error or with silently
incorrect behavior.

A concrete example was reported by a user who serialized a BCF model under
~v0.3.x and then attempted to deserialize after upgrading to v0.4.1. Their JSON
was missing:

- `link` and `outcome` under `outcome_model`
- `rfx_model_spec`
- `multivariate_treatment`
- the `outcome_model` fields inside the nested `bart_propensity_model`

The current code does not attempt to recover from any of these gaps, so
deserialization fails unconditionally for that user.

# Proposed Implementation

## 1. Add `stochtree_version` to JSON output

Every call to `to_json()` in Python and the equivalent serialization functions in R
should write a top-level string field `stochtree_version` containing the current 
package version (e.g., `"0.4.1"`). This requires reading the version at serialization time:

- **Python**: `importlib.metadata.version("stochtree")` or a constant in
  `stochtree/__init__.py`
- **R**: `utils::packageVersion("stochtree")` coerced to a string

## 2. Version inference for legacy JSONs (no stamp present)

When `stochtree_version` is absent from a JSON, infer the version bracket from
the fields that *are* present. The table below maps observable field presence to
the approximate version that introduced each field:

| Field absent | Inferred serialized before |
|---|---|
| `covariate_preprocessor` / `preprocessor_metadata` | ~v0.3.0 |
| `rfx_model_spec` | ~v0.3.0 |
| `internal_propensity_model` | ~v0.3.2 (Jan 2025) |
| `multivariate_treatment` (Python BCF) | ~v0.4.0 (Oct 2025) |
| `has_rfx_basis` / `num_rfx_basis` | ~v0.4.0 (Oct 2025) |
| `outcome` / `link` in `outcome_model` | ~v0.4.1 (Feb 2026) |
| `sample_tau_0` / `tau_0_samples` | v0.4.1-dev |

The inferred version is surfaced as a string (e.g., `"<0.3.0"`, `"<0.4.0"`)
for logging/debugging purposes, but is not used to gate behavior — defaults
handle that (see §3).

## 3. Defaults for missing fields

`from_json()` and the R deserialization functions should check for each optional
field with a `has_*` guard (or equivalent) before reading it, and fall back to a
sensible default. The full default table:

### BART

| Missing field | Safe default | Notes |
|---|---|---|
| `covariate_preprocessor` / `preprocessor_metadata` | `NULL` / `None` | Disables DataFrame prediction; warn user |
| `rfx_model_spec` | `""` / `None` | Only matters if `has_rfx = TRUE`; warn if so |
| `has_rfx_basis` | `FALSE` / `False` | |
| `num_rfx_basis` | `1` | Only used if `has_rfx_basis = TRUE` |
| `outcome` (in `outcome_model`) | `"continuous"` | |
| `link` (in `outcome_model`) | `"identity"` | |
| `probit_outcome_model` | `FALSE` / `False` | Already conditional in current code |
| `cloglog_num_categories` | `0` | Only used if cloglog link |
| `num_chains` | `1` | |
| `keep_every` | `1` | |

### BCF (additional fields beyond BART)

| Missing field | Safe default | Notes |
|---|---|---|
| `multivariate_treatment` | `FALSE` / `False` | |
| `internal_propensity_model` | `FALSE` / `False` | If absent, `bart_propensity_model` also absent |
| `has_rfx_basis` | `FALSE` / `False` | |
| `num_rfx_basis` | `1` | |
| `outcome` / `link` | `"continuous"` / `"identity"` | |
| `sample_tau_0` | `FALSE` / `False` | |
| `tau_0_dim` | `1` | Only used if `sample_tau_0 = TRUE` |
| `rfx_model_spec` | `""` / `None` | Warn if `has_rfx = TRUE` |

## 4. Emit warnings, not errors, for recoverable gaps

When a missing field is filled with a default, emit a `warning()` (R) or
`warnings.warn()` (Python) identifying the field and the assumed default. This
makes debugging straightforward without breaking user code. The warning should
reference the inferred version bracket where possible, e.g.:

> "JSON appears to have been serialized before stochtree v0.4.1 (field
> `outcome_model.link` not found). Assuming link = 'identity'. Re-serialize
> your model to suppress this warning."

Hard errors should be reserved for fields that are genuinely unrecoverable (e.g.,
the forest structures themselves, `outcome_scale`, `outcome_mean`).

## 5. Fix existing field-name mismatches blocking cross-platform work

Two field-name mismatches exist between R and Python in the BCF schema that
should be corrected now, as they affect intra-platform consistency and will be
prerequisite for any future cross-platform work:

| Field | Python (current) | R (current) | Standardize to |
|---|---|---|---|
| BCF initial variance | `sigma2_init` | `initial_sigma2` | `sigma2_init` |
| BCF adaptive coding b0 | `b0_samples` | `b_0_samples` | `b0_samples` |
| BCF adaptive coding b1 | `b1_samples` | `b_1_samples` | `b1_samples` |

For each, the R `to_json` / `from_json` should be updated to use the Python
spelling. Because the old R spelling may appear in JSONs serialized before this
fix, the R `from_json` should also accept the legacy name (with a deprecation
warning) — this is the same pattern as §3 above.

## 6. Tests

- **Snapshot tests**: Serialize a BART and BCF model under the current version,
  save the JSON as a test fixture, then load it in a future version to confirm
  no regressions.
- **Backward-compat tests**: Construct minimal "old-format" JSON strings (with
  specific fields omitted) and assert that deserialization succeeds with correct
  defaults and the expected warnings.
- These tests should exist in both R (`test/R/testthat/`) and Python
  (`test/python/`).

# Value

Any user who fit a BART or BCF model on a prior version of stochtree and saved
it to disk will be able to reload it after upgrading, rather than hitting an
opaque deserialization error. This is a common pain point when a project spans a
stochtree version bump (e.g., a long-running study where models were saved months
ago).

The version stamp also provides a foundation for future compatibility guarantees
and makes debugging serialization issues substantially easier.

# Risks and Drawbacks

- **Default assumptions could silently mask real errors.** If a user's JSON is
  corrupted rather than merely old, applying defaults might produce a model that
  runs but gives wrong predictions. Mitigated by the warnings in §4 and by only
  applying defaults for fields with unambiguous safe values.
- **Maintenance surface.** Every time a new field is added to the JSON schema in
  the future, a corresponding default must also be added to the deserialization
  fallback table. This is low effort per change but requires discipline. A
  code-review checklist item ("did you add a deserialization default?") would
  help.
- **R version of `sigma2_init` rename (§5)** is a breaking change for anyone
  reading BCF JSON fields directly (not via `from_json`). The legacy alias
  mitigates this.

# Alternatives

- **Do nothing / document workaround.** Users can use the low-level JSON API to
  manually extract forests and parameters. This works but is a poor user
  experience and was the approach recommended in [#315](https://github.com/StochasticTree/stochtree/issues/315) as a stopgap only.
- **Strict versioned schemas with no backward compat.** Require exact version
  matches and tell users to refit. Rejected: the point of serialization is to
  avoid refitting.
- **Single large compatibility shim.** One big `upgrade_json()` function that
  applies all transformations in sequence (v0.2→v0.3→v0.4→current). More
  systematic but harder to maintain and overkill given how stable the schema has
  been.

# Open Questions

1. **Should `stochtree_version` be a semver string or an integer schema version?**
   Semver is user-friendly; an integer schema version is more precise (two patch
   releases might bump the schema, or a major release might not). Semver is
   probably fine given the current pace of schema changes.

2. **Python package version at runtime.** During development installs (`pip
   install -e .`), `importlib.metadata.version("stochtree")` may return a dev
   version string or fail. We should handle this gracefully (fall back to
   `"dev"`).

3. **Where to put the version inference logic?** A small helper
   (`_infer_json_version()` / `.infer_json_version()`) shared between BART and
   BCF deserialization seems right. The R equivalent could live in `utils.R`.

# Future Work: Cross-Platform Serialization

Making R-serialized models loadable in Python (and vice versa) is a natural
follow-on to this work but is explicitly out of scope here. The main blocker is
that the **covariate preprocessor** uses fundamentally different internal schemas
in R and Python: R keys categories by variable name while Python uses indexed
arrays with dtype metadata. Bridging these requires either a new shared canonical
preprocessor schema or a translation layer, which is a non-trivial design
decision warranting its own RFC.

The field-name fixes in §5 of this RFC are prerequisites for cross-platform
work and should make the remaining gap smaller and more tractable.

# References

- GitHub issue: https://github.com/StochasticTree/stochtree/issues/315
