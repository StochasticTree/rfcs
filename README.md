# Requests for Comments

While most contributions to [stochtree](https://github.com/StochasticTree/stochtree) should be submitted directly as pull requests (e.g. bug fixes, documentation enhancements, new functionality), we use a request for comments (RFC) process for proposing "significant" changes. What counts as "significant"? This is naturally more art than science, but a few guiding principles for significance include:

1. Major changes to the R / Python interface, namely
    - Breaking changes to **function / method names** or the **interactions** between the R / Python functions, 
    - **New functions or classes** at the level of BART / BCF (i.e. a new high-level model type), or 
    - Changes to the **output** of existing models (i.e. optimizing the output of `stochtree` for compatibility with other sampling packages like `stan` or `PyMC` or plotting / summary packages like `rpart.plot`)
2. Non-trivial updates to the underlying C++ codebase, including
    - Modifications to data structures aimed at performance improvements, or
    - Changes to the API / class model with the goal of increased flexibility / expressiveness

## Process

stochtree's process is a (lightweight) adaptation of other open source RFC processes 
(see [pytorch](https://github.com/pytorch/rfcs) and [rust](https://github.com/rust-lang/rfcs) for examples).

In order to begin an RFC,

1. Fork this repository
2. Copy the markdown RFC template (`0000-rfc-template.md`) and rename as `0000-your-feature-name.md` (we will convert the `0000` in the title to a unique number based on the pull request number)
3. Fill out the template with your proposal, feel free to deviate from the default format but do try to address the "what," "how," and "why" of your proposal in as much detail as possible
4. Open a pull request

After some discussion and iteration, we will come to decision on whether to accept the proposal. 
If the proposal is accepted, we will renumber `0000-your-feature-name.md` based on the PR number, merge the PR, and open a parallel "tracking" issue in the [stochtree](https://github.com/StochasticTree/stochtree) repo. 
If the changes are significant, we may even assign sub-issues and track the changes in a Github project board.
At this point, the work begins on implementing the feature and the RFC is preserved here as a design "record" for future reference.
