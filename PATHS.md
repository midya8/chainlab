# Paths, losses and uncertainty

ChainLab v0.3 adds a separate rational HMM workbench. It does not change the original binary64 trainer, its fitted models, its data, its scores, its Viterbi tie rule or the prefix-evidence inspector.

The point is a decision distinction, not a quality claim: a MAP path minimizes whole-path 0/1 loss; it need not minimize expected wrong positions. State-by-state posterior modes minimize unconstrained Hamming loss, but may be impossible under the model. A support-constrained sum of state marginals differs from a support-constrained product of state marginals. None is the full path distribution.

## Run it

```sh
python -m unittest discover -s tests -v
node --test tests/*.cjs
python examples/path_controls.py
python -m chainlab.pathrisk input.json > report.json
python -m chainlab.path_verify report.json
```

Both commands accept `-` for stdin. The browser download is a complete report with an `input` field. Save that field as `input.json` for a fresh producer replay, or send the whole archive directly to the independent verifier. Exit 2 means invalid input, a limit refusal, or a report mismatch. No partial JSON goes to stdout on failure.

## Input and timing

The request has exactly `model` and `sequence`. Model fields are exactly `version`, `states`, `symbols`, `initial`, `transition`, and `emission`. Version is `chainlab_rational_v1`. Names are distinct nonempty strings of at most 40 Unicode code points. Probabilities are unsigned integer/fraction strings, not JSON numbers or decimals. Every row sums to exactly one. No silent probability normalization.

```json
{
  "model": {
    "version": "chainlab_rational_v1",
    "states": ["A", "B"],
    "symbols": ["x"],
    "initial": ["1/2", "1/2"],
    "transition": [["1", "0"], ["0", "1"]],
    "emission": [["1"], ["1"]]
  },
  "sequence": ["x", "x", "x", "x", "x"]
}
```

There is no initial transition: the initial state emits the first symbol. For each later position, transition once, then emit. All posteriors and decisions use the entire supplied observation window, including observations later than a displayed state. These are retrospective calculations, not online forecasts.

## Exact messages and support

For observations `y[0:T]`, let `f[t,i]` be the joint probability of the observed prefix and state `i`. Let `b[t,i]` be the probability of the observations strictly after `t`, conditional on that state. Then

```text
f[0,i] = initial[i] emission[i,y0]
f[t,j] = emission[j,yt] sum_i f[t-1,i] transition[i,j]
b[T-1,i] = 1
b[t,i] = sum_j transition[i,j] emission[j,y(t+1)] b[t+1,j]
Z = sum_i f[T-1,i]
gamma[t,i] = f[t,i] b[t,i] / Z
xi[t,i,j] = f[t,i] transition[i,j] emission[j,y(t+1)] b[t+1,j] / Z
```

A zero `Z` rejects: no posterior exists. These calculations use `Fraction` in Python and independent `BigInt` fractions in JavaScript. Counts use decimal strings, including values beyond JavaScript's safe integer range. A positive fraction never becomes impossible because a floating representation underflows.

A path is supported if its initial probability, every transition and every required emission are positive. Prior transition support alone is insufficient. `conditional[t,i,j]` is `xi/gamma` when `gamma[t,i]>0`. An entire zero-mass conditioning row is `null`, meaning undefined, not a zero conditional distribution.

## Four different decisions

For any proposed path `a`, expected matches are `sum_t gamma[t,a[t]]`; expected Hamming loss is `T` minus that sum. The report lists its actual joint probability, posterior path probability, expected matches, expected loss and product of state marginals separately.

| Rule | Exact objective | Feasible choices |
|---|---|---|
| `map` | Maximize the path's joint probability | Supported full paths |
| `supported_matches` | Maximize the sum of selected state marginals | Supported full paths |
| `supported_product` | Maximize the product of selected state marginals | Supported full paths |
| `marginal_modes` | Maximize each state marginal separately | All state strings, even unsupported ones |

The supported sum decoder is admissibly constrained PMAP / minimum expected Hamming decoding. The supported product decoder is posterior-Viterbi style; its product is not a joint posterior. The three trellises retain every best supported prefix ending at each state. A prefix score is not a full-path posterior.

Exact objective ties select the lexicographically first whole index path. Marginal-mode ties choose the first index at each position. This rule belongs to the new rational workbench only. The older floating Viterbi API keeps its documented local predecessor/final tie rule.

The default adverse model gives three different supported decisions:

| Rule | Path | Posterior probability, approximate | Expected wrong positions, approximate |
|---|---|---:|---:|
| MAP | B/C/B/C/B | .05823316 | 2.880904 |
| Supported expected matches | B/A/C/A/C | .05686832 | 2.531237 |
| Supported product marginals | B/C/B/A/C | .03639572 | 2.535156 |
| Unconstrained marginal modes | B/C/C/A/C, unsupported | 0 | 2.456165 |

The example is the first qualifying model from a deterministic random search, seed 314159, zero-based candidate 1904. It was selected to exhibit disagreement. It is not a typical-case sample, a benchmark, a trained-model improvement or a real-state accuracy result. `examples/path_controls.py` includes the search, not just its winning model.

## Whole-path entropy

The posterior state sequence is still a time-inhomogeneous Markov chain. For a supported prefix, its next-state conditional is `xi[t,i,j]/gamma[t,i]`; factors depending on earlier states cancel when the current state and all observations are fixed. Thus the chain rule gives

```text
H(path | y) = H(gamma[0])
              + sum_t sum_i gamma[t,i] H(conditional[t,i,:])
```

All terms with zero probability contribute zero. The report also retains each state entropy and each posterior adjacent-state mutual information:

```text
I_t = sum_ij xi[t,i,j] log(xi[t,i,j] / (gamma[t,i] gamma[t+1,j]))
sum_t H(gamma[t]) - H(path | y) = sum_t I_t
```

The identity follows by expanding each conditional entropy as `H(S[t+1]|y)-I_t`. The equality is a property of the posterior Markov chain, not an assumption that positions are independent. Summing marginal entropies generally overstates the joint entropy.

The locked and independent five-position controls have identical `[1/2,1/2]` marginals. The locked chain has two supported paths and entropy `ln(2)`. The independent chain has 32 supported paths and entropy `5 ln(2)`. Both report the same `5/2` expected Hamming loss for their selected paths. A single selected path and its per-state loss do not capture dependence.

Entropy and logarithms are approximate binary64 nats computed from exact rational probabilities. Very tiny positive contributions may round to zero. No exact transcendental value or rigorous entropy enclosure is claimed. The independent verifier uses Decimal precision 80 and absolute tolerance `1e-10` nats for floating fields. Rational fields and decisions must match exactly. The rare-evidence control has evidence `2^-3840`, which underflows in binary64, but retains one supported path, posterior one, finite log evidence and zero entropy.

## Controls and verification

Thirteen complete controls cover one state, independent and locked chains, forbidden marginal modes, emission support, zero occupancy, exact path ties, `4^32` supported paths, rare evidence, the three-objective disagreement and three rounded original-model examples.

The last three are explicitly different rational models: each original JSON decimal probability is rounded to the nearest millionth using Python's half-even rule, then its row is exactly normalized. These examples are not described as unchanged original fitted models or new trained arms. Their first 12 observations come from the existing displayed fixture window. Original saved artifacts remain byte-for-byte unchanged.

The new Python suite uses an independent Cartesian product oracle for small models, with exact path weights, all marginals, all pair marginals, all three supported decisions and Decimal80 entropy. It includes 120 fixed-seed random models, the complete controls, pair consistency, the entropy/dependence identity, schema bounds, explicit work refusal, a support count beyond `2^53`, and rejection of all 438 primitive leaf modifications to the default report. Floating changes below the stated tolerance are intentionally outside the verifier's rejection promise.

The standalone verifier imports neither the producer nor its parser. It parses the canonical source model independently, rebuilds suffix probabilities with memoization, forms independent prefix sums, checks all decoding candidates and whole-path ties, and checks floating entropy with Decimal80. It does not trust a supplied certificate's scores or input/output binding. It is another implementation of the same standard recurrences, not a formal proof assistant. Tiny exhaustive enumeration provides an additional algorithmically different oracle.

The browser engine consumes only input models, never native saved reports. Node compares all complete control reports and 150 generated native/BigInt cases, rejecting impossible models. Actual Chrome checks all 13 controls, 296 state cells, 1,532 joint/conditional edge cells, 867 reachable decision-trellis cells, 39 responsive layouts, physical keyboard model selection and JSON editing, invalidation, clipboard behavior and two completed Blob downloads independently verified natively. The report includes all forward/backward messages even though the UI emphasizes marginals and decisions.

The full suite currently has 58 Python tests and 10 Node/Chrome tests. All earlier 41 Python and four Node/Chrome tests remain. Installed-wheel replay imports all 10 modules, reruns original training/data/prefix reproduction and all nine earlier CLI commands, then replays all 13 new controls, both new module commands and 26 malformed report checks outside the checkout.

## Bounds, costs and non-goals

- 1-4 states, 1-8 symbols, 1-32 observations. Input rational numerator/denominator at most 128 bits and each token at most 80 characters.
- At most 8,192 bits per reported rational. Complex denominators can exceed that cap before the maximum sequence length. Such input is refused, not approximated. Zero probabilities remain exact zeros.
- CLI input at most 64 KiB and report at most 4 MiB for verification. Duplicate keys, extra fields, booleans or JSON numeric probabilities, malformed fractions, impossible evidence and unsupported dimensions reject. Browser JSON parsing also caps nesting at 64.
- Message passing and the number of Bellman alternatives scale as `O(T N²)` arithmetic operations. Stored path prefixes and lexicographic comparisons add length-dependent copying/comparison costs. Exact rational operations have size-dependent costs; this is not a constant-cost arithmetic or speed claim. Full reports retain messages, marginals, transitions, conditionals and all decision prefixes, not a streaming low-memory representation.
- The supplied state labels and model are assumptions. No model fitting, parameter uncertainty, label identification, causal interpretation, true-state recovery test, calibration, coverage, external dataset result, production readiness or algorithm novelty is claimed. Low entropy can simply mean a misspecified model is confidently wrong.

## Attribution

- Lember and Koloydenko, [Bridging Viterbi and Posterior Decoding: A Generalized Risk Approach to Hidden Path Inference Based on Hidden Markov Models](https://jmlr.org/papers/volume15/lember14a/lember14a.pdf), JMLR 2014: risk objectives, admissibility and sum versus product marginal decoding.
- MIT 6.047/6.878, [Lecture 08: HMMs II, Posterior Decoding and Learning](https://web.mit.edu/6.047/book-2012/Lecture08_HMMSII/Lecture08_HMMSII_standalone.pdf): standard forward/backward and posterior decoding.
- Ilic, [Entropy Semiring Forward-backward Algorithm for HMM Entropy Computation](http://arxiv.org/pdf/1108.0347), 2011: conditional hidden-state sequence entropy, entropy decomposition and message-passing context. This workbench uses the explicit smoothed-chain decomposition above, not the paper's entropy-semiring or subsequence-constrained algorithm.
- HMC CS158, [Hidden Markov Models](https://www.cs.hmc.edu/~yjw/teaching/cs158/lectures/17_19_HMMs.pdf): standard HMM message recurrences.

Original implementation, controls and UI under MIT. No third-party source code, dataset endorsement or new sequence-model theorem.
