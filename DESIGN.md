# ChainLab design

## Three distinct questions

Forward inference sums over paths to score observations. Forward/backward gives
smoothed state and adjacent-state posterior marginals. Viterbi maximizes one
joint path, equivalent to the posterior path for fixed observations. They are
not interchangeable. The model emits the first symbol after sampling its initial
state, then transitions before each subsequent emission.

A path probability must respect initial, transition and emission factors.
The exact three-state one-symbol counterexample has initial .4/.35/.25 and
transition rows [0,.5,.5], [1,0,0], [1,0,0]. On x/x, marginal modes A/A use a
zero-probability transition. The valid best path is B/A with posterior .35.
The posterior rows .4/.35/.25 and .6/.2/.2 cannot be multiplied as independent
path factors. Tests enumerate all paths to check both objectives.

## Numerical representation

Zero probabilities map to negative infinity internally. Log-sum-exp sums forward
and backward alternatives; Viterbi uses maxima and backpointers. No epsilon
smoothing or fallback path. Truly impossible observations raise ValueError.
The one-state rare-symbol example keeps log likelihood -3536.77070284 finite
for256 observations with emission1e-6, while its probability underflows to0.

State and transition posterior outputs exponentiate log factors then normalize
for numerical roundoff. Record the largest raw mass deviation from1. Extremely
small posterior mass may still underflow; no arbitrary precision guarantee.
Input rows retain their values if their sum is within1e-12, otherwise reject.
No silent input normalization or matrix repair. Exact local ties take first
state index; whole-path lexicographic order is not guaranteed by that rule.

## Multi-sequence maximum-likelihood EM

Every sequence starts independently. Sum gamma at each start, xi on within-
sequence edges, and gamma for matching observed symbols. Normalize expected
counts to update initial, transition and emission probabilities. No cross-
sequence edges, pseudocounts or prior objective. A single-state check reduces
to empirical symbol frequencies; a tiny exhaustive-path expected-count reference
checks one whole multi-sequence EM update.

If a transition/emission row has zero expected occupancy, keep its previous
row and report its index. All-length-one data cannot fit transitions. Existing
zero support cannot recover through unsmoothed EM; absent symbols can lose
support. These are explicit limitations, not invisible smoothing choices.

Score every proposal. Accept only nonnegative likelihood changes. A tiny
negative change in[-1e-7,0) stops with rejected_proposal, keeping the previous
model/history. A change below-1e-7 raises an error. Simulated numerical branches
are tested. An accepted improvement<=tolerance stops; tolerance is absolute
nats for the sum over all sequences, not per symbol. Budget<=100 updates. No
stationarity, local/global optimum or identifiability certificate.

## Development fixture and controls

Seed23, two persistent regimes, three categorical emissions,20 training and12
test sequences of40 symbols. Supplied asymmetric initialization,30-update cap,
tolerance1e-6. No parameter/initialization search. Test likelihood does not enter
fit or stopping. The synthetic source, initialization and random seed are all
in the public demo; the fixture was inspected during development.

Training log likelihood improves -820.4336249187 to -651.3909460329 over20
accepted updates, then stops on tolerance. Test NLL per symbol (nats):
initial1.0176377671, iid training-frequency control.9859930334,
trained.7756309219, generating-model reference.7716741802. The generating model
is not learned or guaranteed to beat every finite-sample fit. The iid control
fits symbol frequencies but no transitions. Same observed train/test data, not
equal parameter count or a causal decomposition of transition benefits.

No state-recovery accuracy is reported: latent labels are arbitrary and the
likelihood objective is not ground-truth state identification. Full inference
on test sequences uses future symbols, not an online prediction benchmark.
One selected persistent-regime development fixture does not establish general
rankings, anomaly-detection performance or calibration.

## Prefix filtering and unobserved-future forecasts

```python
from chainlab import filter_sequence, forecast

online = filter_sequence(model, ["x", "y", "x"])
future = forecast(model, ["x", "y"], horizon=3)
```

The `model` is the same validated `chainlab_v1` artifact used by infer/fit.
`filter_sequence` returns a state posterior for each prefix, prior-predictive
symbol probabilities **before** its symbol is observed, each observed symbol's
log predictive score, and prefix log likelihoods. At time0 the prediction uses
the initial state distribution; later steps transition the last filtered state.
It never uses later observations to update an earlier row. This is a batch
prefix audit, not a persistent streaming session or evidence that the supplied
model was trained without future-data leakage.

The difference from smoothing is exact on a two-state example: absorbing states
A/B, initial0.5/0.5, emissions A:[1,0], B:[0.5,0.5] for symbols x/y.

| State inference | Available observations | P(state A at time0) |
|---|---|---:|
| Filtering after first symbol | x | 2/3 |
| Full-sequence smoothing after both | x,y | 0 |

Using the smoothed row for an online decision at time0 would use future y.
The final filtered and final smoothed rows agree because they have the same
evidence at that time. This is a conditioning difference, not a performance
contest between algorithms.

Log posterior messages stay in log space between steps, so a path with a
raw-probability product below binary64 range can still become the only supported
state after a later observation. Printed probability entries can underflow to
zero even with a finite `observed_log_predictive`; the log scores are the
likelihood record, not `log` of the rounded probability table.

`forecast` accepts horizon1-32 and conditions on the whole supplied **past**
sequence. Horizon1 transitions once from the last conditioned state, then
predicts its emission. Each later step propagates states again without
assimilating any guessed observation. Returned state/symbol distributions are
**marginals**, not a joint future path and not independent across horizons.
In the absorbing-state example after x, each future marginal has P(y)=1/6,
but P(y,y)=1/12, not (1/6)^2. The implementation does not multiply forecast
marginals or feed their modes back as observed evidence.

```sh
python -m chainlab filter --input examples/request.json
python -m chainlab forecast --input examples/request.json --horizon 3
python -m chainlab online-demo
```

On the same trained seed23 development fixture, summed prefix predictive log
scores across12 independent test sequences (480 symbols) are -372.302842489947,
versus -372.30284248994707 from full forward likelihood: absolute difference
5.68e-14 on this run. Mean negative log predictive score is0.775630921854.
Every sequence resets to the model's initial distribution. This equality is an
implementation audit, not a new quality improvement or calibration result.
The demo forecasts five steps after only the first10 symbols of one test
sequence; no future symbols enter those distributions.

Filtering costs O(T*(N^2+N*M)) for alphabet size M and stores the prefix outputs.
Forecasts add O(H*(N^2+N*M)); no xi matrices or backward pass are needed. Results
are numerically normalized, with filtering mass-error diagnostics. Existing
model bounds, impossible-prefix rejection and no input mutation still apply.
No stateful streaming API, adaptive model update, calibrated forecast promise,
or proof that a caller's training protocol is leak-free.

## Artifact and cost boundary

Strict chainlab_v1: ordered states/symbols, initial vector, transition matrix,
emission matrix. No serialized code, training data or dynamic factors. Every
operation validates. JSON round trips and label permutations have tests. No
mutation of caller inputs.

N1-8, alphabet1-16, T1-256;1-32 training sequences,total<=2048. Full gamma/xi
inference costs O(T*N^2) time/storage at the xi output scale. Viterbi also
computes inference for its posterior score, so its implementation is not the
minimal-memory path-only variant. Training repeats E-steps and proposal scoring.
Prefix filtering and marginal forecasts are separate methods; no dense
accelerator, neural component, persistent streaming-state or calibrated forecast claim.

## Sources

- https://www.cs.cmu.edu/~cga/behavior/rabiner1.pdf
- https://www.stats.ox.ac.uk/~caron/teaching/sb1b/slideshmm.pdf
- https://sites.stat.washington.edu/mmp/courses/stat534/spring19/Handouts/h4-hmm.pdf

Standard HMM/EM theory. Independent implementation, no new algorithm claim.
