# Prefix evidence, not hindsight

ChainLab v0.2 adds a live observation-window inspector to the original seed23 HMM project. It does not change original core/online/demo code, training protocol, data, scores, path counterexample or future-information counterexample. The first three models display only the first 12 observations from the original 40-symbol first test sequence. Every complete original report is retained separately. Smoothing/Viterbi use the displayed window, not the omitted 28 observations.

## Four conditioning questions

For fixed initial probabilities pi, transition matrix A, emission matrix B and displayed observations o[0:T]:

- Filtering f[t] = P(z[t] | o[0:t]). Each row uses its own prefix.
- Smoothing g[t] = P(z[t] | o[0:T]). Future displayed observations may change earlier rows.
- Viterbi maximizes P(z[0:T], o[0:T]), equivalently the posterior path for this fixed window. Per-position marginal modes answer a different question and may form a forbidden path. Never multiply state marginals into a joint path probability.
- Forecast horizon h uses f[cut-1] A^h and that state's emission mixture. Horizon1 transitions once. Every horizon is conditioned on the same observed prefix; no guessed future symbol is fed back. Marginals are neither independent future variables nor a joint path distribution.

The original absorbing-state example gives P(A | x)=2/3 but P(A at first time | x,y)=0. Smoothing that first row into an online decision leaks future evidence. After x, P(next y)=P(second future y)=1/6, while the joint probability of two future y symbols is 1/12, not 1/36. The original three-state example's marginal modes A/A use a forbidden A-to-A edge; Viterbi B/A is supported. These are exact constructed examples, not estimated deployment properties.

## Independent numerical paths

Native inference retains the original log-space forward/backward and prefix messages. The new `evidence` API validates the model/sequence and combines original filtering, smoothing, Viterbi and a prefix-only forecast. It never trains or silently repairs a model.

The independently written browser uses scaled probability recurrences. Starting with the supplied initial prior, it forms each raw state likelihood and divides by its positive total c[t]. Log likelihood is sum(log(c[t])). Backward messages divide by the matching c[t+1]; normalized forward/backward products give smoothed state and adjacent-state posteriors. Viterbi independently uses log scores and strict-greater comparisons, retaining the first predecessor/final index on exact local ties. This is not a promised lexicographic whole-path tie rule.

Scaling is not equivalent to arbitrary-range log inference. A positive path can underflow before normalization, especially for extreme initial/emission products. Browser zero or nonfinite local mass refuses as zero or underflowed mass; it does not claim the sequence is mathematically impossible. A test retains the original extreme tiny-support example: native log inference succeeds while the scaled browser refuses. Ordinary probability outputs can underflow tiny entries even when authoritative native log scores are finite. No adaptive smoothing or invented posterior is returned.

Input model probabilities must already sum to 1 within the original 1e-12 tolerance. Only posterior/predictive outputs are normalized numerically. Native schema permits up to 256 symbols and 32 forecast horizons. Live editing is bounded to 1-16 symbols and 1-8 horizons to keep the views readable. Invalid input clears the current witness, all derived rows/paths/scores and stale export. A complete witness contains the fixed public synthetic model, window, cut and horizon; it is not a source-authenticity signature.

## What the checks establish

- All 114 position/symbol edit controls, including original-symbol/no-op controls, cover three frozen original models and both exact examples. Native archives retain full filter/smoothed/adjacent-state/path/forecast/log values. Three 12-symbol model windows contribute 108 controls, the future-information example 4 and path example 2. Impossible windows explicitly refuse.
- All 40 original-window prefix boundaries retain full native forecasts and witnesses. Suffix changes cannot alter already observed filtered rows or a fixed prefix forecast.
- Independent 70-digit Decimal enumeration checks all 62 nonempty binary observation windows of lengths 1-5 in a separate two-state model. It exhaustively sums represented-binary64 path probabilities for state/adjacent-state marginals, sequence likelihood and best-path probability. This does not establish exact-real equivalence for every model. A separate scalar browser enumeration cross-checks all 62 windows.
- Browser scaled inference matches all numerical native archive fields under a 2e-10*(1+abs(reference)) test tolerance. Actual Chrome exercises all 114 edit controls and 40 prefix boundaries, checks displayed likelihood rounding, state posteriors, paths and future-symbol marginals. It also checks both exact counterexamples, symbol buttons, physical range keyboard control, invalidation, denied/successful clipboard, actual JSON Blob and expanded 320/390/1280 layouts.
- Archive regeneration requires exact keys, shapes and nonfloat values; floats use 1e-11 absolute / 1e-12 relative tolerance for cross-platform math rounding. Reproduction tests reject 1e-8 perturbations and nonfinite values. Original model tolerance and original tests are unchanged.
- Installed reproduction imports all eight modules, runs all nine commands, rebuilds every archive and matches original training/online reports outside checkout. Original SVG reproduction and all original tests remain.

Correctness checks are not calibration, runtime, state accuracy, robustness, causal or external-performance benchmarks. The original trained model, initialization, training-only iid control and generating reference remain as reported. Learned state labels are not aligned with generating labels; no state-recovery claim is made. Edited windows never replace or recompute original test scores.

## Reproduce

```sh
python -m chainlab.evidence_export
python -m chainlab evidence-report
python -m chainlab evidence --input examples/request.json --cut 2 --horizon 5
python -m unittest discover -s tests -v
node --test tests/*.cjs
bash examples/demo.sh
python -m pip wheel . --no-deps -w /tmp/wheels
python -m pip install --target /tmp/installed /tmp/wheels/chainlab-0.2.0-py3-none-any.whl
bash examples/verify_wheel.sh /tmp/installed
```

CLI duplicate JSON keys now reject globally rather than silently overwriting. All seven original commands remain; `evidence` and `evidence-report` are added.

## Sources and differences

- [Rabiner, 1989 HMM tutorial](https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf): established forward/backward and Viterbi distinctions. Educational independent implementation, no new method and no library code copied.
- [Marina Meila, STAT 534 HMM summary, 2013](https://sites.stat.washington.edu/mmp/courses/stat534/spring19/Handouts/h4-hmm.pdf): state/adjacent-state posterior and Viterbi recurrences. Our explicit local tie rule and browser refusal behavior are implementation choices.
- [Zucchini and MacDonald, Forecasting, decoding and state prediction, 2009](https://civil.colorado.edu/~balajir/hmm/HMM-book-Chapter5.pdf): sections 5.2-5.4 distinguish forecast marginals, local/global decoding and future-state distributions. No stationary-limit or joint-independence claim is imported into the UI.
- [hmmlearn tutorial](https://hmmlearn.readthedocs.io/en/latest/tutorial.html): supplied/fitted parameters, decoder choice and state-label order warning. This page does not call hmmlearn, choose an initialization sweep or rerun training after edits.
- [hmmlearn API](https://hmmlearn.readthedocs.io/en/latest/api.html): log versus scaling implementations and Viterbi versus marginal-MAP objectives. No library performance or implementation-equivalence claim.
- [Python math.fsum](https://docs.python.org/3/library/math.html): accurate partial sums and platform rounding caveat. Native fsum and browser compensated sums are not exact-real arithmetic.

Primary tutorial, textbook, lecture notes and software documentation were checked. Established HMM theory; independent educational implementation. MIT; attribution is not endorsement.
