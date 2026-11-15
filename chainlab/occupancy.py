"""Exact posterior visit-count laws for the bounded rational HMM contract."""
from fractions import Fraction as F
from .pathrisk import parse, _bounded, _pack


def visit_distribution(payload, selected_states):
    """Count selected-state visits over the whole observed window, including t=0.

    The coefficient trellis sums joint observation/path mass by terminal state
    and visit count. It never treats smoothed state marginals as independent.
    Empty and full state selections are allowed. Inputs are not changed.
    """
    model, obs = parse(payload)
    names = model['states']
    if (not isinstance(selected_states, list)
            or any(type(s) is not str or s not in names for s in selected_states)
            or len(set(selected_states)) != len(selected_states)):
        raise ValueError('selected_states must be a list of distinct known state names')
    chosen = [name in selected_states for name in names]
    n = len(names)
    first = [[F(0), F(0)] for _ in names]
    for j in range(n):
        first[j][int(chosen[j])] = _bounded(model['initial'][j] * model['emission'][j][obs[0]])
    trellis = [first]
    for t, symbol in enumerate(obs[1:], 1):
        previous = trellis[-1]
        current = [[F(0) for _ in range(t + 2)] for _ in names]
        for j in range(n):
            shift = int(chosen[j])
            for k in range(t + 1):
                current[j][k + shift] = _bounded(model['emission'][j][symbol] * sum(
                    previous[i][k] * model['transition'][i][j] for i in range(n)))
        trellis.append(current)
    joint = [_bounded(sum(row[k] for row in trellis[-1])) for k in range(len(obs) + 1)]
    evidence = _bounded(sum(joint))
    if not evidence:
        raise ValueError('observation sequence has zero probability; no posterior exists')
    probabilities = [_bounded(v / evidence) for v in joint]
    tails = [F(0) for _ in range(len(probabilities) + 1)]
    for k in range(len(probabilities) - 1, -1, -1):
        tails[k] = _bounded(probabilities[k] + tails[k + 1])
    mean = _bounded(sum(k * p for k, p in enumerate(probabilities)))
    variance = _bounded(sum((k - mean) ** 2 * p for k, p in enumerate(probabilities)))
    return _pack(dict(
        version='chainlab_visit_report_v1',
        input=dict(model=model, sequence=payload['sequence'][:]),
        selected_states=[name for name in names if name in selected_states],
        prefix_joint_count_mass=trellis,
        evidence=evidence,
        joint_count_mass=joint,
        posterior_count_probabilities=probabilities,
        posterior_count_at_least=tails,
        expected_count=mean,
        count_variance=variance,
        limits=dict(states=4, symbols=8, length=32, input_bits=128, work_bits=8192),
        scope='Exact rational law conditional on the complete supplied observation window. '
              'Count includes the initial state and each later state once. Array index is the count; '
              'tail index k means P(count>=k), including the terminal zero at length+1. '
              'Trellis axes are time, terminal state, count; cells are joint prefix evidence, not posterior probabilities. '
              'Not online filtering, independent marginal trials, a state-recovery or calibration guarantee. '
              'Separate rational model contract; no fit or original binary64 inference changes.'
    ))
