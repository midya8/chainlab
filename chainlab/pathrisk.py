"""Exact finite-HMM path objectives, with approximate entropy in nats.

Separate from the original binary64 trainer and prefix inspector. All support,
probability and decoding comparisons use fractions; only logarithms round.
"""

from fractions import Fraction as F
import math
import re

KEYS = {"version", "states", "symbols", "initial", "transition", "emission"}
PATTERN = re.compile(r"(?:0|[1-9][0-9]*)(?:/(?:[1-9][0-9]*))?\Z")


def _names(xs, cap):
    if (
        not isinstance(xs, list)
        or not 1 <= len(xs) <= cap
        or any(type(x) is not str or not 1 <= len(x) <= 40 for x in xs)
        or len(set(xs)) != len(xs)
    ):
        raise ValueError(
            "names must be distinct strings, length1..40, within model limits"
        )
    return xs[:]


def _row(xs, n):
    if not isinstance(xs, list) or len(xs) != n:
        raise ValueError("probability row has wrong shape")
    out = []
    for x in xs:
        if type(x) is not str or len(x) > 80 or not PATTERN.fullmatch(x):
            raise ValueError("probabilities must be unsigned integer/fraction strings")
        p = F(x)
        if (
            p.numerator.bit_length() > 128
            or p.denominator.bit_length() > 128
            or not 0 <= p <= 1
        ):
            raise ValueError("probability outside[0,1] or128-bit input limit")
        out.append(p)
    if sum(out) != 1:
        raise ValueError("each probability row must sum to exactly1")
    return out


def parse(payload):
    if not isinstance(payload, dict) or set(payload) != {"model", "sequence"}:
        raise ValueError("input requires model and sequence only")
    m = payload["model"]
    if (
        not isinstance(m, dict)
        or set(m) != KEYS
        or m["version"] != "chainlab_rational_v1"
    ):
        raise ValueError("invalid rational model schema")
    states = _names(m["states"], 4)
    symbols = _names(m["symbols"], 8)
    n = len(states)
    model = dict(
        version=m["version"],
        states=states,
        symbols=symbols,
        initial=_row(m["initial"], n),
    )
    for k, w in [("transition", n), ("emission", len(symbols))]:
        if not isinstance(m[k], list) or len(m[k]) != n:
            raise ValueError("one " + k + " row required per state")
        model[k] = [_row(row, w) for row in m[k]]
    seq = payload["sequence"]
    if (
        not isinstance(seq, list)
        or not 1 <= len(seq) <= 32
        or any(type(s) is not str or s not in symbols for s in seq)
    ):
        raise ValueError("sequence must contain1..32 known symbol names")
    return model, [symbols.index(s) for s in seq]


def _bounded(p):
    if max(p.numerator.bit_length(), p.denominator.bit_length()) > 8192:
        raise ValueError(
            "exact work exceeds8192-bit fraction limit; simplify probabilities or shorten sequence"
        )
    return p


def _log(p):
    if not p:
        return None
    v = float(p)
    return math.log(v) if v else math.log(p.numerator) - math.log(p.denominator)


def _entropy(row):
    return math.fsum(-float(p) * _log(p) for p in row if p)


def _pack(x):
    if isinstance(x, F):
        return str(_bounded(x))
    if isinstance(x, list):
        return [_pack(v) for v in x]
    if isinstance(x, dict):
        return {k: _pack(v) for k, v in x.items()}
    return x


def audit(payload):
    model, obs = parse(payload)
    n = len(model["states"])
    T = len(obs)
    a = model["transition"]
    e = model["emission"]
    p = model["initial"]
    f = [[p[i] * e[i][obs[0]] for i in range(n)]]
    for o in obs[1:]:
        f.append(
            [
                _bounded(e[j][o] * sum(f[-1][i] * a[i][j] for i in range(n)))
                for j in range(n)
            ]
        )
    z = sum(f[-1])
    if not z:
        raise ValueError(
            "observation sequence has zero probability; no posterior exists"
        )
    b = [[F(1)] * n for _ in obs]
    for t in range(T - 2, -1, -1):
        b[t] = [
            _bounded(sum(a[i][j] * e[j][obs[t + 1]] * b[t + 1][j] for j in range(n)))
            for i in range(n)
        ]
    gamma = [[f[t][i] * b[t][i] / z for i in range(n)] for t in range(T)]
    xi = [
        [
            [f[t][i] * a[i][j] * e[j][obs[t + 1]] * b[t + 1][j] / z for j in range(n)]
            for i in range(n)
        ]
        for t in range(T - 1)
    ]
    conditional = [
        [
            [xi[t][i][j] / gamma[t][i] if gamma[t][i] else None for j in range(n)]
            for i in range(n)
        ]
        for t in range(T - 1)
    ]
    ent = [_entropy(row) for row in gamma]
    cond = [
        math.fsum(
            float(gamma[t][i]) * _entropy([v for v in rows[i] if v is not None])
            for i in range(n)
        )
        for t, rows in enumerate(conditional)
    ]
    mutual = [
        math.fsum(
            float(v) * (_log(v) - _log(gamma[t][i]) - _log(gamma[t + 1][j]))
            for i, row in enumerate(edge)
            for j, v in enumerate(row)
            if v
        )
        for t, edge in enumerate(xi)
    ]
    path_entropy = math.fsum([ent[0]] + cond)
    marginal_entropy = math.fsum(ent)
    trellises = {}
    decoders = {}
    for name in ["map", "supported_matches", "supported_product"]:
        rows = []
        for t in range(T):
            cells = []
            for j in range(n):
                candidates = []
                if t == 0:
                    if p[j] * e[j][obs[0]]:
                        value = p[j] * e[j][obs[0]] if name == "map" else gamma[t][j]
                        candidates = [(value, [j])]
                elif e[j][obs[t]]:
                    for i, cell in enumerate(rows[-1]):
                        if cell is not None and a[i][j]:
                            value = (
                                cell["score"] + gamma[t][j]
                                if name == "supported_matches"
                                else cell["score"]
                                * (
                                    a[i][j] * e[j][obs[t]]
                                    if name == "map"
                                    else gamma[t][j]
                                )
                            )
                            candidates.append((_bounded(value), cell["path"] + [j]))
                if candidates:
                    best = min(candidates, key=lambda x: (-x[0], x[1]))
                    cells.append(dict(score=best[0], path=best[1]))
                else:
                    cells.append(None)
            rows.append(cells)
        best = min(
            [v for v in rows[-1] if v is not None],
            key=lambda x: (-x["score"], x["path"]),
        )
        trellises[name] = rows
        decoders[name] = dict(objective=best["score"], indices=best["path"])
    modes = [max(range(n), key=lambda i: gamma[t][i]) for t in range(T)]
    decoders["marginal_modes"] = dict(
        objective=sum(gamma[t][i] for t, i in enumerate(modes)), indices=modes
    )
    for d in decoders.values():
        path = d["indices"]
        joint = p[path[0]] * e[path[0]][obs[0]]
        for t in range(1, T):
            joint *= a[path[t - 1]][path[t]] * e[path[t]][obs[t]]
        matches = sum(gamma[t][i] for t, i in enumerate(path))
        d.update(
            path=[model["states"][i] for i in path],
            joint=joint,
            posterior=joint / z,
            log_posterior=_log(joint / z),
            supported=bool(joint),
            expected_matches=matches,
            expected_hamming=F(T) - matches,
            marginal_product=math.prod(gamma[t][i] for t, i in enumerate(path)),
        )
    counts = [[int(bool(p[i] * e[i][obs[0]])) for i in range(n)]]
    for t in range(1, T):
        counts.append(
            [
                sum(counts[-1][i] for i in range(n) if a[i][j]) if e[j][obs[t]] else 0
                for j in range(n)
            ]
        )
    canonical = _pack(model)
    return _pack(
        dict(
            version="chainlab_path_report_v1",
            input=dict(model=canonical, sequence=payload["sequence"][:]),
            forward=f,
            backward=b,
            evidence=z,
            log_evidence=_log(z),
            state=gamma,
            edge=xi,
            conditional=conditional,
            support_counts=[[str(v) for v in row] for row in counts],
            supported_paths=str(sum(counts[-1])),
            decoders=decoders,
            trellises=trellises,
            entropy=dict(
                state=ent,
                conditional=cond,
                edge_mutual_information=mutual,
                path=path_entropy,
                sum_state=marginal_entropy,
                dependence_gap=marginal_entropy - path_entropy,
                sum_edge_mutual_information=math.fsum(mutual),
            ),
            limits=dict(
                states=4,
                symbols=8,
                length=32,
                input_bits=128,
                work_bits=8192,
                entropy="Approximate binary64 nats from exact probabilities; model-conditional, not calibrated uncertainty.",
            ),
            tie_rule="Lexicographically first whole index path among exact objective ties. Marginal modes keep first index independently.",
        )
    )


def main(argv=None):
    import argparse, json, sys
    from .cli import _unique

    p = argparse.ArgumentParser(
        description="Exact finite-HMM path objectives; approximate entropy"
    )
    p.add_argument("input")
    args = p.parse_args(argv)
    try:
        if args.input == "-":
            text = sys.stdin.read(65537)
        else:
            with open(args.input, encoding="utf-8") as f:
                text = f.read(65537)
        if len(text.encode("utf-8")) > 65536:
            raise ValueError("input exceeds64KiB")
        r = audit(json.loads(text, object_pairs_hook=_unique))
        out = __import__("json").dumps(r, allow_nan=False, sort_keys=True, indent=2)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as err:
        print("error: " + str(err), file=sys.stderr)
        return 2
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
