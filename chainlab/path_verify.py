"""Check path reports without importing the producer or its parser.

Exact probability, support and decision claims are checked by a suffix oracle,
independent prefix sums and all Bellman alternatives. Decimal80 logarithms check
approximate entropy fields. This is bounded model arithmetic, not model validation
against the world. Floats use1e-10 absolute tolerance; rational fields are exact.
"""

from fractions import Fraction as Q
from decimal import Decimal as D, localcontext
from functools import lru_cache
import math, re, json


def check(report):
    def require(ok, where):
        if not ok:
            raise ValueError("report mismatch: " + where)

    require(type(report) is dict, "object")
    inp = report.get("input")
    require(type(inp) is dict and set(inp) == {"model", "sequence"}, "input keys")
    m = inp["model"]
    seq = inp["sequence"]
    require(
        type(m) is dict
        and set(m)
        == {"version", "states", "symbols", "initial", "transition", "emission"},
        "model keys",
    )
    require(m["version"] == "chainlab_rational_v1", "model version")
    for field, cap in [("states", 4), ("symbols", 8)]:
        xs = m[field]
        require(
            type(xs) is list
            and 1 <= len(xs) <= cap
            and all(type(v) is str and 1 <= len(v) <= 40 for v in xs)
            and len(set(xs)) == len(xs),
            field,
        )
    n = len(m["states"])
    require(
        type(seq) is list
        and 1 <= len(seq) <= 32
        and all(type(v) is str and v in m["symbols"] for v in seq),
        "sequence",
    )
    T = len(seq)
    obs = [m["symbols"].index(s) for s in seq]

    def row(xs, k):
        require(type(xs) is list and len(xs) == k, "row shape")
        out = []
        for x in xs:
            require(
                type(x) is str
                and len(x) <= 80
                and re.fullmatch(r"(0|[1-9][0-9]*)(/[1-9][0-9]*)?", x) is not None,
                "rational syntax",
            )
            v = Q(x)
            require(
                0 <= v <= 1
                and v.numerator.bit_length() <= 128
                and v.denominator.bit_length() <= 128,
                "rational bounds",
            )
            require(str(v) == x, "canonical rational")
            out.append(v)
        require(sum(out) == 1, "row mass")
        return out

    p = row(m["initial"], n)
    a = []
    e = []
    for name, target, k in [("transition", a, n), ("emission", e, len(m["symbols"]))]:
        require(type(m[name]) is list and len(m[name]) == n, name + " shape")
        target.extend(row(xs, k) for xs in m[name])

    @lru_cache(None)
    def tail(t, i):
        if t == T - 1:
            return Q(1)
        return sum(a[i][j] * e[j][obs[t + 1]] * tail(t + 1, j) for j in range(n))

    prefixes = []
    cur = {i: p[i] * e[i][obs[0]] for i in range(n)}
    for t in range(T):
        prefixes.append(cur)
        cur = (
            {
                j: sum(cur[i] * a[i][j] for i in range(n)) * e[j][obs[t + 1]]
                for j in range(n)
            }
            if t < T - 1
            else {}
        )
    z = sum(prefixes[-1].values())
    require(z > 0, "nonzero evidence")
    g = [[prefixes[t][i] * tail(t, i) / z for i in range(n)] for t in range(T)]
    edges = [
        [
            [
                prefixes[t][i] * a[i][j] * e[j][obs[t + 1]] * tail(t + 1, j) / z
                for j in range(n)
            ]
            for i in range(n)
        ]
        for t in range(T - 1)
    ]

    def packed(x):
        if type(x) is Q:
            require(
                max(x.numerator.bit_length(), x.denominator.bit_length()) <= 8192,
                "work fraction bound",
            )
            return str(x)
        if type(x) is list:
            return [packed(v) for v in x]
        if type(x) is dict:
            return {k: packed(v) for k, v in x.items()}
        return x

    def compare(actual, expected, path):
        if type(expected) is float:
            require(
                type(actual) in (int, float)
                and math.isfinite(actual)
                and abs(actual - expected) <= 1e-10,
                path,
            )
        elif type(expected) is dict:
            require(
                type(actual) is dict and actual.keys() == expected.keys(),
                path + " keys",
            )
            for k, v in expected.items():
                compare(actual[k], v, path + "." + k)
        elif type(expected) is list:
            require(
                type(actual) is list and len(actual) == len(expected), path + " shape"
            )
            for i, v in enumerate(expected):
                compare(actual[i], v, path + "." + str(i))
        else:
            require(type(actual) is type(expected) and actual == expected, path)

    with localcontext() as ctx:
        ctx.prec = 80

        def logq(q):
            return None if not q else (D(q.numerator) / D(q.denominator)).ln()

        def h(qs):
            return sum(
                (-D(q.numerator) / D(q.denominator) * logq(q) for q in qs if q), D(0)
            )

        ent = [h(xs) for xs in g]
        cond = [
            sum(
                (
                    D(g[t][i].numerator)
                    / D(g[t][i].denominator)
                    * h([v / g[t][i] for v in edge[i]])
                    for i in range(n)
                    if g[t][i]
                ),
                D(0),
            )
            for t, edge in enumerate(edges)
        ]
        mi = [
            sum(
                (
                    D(v.numerator)
                    / D(v.denominator)
                    * (logq(v) - logq(g[t][i]) - logq(g[t + 1][j]))
                    for i, erow in enumerate(edge)
                    for j, v in enumerate(erow)
                    if v
                ),
                D(0),
            )
            for t, edge in enumerate(edges)
        ]
        Hp = ent[0] + sum(cond)
        Hs = sum(ent)
        entropy = {
            "state": list(map(float, ent)),
            "conditional": list(map(float, cond)),
            "edge_mutual_information": list(map(float, mi)),
            "path": float(Hp),
            "sum_state": float(Hs),
            "dependence_gap": float(Hs - Hp),
            "sum_edge_mutual_information": float(sum(mi)),
        }
        logz = float(logq(z))
        # Rebuild every Bellman state independently. No producer functions or
        # stored references enter the candidate list. Compare full tied paths.
        trellises = {}
        choices = {}
        for method in ["map", "supported_matches", "supported_product"]:
            table = []
            for t in range(T):
                cells = []
                for j in range(n):
                    candidates = []
                    if not t and p[j] and e[j][obs[0]]:
                        candidates = [
                            {
                                "path": [j],
                                "score": (
                                    p[j] * e[j][obs[0]] if method == "map" else g[0][j]
                                ),
                            }
                        ]
                    elif t and e[j][obs[t]]:
                        for i in range(n):
                            prev = table[t - 1][i]
                            if prev is None or not a[i][j]:
                                continue
                            score = (
                                prev["score"] + g[t][j]
                                if method == "supported_matches"
                                else prev["score"]
                                * (
                                    g[t][j]
                                    if method == "supported_product"
                                    else a[i][j] * e[j][obs[t]]
                                )
                            )
                            candidates.append(
                                {"path": prev["path"] + [j], "score": score}
                            )
                    cells.append(
                        sorted(candidates, key=lambda v: (-v["score"], v["path"]))[0]
                        if candidates
                        else None
                    )
                table.append(cells)
            win = sorted(
                [v for v in table[-1] if v is not None],
                key=lambda v: (-v["score"], v["path"]),
            )[0]
            trellises[method] = table
            choices[method] = (win["path"], win["score"])
        modes = [sorted(range(n), key=lambda i: (-g[t][i], i))[0] for t in range(T)]
        choices["marginal_modes"] = (modes, sum(g[t][i] for t, i in enumerate(modes)))
        decoders = {}
        for name, (path, score) in choices.items():
            joint = p[path[0]] * e[path[0]][obs[0]]
            for t in range(1, T):
                joint *= e[path[t]][obs[t]] * a[path[t - 1]][path[t]]
            matches = sum(g[t][i] for t, i in enumerate(path))
            product = Q(1)
            for t, i in enumerate(path):
                product *= g[t][i]
            decoders[name] = dict(
                indices=path,
                objective=score,
                path=[m["states"][i] for i in path],
                joint=joint,
                posterior=joint / z,
                log_posterior=float(logq(joint / z)) if joint else None,
                supported=bool(joint),
                expected_matches=matches,
                expected_hamming=Q(T) - matches,
                marginal_product=product,
            )
    counts = [[int(bool(p[i] * e[i][obs[0]])) for i in range(n)]]
    for t in range(1, T):
        counts.append(
            [
                sum(counts[-1][i] for i in range(n) if a[i][j]) if e[j][obs[t]] else 0
                for j in range(n)
            ]
        )
    expected = dict(
        version="chainlab_path_report_v1",
        input=inp,
        forward=[[prefixes[t][i] for i in range(n)] for t in range(T)],
        backward=[[tail(t, i) for i in range(n)] for t in range(T)],
        evidence=z,
        log_evidence=logz,
        state=g,
        edge=edges,
        conditional=[
            [[v / g[t][i] if g[t][i] else None for v in edge[i]] for i in range(n)]
            for t, edge in enumerate(edges)
        ],
        support_counts=[[str(v) for v in row] for row in counts],
        supported_paths=str(sum(counts[-1])),
        decoders=decoders,
        trellises=trellises,
        entropy=entropy,
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
    compare(report, packed(expected), "report")
    return True


def main(argv=None):
    import argparse, sys

    p = argparse.ArgumentParser(
        description="Independent exact path report verifier, Decimal80 entropy check"
    )
    p.add_argument("report")
    args = p.parse_args(argv)

    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate JSON key")
            out[k] = v
        return out

    try:
        if args.report == "-":
            text = sys.stdin.read(4194305)
        else:
            with open(args.report, encoding="utf-8") as f:
                text = f.read(4194305)
        if len(text.encode()) > 4194304:
            raise ValueError("report exceeds4MiB")
        check(json.loads(text, object_pairs_hook=unique))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
        RecursionError,
    ) as err:
        print("error: " + str(err), file=sys.stderr)
        return 2
    print(
        "Verified exact probabilities, support and decoding; approximate entropy within1e-10 nats."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
