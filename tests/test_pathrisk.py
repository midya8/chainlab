import copy, itertools, json, math, random, subprocess, sys, unittest
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
from chainlab.pathrisk import audit, parse
from chainlab.path_verify import check

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = [
    c
    for kind in ["core", "rounded"]
    for c in json.loads(
        (ROOT / ("chainlab/artifacts/path-controls-" + kind + ".json")).read_text()
    )
]
BY = {c["name"]: c for c in CONTROLS}


def enumerated(inp):
    # Tiny Cartesian oracle: no production parsing or message-passing helpers.
    m = inp["model"]
    n = len(m["states"])
    obs = [m["symbols"].index(s) for s in inp["sequence"]]
    weights = []
    for path in itertools.product(range(n), repeat=len(obs)):
        v = F(m["initial"][path[0]]) * F(m["emission"][path[0]][obs[0]])
        for t in range(1, len(obs)):
            v *= F(m["transition"][path[t - 1]][path[t]]) * F(
                m["emission"][path[t]][obs[t]]
            )
        weights.append((path, v))
    z = sum(v for _, v in weights)
    if not z:
        return None
    post = [(p, v / z) for p, v in weights if v]
    g = [
        [sum(v for p, v in post if p[t] == i) for i in range(n)]
        for t in range(len(obs))
    ]
    xi = [
        [
            [sum(v for p, v in post if p[t] == i and p[t + 1] == j) for j in range(n)]
            for i in range(n)
        ]
        for t in range(len(obs) - 1)
    ]
    with localcontext() as ctx:
        ctx.prec = 80
        ent = -sum(
            (
                Decimal(v.numerator)
                / Decimal(v.denominator)
                * (Decimal(v.numerator) / Decimal(v.denominator)).ln()
                for _, v in post
            ),
            Decimal(0),
        )
    scores = {
        "map": lambda p, v: v,
        "supported_matches": lambda p, v: sum(g[t][i] for t, i in enumerate(p)),
        "supported_product": lambda p, v: math.prod(g[t][i] for t, i in enumerate(p)),
    }
    best = {
        name: min(post, key=lambda pair: (-score(*pair), pair[0]))[0]
        for name, score in scores.items()
    }
    return z, g, xi, float(ent), best, len(post)


class PathRiskTests(unittest.TestCase):
    def test_saved_reports_rebuild_and_independent_verifier(self):
        self.assertEqual(len(CONTROLS), 13)
        for c in CONTROLS:
            self.assertEqual(audit(c["input"]), c["report"])
            self.assertTrue(check(c["report"]))

    def test_all_tiny_controls_cartesian_fraction_decimal(self):
        for c in CONTROLS:
            if len(c["input"]["model"]["states"]) ** len(c["input"]["sequence"]) > 5000:
                continue
            self.assert_oracle(c["input"])

    def assert_oracle(self, inp):
        oracle = enumerated(inp)
        if oracle is None:
            with self.assertRaises(ValueError):
                audit(inp)
            return
        z, g, xi, h, best, count = oracle
        r = audit(inp)
        self.assertEqual(F(r["evidence"]), z)
        self.assertEqual([[F(v) for v in row] for row in r["state"]], g)
        self.assertEqual(
            [[[F(v) for v in row] for row in edge] for edge in r["edge"]], xi
        )
        self.assertAlmostEqual(r["entropy"]["path"], h, places=10)
        self.assertEqual(int(r["supported_paths"]), count)
        for k, p in best.items():
            self.assertEqual(r["decoders"][k]["indices"], list(p))
        self.assertTrue(check(r))

    def test_120_random_full_cartesian_models(self):
        rng = random.Random(271828)

        def row(k):
            xs = [rng.randrange(5) for _ in range(k)]
            if sum(xs) == 0:
                xs[0] = 1
            return [str(F(v, sum(xs))) for v in xs]

        for _ in range(120):
            n = rng.randrange(1, 4)
            T = rng.randrange(1, 6)
            m = dict(
                version="chainlab_rational_v1",
                states=[str(i) for i in range(n)],
                symbols=["x", "y"],
                initial=row(n),
                transition=[row(n) for _ in range(n)],
                emission=[row(2) for _ in range(n)],
            )
            self.assert_oracle(
                dict(model=m, sequence=[rng.choice(m["symbols"]) for _ in range(T)])
            )

    def test_same_marginals_different_path_entropy(self):
        independent = BY["independent"]["report"]
        locked = BY["locked"]["report"]
        self.assertEqual(independent["state"], locked["state"])
        self.assertAlmostEqual(locked["entropy"]["path"], math.log(2))
        self.assertAlmostEqual(independent["entropy"]["path"], 5 * math.log(2))
        self.assertAlmostEqual(locked["entropy"]["dependence_gap"], 4 * math.log(2))

    def test_three_different_objectives_and_impossible_modes(self):
        d = BY["three-objectives"]["report"]["decoders"]
        self.assertEqual(
            len(
                {
                    tuple(d[k]["indices"])
                    for k in ["map", "supported_matches", "supported_product"]
                }
            ),
            3,
        )
        self.assertGreater(
            F(d["map"]["posterior"]), F(d["supported_matches"]["posterior"])
        )
        self.assertLess(
            F(d["supported_matches"]["expected_hamming"]),
            F(d["map"]["expected_hamming"]),
        )
        self.assertLess(
            F(d["supported_matches"]["expected_hamming"]),
            F(d["supported_product"]["expected_hamming"]),
        )
        self.assertFalse(d["marginal_modes"]["supported"])

    def test_forbidden_mode_not_supported(self):
        d = BY["forbidden-modes"]["report"]["decoders"]
        self.assertEqual(d["marginal_modes"]["path"], ["A", "A"])
        self.assertEqual(d["marginal_modes"]["posterior"], "0")
        self.assertIsNone(d["marginal_modes"]["log_posterior"])
        self.assertEqual(d["map"]["path"], ["B", "A"])

    def test_emission_constraints_and_zero_occupancy(self):
        r = BY["emission-support"]["report"]
        self.assertEqual(r["supported_paths"], "1")
        self.assertEqual(r["decoders"]["map"]["path"], ["A", "B", "A"])
        self.assertEqual(
            BY["zero-occupancy"]["report"]["conditional"][0][1], [None, None]
        )

    def test_canonical_whole_path_ties(self):
        for k in ["map", "supported_matches", "supported_product"]:
            self.assertEqual(
                BY["canonical-tie"]["report"]["decoders"][k]["indices"], [0, 1, 0, 1]
            )

    def test_beyond_safe_js_counts(self):
        r = BY["large-support"]["report"]
        self.assertEqual(r["supported_paths"], str(4**32))
        self.assertEqual(r["decoders"]["map"]["posterior"], str(F(1, 4**32)))

    def test_underflow_does_not_erase_evidence_or_support(self):
        r = BY["rare-evidence"]["report"]
        self.assertEqual(float(F(r["evidence"])), 0)
        self.assertTrue(math.isfinite(r["log_evidence"]))
        self.assertEqual(r["decoders"]["map"]["posterior"], "1")
        self.assertEqual(r["entropy"]["path"], 0)

    def test_mass_pair_consistency_chain_entropy_identity(self):
        for c in CONTROLS:
            r = c["report"]
            g = [[F(x) for x in row] for row in r["state"]]
            for row in g:
                self.assertEqual(sum(row), 1)
            for t, edge in enumerate(r["edge"]):
                v = [[F(x) for x in row] for row in edge]
                self.assertEqual([sum(row) for row in v], g[t])
                self.assertEqual(
                    [sum(row[j] for row in v) for j in range(len(g[0]))], g[t + 1]
                )
            self.assertAlmostEqual(
                r["entropy"]["dependence_gap"],
                r["entropy"]["sum_edge_mutual_information"],
                places=10,
            )
            self.assertGreaterEqual(r["entropy"]["path"], -1e-10)
            self.assertLessEqual(
                r["entropy"]["path"], r["entropy"]["sum_state"] + 1e-10
            )

    def test_parser_bounds_and_no_mutation(self):
        inp = copy.deepcopy(CONTROLS[0]["input"])
        old = copy.deepcopy(inp)
        audit(inp)
        self.assertEqual(inp, old)
        for value in [
            True,
            1,
            1.0,
            "-1",
            "01",
            "1/0",
            "1/02",
            " 1",
            "1e0",
            "2",
            str(2**129) + "/1",
            "1/" + str(2**129),
        ]:
            x = copy.deepcopy(inp)
            x["model"]["initial"] = [value]
            with self.assertRaises(ValueError, msg=str(value)):
                audit(x)
        for key, value in [
            ("sequence", []),
            ("sequence", ["?"]),
            ("sequence", ["x"] * 33),
            ("extra", 1),
        ]:
            x = copy.deepcopy(inp)
            x[key] = value
            with self.assertRaises(ValueError):
                audit(x)

    def test_internal_work_bound_is_explicit(self):
        x = copy.deepcopy(BY["rare-evidence"]["input"])
        x["model"]["states"] = ["A", "B"]
        x["model"]["initial"] = ["1/2", "1/2"]
        x["model"]["transition"] = [["1/3", "2/3"], ["2/7", "5/7"]]
        x["model"]["emission"] = [
            ["1/" + str(2**120), "%s/%s" % (2**120 - 1, 2**120)],
            ["1/3", "2/3"],
        ]
        with self.assertRaisesRegex(ValueError, "8192-bit"):
            audit(x)

    def test_impossible_sequence_and_single_step(self):
        x = copy.deepcopy(CONTROLS[0]["input"])
        x["model"]["emission"] = [["1", "0"]]
        x["sequence"] = ["y"]
        with self.assertRaisesRegex(ValueError, "zero probability"):
            audit(x)
        x["sequence"] = ["x"]
        r = audit(x)
        self.assertEqual(r["edge"], [])
        self.assertEqual(r["entropy"]["conditional"], [])
        self.assertTrue(check(r))

    def test_all_primitive_leaf_forgeries_default(self):
        r = BY["three-objectives"]["report"]
        paths = []

        def visit(x, p):
            if isinstance(x, dict):
                for k, v in x.items():
                    visit(v, p + [k])
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    visit(v, p + [i])
            else:
                paths.append(p)

        visit(r, [])
        self.assertEqual(len(paths), 438)
        for path in paths:
            forged = copy.deepcopy(r)
            node = forged
            for k in path[:-1]:
                node = node[k]
            v = node[path[-1]]
            node[path[-1]] = (
                not v
                if type(v) is bool
                else (
                    "bad"
                    if v is None
                    else (v + 1 if type(v) in (int, float) else v + "x")
                )
            )
            with self.assertRaises(
                (ValueError, TypeError, IndexError, KeyError), msg=str(path)
            ):
                check(forged)
        self.assertTrue(check(r))

    def test_verifier_independent_imports(self):
        code = (ROOT / "chainlab/path_verify.py").read_text()
        self.assertNotIn("from .pathrisk", code)
        self.assertNotIn("import pathrisk", code)

    def test_cli_source_bound_report_and_rejects_duplicates(self):
        def run(module, text):
            return subprocess.run(
                [sys.executable, "-m", module, "-"],
                input=text,
                text=True,
                capture_output=True,
                cwd=ROOT,
            )

        result = run("chainlab.pathrisk", json.dumps(BY["three-objectives"]["input"]))
        self.assertEqual(result.returncode, 0, result.stderr)
        verified = run("chainlab.path_verify", result.stdout)
        self.assertEqual(verified.returncode, 0, verified.stderr)
        for text in ['{"model":1,"model":2,"sequence":[]}', " " * 65537, '{"x":NaN}']:
            bad = run("chainlab.pathrisk", text)
            self.assertEqual(bad.returncode, 2)
            self.assertEqual(bad.stdout, "")
        bad = run("chainlab.path_verify", " " * 4194305)
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(bad.stdout, "")


if __name__ == "__main__":
    unittest.main()
