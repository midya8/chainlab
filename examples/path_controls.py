import json, random
from fractions import Fraction as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chainlab.pathrisk import audit
from chainlab.path_verify import check

root = Path(__file__).resolve().parents[1]


def request(p, a, e, seq):
    def fmt(x):
        return str(F(x))

    return dict(
        model=dict(
            version="chainlab_rational_v1",
            states=[chr(65 + i) for i in range(len(p))],
            symbols=["x", "y"][: len(e[0])],
            initial=list(map(fmt, p)),
            transition=[list(map(fmt, r)) for r in a],
            emission=[list(map(fmt, r)) for r in e],
        ),
        sequence=seq,
    )


c = {}
c["one-state"] = request([1], [[1]], [[F(1, 3), F(2, 3)]], ["x", "y", "x"])
c["independent"] = request([F(1, 2)] * 2, [[F(1, 2)] * 2] * 2, [[1]] * 2, ["x"] * 5)
c["locked"] = request([F(1, 2)] * 2, [[1, 0], [0, 1]], [[1]] * 2, ["x"] * 5)
c["forbidden-modes"] = request(
    [F(2, 5), F(7, 20), F(1, 4)],
    [[0, F(1, 2), F(1, 2)], [1, 0, 0], [1, 0, 0]],
    [[1]] * 3,
    ["x"] * 2,
)
c["emission-support"] = request(
    [F(1, 2)] * 2, [[F(1, 2)] * 2] * 2, [[1, 0], [0, 1]], ["x", "y", "x"]
)
c["zero-occupancy"] = request([1, 0], [[1, 0], [0, 1]], [[1], [1]], ["x"] * 3)
c["canonical-tie"] = request([F(1, 2)] * 2, [[0, 1], [1, 0]], [[1]] * 2, ["x"] * 4)
c["large-support"] = request([F(1, 4)] * 4, [[F(1, 4)] * 4] * 4, [[1]] * 4, ["x"] * 32)
small = F(1, 2**120)
c["rare-evidence"] = request([1], [[1]], [[small, 1 - small]], ["x"] * 32)
rng = random.Random(314159)


def randrow(k):
    xs = [rng.randrange(5) for _ in range(k)]
    if not sum(xs):
        xs[0] = 1
    return [F(v, sum(xs)) for v in xs]


for k in range(5000):
    inp = request(
        randrow(3),
        [randrow(3) for _ in range(3)],
        [randrow(2) for _ in range(3)],
        ["x", "y", "x", "y", "x"],
    )
    try:
        r = audit(inp)
    except ValueError:
        continue
    ds = r["decoders"]
    if (
        ds["supported_matches"]["indices"] != ds["supported_product"]["indices"]
        and ds["map"]["indices"] != ds["supported_matches"]["indices"]
    ):
        c["three-objectives"] = inp
        print("first three-objective control search index", k)
        break
# Original artifact model values converted from the JSON decimal representation.
for name in ["trained", "initial", "reference"]:
    old = json.loads(
        (root / ("chainlab/artifacts/evidence-" + name + ".json")).read_text()
    )
    m = old["model"]
    converted = request(m["initial"], m["transition"], m["emission"], old["sequence"])
    # Round original JSON decimal values to millionths, then normalize exactly.
    for key in ["initial", "transition", "emission"]:

        def decimal(v):
            return [decimal(x) for x in v] if isinstance(v, list) else F(str(v))

        raw = decimal(m[key])
        rows = [raw] if key == "initial" else raw
        # This deliberately makes a different rational model, not an unchanged fit.
        quantized = [[round(x * 1000000) for x in row] for row in rows]
        normalized = [[str(F(x, sum(row))) for x in row] for row in quantized]
        converted["model"][key] = normalized[0] if key == "initial" else normalized
    converted["model"]["states"] = m["states"]
    converted["model"]["symbols"] = m["symbols"]
    c["millionth-rounded-" + name] = converted
out = []
for name, inp in c.items():
    r = audit(inp)
    check(r)
    out.append(dict(name=name, input=inp, report=r))
    print(
        name,
        "supported_paths=" + r["supported_paths"],
        "path_entropy=" + str(r["entropy"]["path"]),
    )
for kind, rows in [("core", out[:10]), ("rounded", out[10:])]:
    (root / ("chainlab/artifacts/path-controls-" + kind + ".json")).write_text(
        json.dumps(rows, separators=(",", ":"), allow_nan=False) + "\n"
    )
(root / "docs/path-data.js").write_text(
    "globalThis.PATH_DATA="
    + json.dumps(
        [{k: v for k, v in x.items() if k != "report"} for x in out],
        separators=(",", ":"),
    )
    + ";\n"
)
