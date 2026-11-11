"use strict";
const test = require("node:test"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  { spawnSync } = require("node:child_process");
const path = require("node:path");
const E = require("../docs/path-engine.js"),
  controls = ["core", "rounded"].flatMap((kind) =>
    JSON.parse(
      fs.readFileSync(
        path.join(
          __dirname,
          "../chainlab/artifacts/path-controls-" + kind + ".json",
        ),
      ),
    ),
  );
function same(a, b, p = "report") {
  if (typeof b === "number" && !Number.isInteger(b)) {
    assert.equal(typeof a, "number", p);
    assert.ok(
      Number.isFinite(a) && Math.abs(a - b) < 1e-10,
      `${p}: ${a} vs ${b}`,
    );
  } else if (Array.isArray(b)) {
    assert.ok(Array.isArray(a), p);
    assert.equal(a.length, b.length, p);
    b.forEach((v, i) => same(a[i], v, p + "." + i));
  } else if (b && typeof b === "object") {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), p);
    for (const k of Object.keys(b)) same(a[k], b[k], p + "." + k);
  } else if (typeof b === "number") {
    assert.ok(Math.abs(a - b) < 1e-10, p);
  } else assert.deepEqual(a, b, p);
}
test("all13 complete exact native/BigInt reports, every probability and decision state", () => {
  assert.equal(controls.length, 13);
  for (const c of controls) same(E.audit(c.input), c.report, c.name);
});
test("separate losses, dependence and forbidden states remain visible", () => {
  const by = Object.fromEntries(
    controls.map((c) => [c.name, E.audit(c.input)]),
  );
  assert.equal(by.locked.entropy.path, Math.log(2));
  assert.ok(Math.abs(by.locked.entropy.sum_state - 5 * Math.log(2)) < 1e-12);
  assert.equal(by["large-support"].supported_paths, "18446744073709551616");
  const d = by["three-objectives"].decoders;
  assert.notDeepEqual(d.map.indices, d.supported_matches.indices);
  assert.notDeepEqual(d.supported_matches.indices, d.supported_product.indices);
  assert.equal(d.marginal_modes.supported, false);
  assert.equal(by["rare-evidence"].decoders.map.posterior, "1");
  assert.equal(by["forbidden-modes"].decoders.marginal_modes.posterior, "0");
});
test("strict JSON, shape, probability, bit and support rejection", () => {
  for (const text of [
    '{"model":1,"model":2,"sequence":[]}',
    '{"x":1,}',
    "[1,]",
    "{} x",
    "[01]",
    '{"x":"bad\n"}',
    '{"x":1}\u00a0',
  ])
    assert.throws(() => E.strictParse(text));
  assert.deepEqual(E.strictParse(' {"x": [true,false,null,2e2,"\\u0061"]} '), {
    x: [true, false, null, 200, "a"],
  });
  for (const mutate of [
    (x) => (x.model.initial[0] = 1),
    (x) => (x.model.initial[0] = "01"),
    (x) => (x.model.initial[0] = "-1"),
    (x) => (x.model.initial[0] = "1/0"),
    (x) => (x.model.initial[0] = "1/" + 2n ** 129n),
    (x) => (x.model.initial[0] = "0"),
    (x) => (x.sequence = []),
    (x) => (x.sequence = ["unknown"]),
    (x) => (x.sequence = Array(33).fill("x")),
    (x) => (x.extra = true),
    (x) => (x.model.extra = 1),
    (x) => (x.model.states = ["A", "A"]),
    (x) => (x.model.emission = [[1]]),
    (x) => (x.model.version = "chainlab_v1"),
  ]) {
    const x = structuredClone(controls[0].input);
    mutate(x);
    assert.throws(() => E.audit(x));
  }
  assert.throws(() => E.strictParse(" ".repeat(65537)));
});
test("all browser-produced control archives independently pass native verifier", () => {
  for (const c of controls) {
    const r = spawnSync("python", ["-m", "chainlab.path_verify", "-"], {
      cwd: __dirname + "/..",
      input: JSON.stringify(E.audit(c.input)),
      encoding: "utf8",
    });
    assert.equal(r.status, 0, c.name + ": " + r.stderr);
  }
});
module.exports = { same };
test(
  "150 generated models: every native/BigInt report and independent native check",
  { timeout: 120000 },
  () => {
    let seed = 8675309;
    const rand = (n) => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return (seed >>> 8) % n;
    };
    const row = (n) => {
      const xs = Array.from({ length: n }, () => rand(5));
      if (!xs.some(Boolean)) xs[0] = 1;
      const total = xs.reduce((a, b) => a + b, 0);
      return xs.map((v) => `${v}/${total}`);
    };
    const inputs = Array.from({ length: 150 }, () => {
      const n = 1 + rand(4),
        T = 1 + rand(6);
      return {
        model: {
          version: "chainlab_rational_v1",
          states: Array.from({ length: n }, (_, i) => String(i)),
          symbols: ["x", "y"],
          initial: row(n),
          transition: Array.from({ length: n }, () => row(n)),
          emission: Array.from({ length: n }, () => row(2)),
        },
        sequence: Array.from({ length: T }, () => (rand(2) ? "x" : "y")),
      };
    });
    const code =
      'import sys,json\nfrom chainlab.pathrisk import audit\nfrom chainlab.path_verify import check\nout=[]\nfor x in json.load(sys.stdin):\n try:\n  r=audit(x);check(r);out.append(r)\n except ValueError as e:out.append({"error":str(e)})\nprint(json.dumps(out,allow_nan=False))';
    const result = spawnSync("python", ["-c", code], {
      cwd: __dirname + "/..",
      input: JSON.stringify(inputs),
      encoding: "utf8",
      maxBuffer: 16 * 1024 * 1024,
    });
    assert.equal(result.status, 0, result.stderr);
    const refs = JSON.parse(result.stdout);
    let feasible = 0;
    refs.forEach((r, i) => {
      if (r.error) assert.throws(() => E.audit(inputs[i]));
      else {
        same(E.audit(inputs[i]), r, "random" + i);
        feasible++;
      }
    });
    console.log(
      JSON.stringify({
        generated: 150,
        feasible,
        allOtherModelsRejected: true,
      }),
    );
  },
);
