"use strict";
// Independent BigInt finite-HMM inference. This code reads no native reports.
(function (root) {
  const gcd = (a, b) => {
    a = a < 0n ? -a : a;
    while (b) {
      const t = a % b;
      a = b;
      b = t;
    }
    return a;
  };
  class R {
    constructor(n, d = 1n) {
      if (!d) throw Error("Zero denominator");
      if (d < 0n) {
        n = -n;
        d = -d;
      }
      const g = gcd(n, d);
      this.n = n / g;
      this.d = d / g;
    }
    add(q) {
      return new R(this.n * q.d + q.n * this.d, this.d * q.d);
    }
    mul(q) {
      return new R(this.n * q.n, this.d * q.d);
    }
    div(q) {
      return new R(this.n * q.d, this.d * q.n);
    }
    sub(q) {
      return this.add(new R(-q.n, q.d));
    }
    cmp(q) {
      const v = this.n * q.d - q.n * this.d;
      return v > 0n ? 1 : v < 0n ? -1 : 0;
    }
    text() {
      if (this.n.toString(2).length > 8192 || this.d.toString(2).length > 8192)
        throw Error(
          "Exact work exceeds8192-bit fraction limit; simplify probabilities or shorten sequence",
        );
      return this.d === 1n ? String(this.n) : this.n + "/" + this.d;
    }
  }
  const zero = new R(0n),
    one = new R(1n),
    sum = (xs) => xs.reduce((a, b) => a.add(b), zero),
    prod = (xs) => xs.reduce((a, b) => a.mul(b), one),
    range = (n) => Array.from({ length: n }, (_, i) => i);
  function strictParse(text) {
    if (new TextEncoder().encode(text).length > 65536)
      throw Error("Input exceeds64KiB");
    let at = 0,
      depth = 0;
    const ws = () => {
      while (/[ \t\r\n]/.test(text[at] || "") && at < text.length) at++;
    };
    function string() {
      const start = at++;
      while (at < text.length) {
        const c = text[at++];
        if (c === "\\") {
          at++;
          continue;
        }
        if (c === '"') return JSON.parse(text.slice(start, at));
      }
      throw Error("Unterminated string");
    }
    function value() {
      ws();
      if (++depth > 64) throw Error("JSON nesting exceeds64");
      let x;
      const c = text[at];
      if (c === '"') x = string();
      else if (c === "[") {
        at++;
        x = [];
        ws();
        if (text[at] !== "]")
          while (true) {
            x.push(value());
            ws();
            if (text[at] !== ",") break;
            at++;
          }
        if (text[at++] !== "]") throw Error("Invalid array");
      } else if (c === "{") {
        at++;
        x = {};
        const seen = new Set();
        ws();
        if (text[at] !== "}")
          while (true) {
            ws();
            if (text[at] !== '"') throw Error("Expected JSON key");
            const k = string();
            if (seen.has(k)) throw Error("Duplicate JSON key");
            seen.add(k);
            ws();
            if (text[at++] !== ":") throw Error("Expected colon");
            Object.defineProperty(x, k, {
              value: value(),
              writable: true,
              enumerable: true,
              configurable: true,
            });
            ws();
            if (text[at] !== ",") break;
            at++;
          }
        if (text[at++] !== "}") throw Error("Invalid object");
      } else {
        const m =
          /^(?:true|false|null|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)/.exec(
            text.slice(at),
          );
        if (!m) throw Error("Invalid JSON value");
        at += m[0].length;
        x = JSON.parse(m[0]);
      }
      depth--;
      return x;
    }
    const out = value();
    ws();
    if (at !== text.length) throw Error("Trailing JSON data");
    return out;
  }
  const keys = (o, ks) =>
    o &&
    typeof o === "object" &&
    !Array.isArray(o) &&
    Object.keys(o).sort().join("|") === ks.slice().sort().join("|");
  function parse(payload) {
    if (!keys(payload, ["model", "sequence"]))
      throw Error("Input requires model and sequence only");
    const m = payload.model;
    if (
      !keys(m, [
        "version",
        "states",
        "symbols",
        "initial",
        "transition",
        "emission",
      ]) ||
      m.version !== "chainlab_rational_v1"
    )
      throw Error("Invalid rational model schema");
    for (const [field, cap] of [
      ["states", 4],
      ["symbols", 8],
    ]) {
      const xs = m[field];
      if (
        !Array.isArray(xs) ||
        xs.length < 1 ||
        xs.length > cap ||
        new Set(xs).size !== xs.length ||
        xs.some(
          (x) =>
            typeof x !== "string" || [...x].length < 1 || [...x].length > 40,
        )
      )
        throw Error("Invalid state/symbol names");
    }
    const n = m.states.length;
    function row(xs, k) {
      if (!Array.isArray(xs) || xs.length !== k)
        throw Error("Probability row has wrong shape");
      const out = xs.map((s) => {
        if (
          typeof s !== "string" ||
          s.length > 80 ||
          !/^(?:0|[1-9][0-9]*)(?:\/[1-9][0-9]*)?$/.test(s)
        )
          throw Error(
            "Probabilities must be unsigned integer/fraction strings",
          );
        const [num, den = "1"] = s.split("/"),
          q = new R(BigInt(num), BigInt(den));
        if (
          q.n.toString(2).length > 128 ||
          q.d.toString(2).length > 128 ||
          q.cmp(one) > 0
        )
          throw Error("Probability outside[0,1] or128-bit input limit");
        return q;
      });
      if (sum(out).cmp(one) !== 0)
        throw Error("Each probability row must sum to exactly1");
      return out;
    }
    const model = {
      version: m.version,
      states: m.states.slice(),
      symbols: m.symbols.slice(),
      initial: row(m.initial, n),
    };
    for (const [k, w] of [
      ["transition", n],
      ["emission", m.symbols.length],
    ]) {
      if (!Array.isArray(m[k]) || m[k].length !== n)
        throw Error("One " + k + " row required per state");
      model[k] = m[k].map((xs) => row(xs, w));
    }
    const seq = payload.sequence;
    if (
      !Array.isArray(seq) ||
      seq.length < 1 ||
      seq.length > 32 ||
      seq.some((s) => typeof s !== "string" || !m.symbols.includes(s))
    )
      throw Error("Sequence must contain1..32 known symbol names");
    return [model, seq.map((s) => m.symbols.indexOf(s))];
  }
  function logInteger(n) {
    const s = n.toString();
    return (
      Math.log(Number(s.slice(0, 15))) +
      (s.length - Math.min(15, s.length)) * Math.LN10
    );
  }
  function log(q) {
    if (!q.n) return null;
    const v = Number(q.n) / Number(q.d);
    return v > 0 && Number.isFinite(v)
      ? Math.log(v)
      : logInteger(q.n) - logInteger(q.d);
  }
  function number(q) {
    const v = Number(q.n) / Number(q.d);
    return Number.isFinite(v) ? v : Math.exp(log(q));
  }
  const entropy = (row) =>
    row.reduce((s, q) => s + (q.n ? -number(q) * log(q) : 0), 0);
  function pack(x) {
    if (x instanceof R) return x.text();
    if (typeof x === "bigint") return x.toString();
    if (Array.isArray(x)) return x.map(pack);
    if (x && typeof x === "object")
      return Object.fromEntries(
        Object.entries(x).map(([k, v]) => [k, pack(v)]),
      );
    return x;
  }
  const lexical = (a, b) => {
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return a[i] - b[i];
    }
    return 0;
  };
  const best = (xs) =>
    xs.sort((a, b) => -a.score.cmp(b.score) || lexical(a.path, b.path))[0] ||
    null;
  function audit(payload) {
    const [m, obs] = parse(payload),
      n = m.states.length,
      T = obs.length,
      a = m.transition,
      e = m.emission,
      p = m.initial;
    // Reverse suffix dynamic programming is computed first in this implementation.
    const b = obs.map(() => range(n).map(() => one));
    for (let t = T - 2; t >= 0; t--)
      b[t] = range(n).map((i) =>
        sum(
          range(n).map((j) => a[i][j].mul(e[j][obs[t + 1]]).mul(b[t + 1][j])),
        ),
      );
    const f = [p.map((v, i) => v.mul(e[i][obs[0]]))];
    for (let t = 1; t < T; t++)
      f.push(
        range(n).map((j) =>
          sum(range(n).map((i) => f[t - 1][i].mul(a[i][j]))).mul(e[j][obs[t]]),
        ),
      );
    const z = sum(f[T - 1]);
    if (!z.n)
      throw Error(
        "Observation sequence has zero probability; no posterior exists",
      );
    const g = f.map((row, t) => row.map((v, i) => v.mul(b[t][i]).div(z)));
    const edge = range(T - 1).map((t) =>
      range(n).map((i) =>
        range(n).map((j) =>
          f[t][i]
            .mul(a[i][j])
            .mul(e[j][obs[t + 1]])
            .mul(b[t + 1][j])
            .div(z),
        ),
      ),
    );
    const conditional = edge.map((rows, t) =>
      rows.map((row, i) => row.map((v) => (g[t][i].n ? v.div(g[t][i]) : null))),
    );
    const ent = g.map(entropy),
      cond = conditional.map((rows, t) =>
        rows.reduce(
          (s, row, i) =>
            s + number(g[t][i]) * entropy(row.filter((v) => v !== null)),
          0,
        ),
      );
    const mi = edge.map((rows, t) =>
      rows.reduce(
        (s, row, i) =>
          s +
          row.reduce(
            (v, q, j) =>
              v +
              (q.n
                ? number(q) * (log(q) - log(g[t][i]) - log(g[t + 1][j]))
                : 0),
            0,
          ),
        0,
      ),
    );
    const trellises = {},
      ds = {};
    for (const name of ["map", "supported_matches", "supported_product"]) {
      const rows = [];
      for (let t = 0; t < T; t++) {
        const cells = [];
        for (let j = 0; j < n; j++) {
          const candidates = [];
          if (!t && p[j].n && e[j][obs[0]].n)
            candidates.push({
              score: name === "map" ? p[j].mul(e[j][obs[0]]) : g[0][j],
              path: [j],
            });
          else if (t && e[j][obs[t]].n)
            for (let i = 0; i < n; i++) {
              const prev = rows[t - 1][i];
              if (prev && a[i][j].n)
                candidates.push({
                  score:
                    name === "supported_matches"
                      ? prev.score.add(g[t][j])
                      : prev.score.mul(
                          name === "map" ? a[i][j].mul(e[j][obs[t]]) : g[t][j],
                        ),
                  path: prev.path.concat(j),
                });
            }
          cells.push(best(candidates));
        }
        rows.push(cells);
      }
      const win = best(rows[T - 1].filter(Boolean));
      trellises[name] = rows;
      ds[name] = { objective: win.score, indices: win.path };
    }
    const modes = g.map(
      (row) => best(row.map((v, i) => ({ score: v, path: [i] }))).path[0],
    );
    ds.marginal_modes = {
      objective: sum(modes.map((i, t) => g[t][i])),
      indices: modes,
    };
    for (const d of Object.values(ds)) {
      const path = d.indices;
      let joint = p[path[0]].mul(e[path[0]][obs[0]]);
      for (let t = 1; t < T; t++)
        joint = joint.mul(a[path[t - 1]][path[t]]).mul(e[path[t]][obs[t]]);
      const matches = sum(path.map((i, t) => g[t][i])),
        post = joint.div(z);
      Object.assign(d, {
        path: path.map((i) => m.states[i]),
        joint,
        posterior: post,
        log_posterior: log(post),
        supported: !!joint.n,
        expected_matches: matches,
        expected_hamming: new R(BigInt(T)).sub(matches),
        marginal_product: prod(path.map((i, t) => g[t][i])),
      });
    }
    const counts = [range(n).map((i) => (p[i].n && e[i][obs[0]].n ? 1n : 0n))];
    for (let t = 1; t < T; t++)
      counts.push(
        range(n).map((j) =>
          e[j][obs[t]].n
            ? range(n).reduce(
                (s, i) => s + (a[i][j].n ? counts[t - 1][i] : 0n),
                0n,
              )
            : 0n,
        ),
      );
    const Hp = ent[0] + cond.reduce((a, b) => a + b, 0),
      Hs = ent.reduce((a, b) => a + b, 0);
    return pack({
      version: "chainlab_path_report_v1",
      input: { model: m, sequence: payload.sequence.slice() },
      forward: f,
      backward: b,
      evidence: z,
      log_evidence: log(z),
      state: g,
      edge,
      conditional,
      support_counts: counts,
      supported_paths: counts[T - 1].reduce((a, b) => a + b, 0n),
      decoders: ds,
      trellises,
      entropy: {
        state: ent,
        conditional: cond,
        edge_mutual_information: mi,
        path: Hp,
        sum_state: Hs,
        dependence_gap: Hs - Hp,
        sum_edge_mutual_information: mi.reduce((a, b) => a + b, 0),
      },
      limits: {
        states: 4,
        symbols: 8,
        length: 32,
        input_bits: 128,
        work_bits: 8192,
        entropy:
          "Approximate binary64 nats from exact probabilities; model-conditional, not calibrated uncertainty.",
      },
      tie_rule:
        "Lexicographically first whole index path among exact objective ties. Marginal modes keep first index independently.",
    });
  }
  const api = { audit, strictParse };
  if (typeof module !== "undefined") module.exports = api;
  root.PATH_ENGINE = api;
})(globalThis);
