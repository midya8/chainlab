"use strict";
(function () {
  const $ = (id) => document.getElementById(id),
    make = (tag, text) => {
      const e = document.createElement(tag);
      if (text !== undefined) e.textContent = text;
      return e;
    };
  let report = null,
    url = null,
    selected = "";
  const labels = {
    map: "MAP whole path",
    supported_matches: "Supported expected matches",
    supported_product: "Supported product marginals",
    marginal_modes: "Unconstrained marginal modes",
  };
  const names = {
    "one-state": "One possible state",
    independent: "Independent: same marginals, more paths",
    locked: "Locked: same marginals, less uncertainty",
    "forbidden-modes": "Marginal modes violate transition support",
    "emission-support": "Emission support, not just transitions",
    "zero-occupancy": "Unreachable conditional row",
    "canonical-tie": "Whole-path lexicographic ties",
    "large-support": "4³² possible paths, no enumeration",
    "rare-evidence": "Nonzero evidence below binary64 range",
    "three-objectives": "Three objectives, three different paths",
    "millionth-rounded-trained": "Rounded original trained model",
    "millionth-rounded-initial": "Rounded original initial model",
    "millionth-rounded-reference": "Rounded original generating reference",
  };
  for (const c of PATH_DATA) {
    const o = make("option", names[c.name]);
    o.value = c.name;
    $("control").append(o);
  }
  $("control").value = "three-objectives";
  function approx(s) {
    if (s === null) return "undefined";
    const parts = s.split("/"),
      a = BigInt(parts[0]),
      b = BigInt(parts[1] || 1);
    let v = Number(a) / Number(b);
    if (!Number.isFinite(v)) {
      const log = (x) => {
        const t = x.toString();
        return (
          Math.log(Number(t.slice(0, 15))) +
          (t.length - Math.min(15, t.length)) * Math.LN10
        );
      };
      v = Math.exp(log(a) - log(b));
    }
    return v === 0 && a !== 0n
      ? "below display range"
      : Number(v).toPrecision(7);
  }
  function exactButton(q, label) {
    const b = make("button", q === null ? "undefined" : approx(q));
    b.className = "cell";
    b.type = "button";
    b.dataset.exact = q === null ? "undefined" : q;
    b.setAttribute("aria-label", label + "; select exact value");
    b.onclick = () => {
      $("exact").value =
        label +
        "\n" +
        (q === null
          ? "Undefined: conditioning state has zero posterior mass."
          : q);
      selected = label;
    };
    return b;
  }
  function table(id, headers, rows) {
    const t = make("table"),
      head = make("thead"),
      tr = make("tr");
    for (const h of headers) {
      const th = make("th", h);
      th.scope = "col";
      tr.append(th);
    }
    head.append(tr);
    t.append(head);
    const body = make("tbody");
    for (const row of rows) {
      const tr = make("tr");
      for (const [column, x] of row.entries()) {
        const cell = make("td");
        cell.dataset.label = headers[column];
        cell.append(x instanceof Node ? x : document.createTextNode(String(x)));
        tr.append(cell);
      }
      body.append(tr);
    }
    t.append(body);
    $(id).replaceChildren(t);
  }
  function detailPath(d) {
    const e = make("div");
    e.append(make("code", d.path.join(" → ")));
    e.append(
      make(
        "small",
        d.supported
          ? "Supported by initial, transition and emission probabilities"
          : "Impossible under this model and evidence",
      ),
    );
    if (!d.supported) e.className = "unsupported";
    return e;
  }
  function edge() {
    if (!report) return;
    const t = Number($("edge-position").value),
      m = report.input.model;
    if (!report.edge.length) {
      $("edges").textContent = "One observation has no adjacent-state edge.";
      $("conditionals").replaceChildren();
      return;
    }
    table(
      "edges",
      ["Joint posterior edge at " + t, ...m.states],
      report.edge[t].map((row, i) => [
        m.states[i],
        ...row.map((q, j) =>
          exactButton(
            q,
            `P(S${t}=${m.states[i]}, S${t + 1}=${m.states[j]} | observations)`,
          ),
        ),
      ]),
    );
    table(
      "conditionals",
      ["Next state given current", ...m.states],
      report.conditional[t].map((row, i) => [
        m.states[i],
        ...row.map((q, j) =>
          exactButton(
            q,
            `P(S${t + 1}=${m.states[j]} | S${t}=${m.states[i]}, observations)`,
          ),
        ),
      ]),
    );
  }
  function trellis() {
    if (!report) return;
    const name = $("objective").value,
      m = report.input.model;
    table(
      "trellis",
      ["Time", ...m.states],
      report.trellises[name].map((row, t) => [
        t,
        ...row.map((v, j) => {
          if (v === null) return "No supported prefix";
          const e = make("div");
          e.append(
            exactButton(
              v.score,
              `${labels[name]} prefix score at time${t}, state${m.states[j]}`,
            ),
          );
          e.append(make("small", v.path.map((i) => m.states[i]).join(" → ")));
          return e;
        }),
      ]),
    );
  }
  function invalidate(message) {
    report = null;
    selected = "";
    if (url) {
      URL.revokeObjectURL(url);
      url = null;
    }
    $("error").textContent = message;
    $("exact").value = "";
    $("report").value = "";
    $("copy-status").textContent = "";
    for (const id of [
      "decisions",
      "states",
      "edges",
      "conditionals",
      "entropy-table",
      "trellis",
    ])
      $(id).replaceChildren();
    for (const id of ["path-entropy", "sum-entropy", "gap"])
      $(id).textContent = "-";
    $("support").textContent = "";
    $("evidence").textContent = "";
    $("edge-position").replaceChildren();
    $("copy").disabled = true;
    $("download").disabled = true;
  }
  function update() {
    invalidate("");
    try {
      report = PATH_ENGINE.audit(PATH_ENGINE.strictParse($("input").value));
    } catch (err) {
      invalidate(err.message);
      return;
    }
    const r = report,
      m = r.input.model;
    $("copy").disabled = false;
    $("download").disabled = false;
    table(
      "decisions",
      [
        "Decision rule",
        "Selected path",
        "P(whole path | observations)",
        "Expected wrong positions",
      ],
      Object.entries(r.decoders).map(([name, d]) => [
        labels[name],
        detailPath(d),
        exactButton(d.posterior, labels[name] + " whole-path posterior"),
        exactButton(
          d.expected_hamming,
          labels[name] + " expected Hamming loss",
        ),
      ]),
    );
    $("path-entropy").textContent = r.entropy.path.toFixed(7);
    $("sum-entropy").textContent = r.entropy.sum_state.toFixed(7);
    $("gap").textContent = r.entropy.dependence_gap.toFixed(7);
    $("support").textContent =
      r.supported_paths +
      " supported full paths. Counted exactly by dynamic programming.";
    $("evidence").textContent =
      "log P(observations) = " +
      r.log_evidence.toPrecision(10) +
      " nats. Evidence probability remains exact in the archive.";
    table(
      "entropy-table",
      [
        "Time",
        "State entropy (nats)",
        "Next-state conditional entropy",
        "Adjacent-state mutual information",
      ],
      r.entropy.state.map((h, t) => [
        t,
        h.toFixed(9),
        r.entropy.conditional[t] === undefined
          ? "last position"
          : r.entropy.conditional[t].toFixed(9),
        r.entropy.edge_mutual_information[t] === undefined
          ? "last position"
          : r.entropy.edge_mutual_information[t].toFixed(9),
      ]),
    );
    table(
      "states",
      ["Time / observed symbol", ...m.states],
      r.state.map((row, t) => [
        t + " / " + r.input.sequence[t],
        ...row.map((q, i) =>
          exactButton(q, `P(S${t}=${m.states[i]} | observations)`),
        ),
      ]),
    );
    for (let t = 0; t < r.edge.length; t++) {
      const o = make("option", t + " → " + (t + 1));
      o.value = String(t);
      $("edge-position").append(o);
    }
    $("edge-position").disabled = !r.edge.length;
    edge();
    trellis();
    if ($("report-details").open)
      $("report").value = JSON.stringify(report, null, 2);
  }
  function restore() {
    const c = PATH_DATA.find((c) => c.name === $("control").value);
    $("input").value = JSON.stringify(c.input, null, 2);
    $("control-note").textContent = c.name.startsWith("millionth-rounded-")
      ? "Each original saved probability is rounded to the nearest millionth, then its row is exactly normalized. This is a different rational model, not the unchanged floating model or a new trained fit. All 12 observations are from the earlier displayed fixture window."
      : c.name === "three-objectives"
        ? "Adverse example selected by the first qualifying result in a fixed-seed search (seed 314159, zero-based candidate 1904). It illustrates disagreement, not typical behavior or a benchmark ranking."
        : "A synthetic arithmetic control. Its states are supplied labels, not inferred semantic categories.";
    update();
  }
  $("control").addEventListener("change", restore);
  $("restore").addEventListener("click", restore);
  $("input").addEventListener("input", update);
  $("edge-position").addEventListener("change", edge);
  $("objective").addEventListener("change", trellis);
  $("report-details").addEventListener("toggle", () => {
    if ($("report-details").open && report)
      $("report").value = JSON.stringify(report, null, 2);
  });
  $("copy").onclick = async () => {
    if (!report) {
      $("copy-status").textContent = "Enter a valid model first.";
      return;
    }
    try {
      await navigator.clipboard.writeText(JSON.stringify(report, null, 2));
      $("copy-status").textContent = "Copied complete input and report.";
    } catch {
      $("copy-status").textContent =
        "Copy was blocked. Download the archive instead.";
    }
  };
  $("download").onclick = () => {
    if (!report) {
      $("copy-status").textContent = "Enter a valid model first.";
      return;
    }
    if (url) URL.revokeObjectURL(url);
    url = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2) + "\n"], {
        type: "application/json",
      }),
    );
    const a = make("a");
    a.href = url;
    a.download = "chainlab-path-report.json";
    document.body.append(a);
    a.click();
    a.remove();
    $("copy-status").textContent = "Downloaded complete input and report.";
  };
  globalThis.PATH_APP = {
    getReport: () => report,
    update,
    restore,
    getSelection: () => selected,
  };
  restore();
})();
