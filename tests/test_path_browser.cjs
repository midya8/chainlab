"use strict";
const test = require("node:test"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  os = require("node:os"),
  path = require("node:path"),
  { spawn, spawnSync } = require("node:child_process");
const { waitForDebugger } = require("./chrome_support.js"),
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
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
test(
  "actual Chrome: all13 full models, all probability/decision cells, physical editing, downloads and39 layouts",
  { timeout: 180000 },
  async () => {
    const profile = fs.mkdtempSync(
        path.join(os.tmpdir(), "chain-path-browser-"),
      ),
      downloads = path.join(profile, "downloads");
    fs.mkdirSync(downloads);
    const browser = spawn(
      process.env.CHROME_BIN || "google-chrome",
      [
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-extensions",
        "--disable-dev-shm-usage",
        "--remote-debugging-port=0",
        `--user-data-dir=${profile}`,
        "about:blank",
      ],
      { stdio: ["ignore", "ignore", "pipe"] },
    );
    let ws,
      startupError,
      diagnostics = "";
    browser.on("error", (e) => (startupError = e));
    browser.stderr.on(
      "data",
      (d) => (diagnostics = (diagnostics + d).slice(-4000)),
    );
    try {
      for (
        let i = 0;
        i < 1200 &&
        !startupError &&
        browser.exitCode === null &&
        !fs.existsSync(path.join(profile, "DevToolsActivePort"));
        i++
      )
        await delay(25);
      assert.ok(
        fs.existsSync(path.join(profile, "DevToolsActivePort")),
        `${startupError?.message || diagnostics}`,
      );
      const port = fs
        .readFileSync(path.join(profile, "DevToolsActivePort"), "utf8")
        .split("\n")[0];
      const tab = await waitForDebugger(`http://127.0.0.1:${port}/json/list`, {
        alive: () => browser.exitCode === null && !startupError,
      });
      ws = new WebSocket(tab.webSocketDebuggerUrl);
      await new Promise((res, rej) => {
        ws.onopen = res;
        ws.onerror = rej;
      });
      let seq = 0;
      const pending = new Map();
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data);
        if (pending.has(m.id)) {
          const { res, rej } = pending.get(m.id);
          pending.delete(m.id);
          m.error ? rej(Error(JSON.stringify(m.error))) : res(m.result);
        }
      };
      const call = (method, params = {}) =>
        new Promise((res, rej) => {
          const id = ++seq;
          pending.set(id, { res, rej });
          ws.send(JSON.stringify({ id, method, params }));
        });
      const js = async (expression) => {
        const r = await call("Runtime.evaluate", {
          expression,
          returnByValue: true,
          awaitPromise: true,
        });
        if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails));
        return r.result.value;
      };
      await call("Page.enable");
      await call("Page.navigate", {
        url:
          process.env.CHAIN_PATH_URL ||
          "file://" + path.resolve(__dirname, "../docs/paths.html"),
      });
      for (let i = 0; i < 500; i++) {
        if (await js("Boolean(globalThis.PATH_APP?.getReport())")) break;
        await delay(20);
      }
      assert.equal(
        await js(`document.getElementById('error').textContent`),
        "",
      );
      const key = async (key, code, which, modifiers = 0) => {
        await call("Input.dispatchKeyEvent", {
          type: "keyDown",
          key,
          code,
          windowsVirtualKeyCode: which,
          modifiers,
        });
        await call("Input.dispatchKeyEvent", {
          type: "keyUp",
          key,
          code,
          windowsVirtualKeyCode: which,
          modifiers,
        });
      };
      await js(`document.getElementById('control').focus()`);
      await key("ArrowDown", "ArrowDown", 40);
      assert.equal(
        await js(`document.getElementById('control').value`),
        "millionth-rounded-trained",
      );
      let stateCells = 0,
        edgeCells = 0,
        trellisCells = 0,
        layouts = 0;
      const near = (a, b) =>
        assert.ok(
          Number.isFinite(a) && Math.abs(a - b) < 1e-10,
          `${a} vs ${b}`,
        );
      for (const c of controls) {
        await js(
          `document.getElementById('control').value=${JSON.stringify(c.name)};document.getElementById('control').dispatchEvent(new Event('change'));`,
        );
        const r = await js("PATH_APP.getReport()");
        assert.deepEqual(r.state, c.report.state, c.name);
        assert.deepEqual(r.edge, c.report.edge, c.name);
        assert.deepEqual(
          Object.fromEntries(
            Object.entries(r.decoders).map(([k, v]) => [
              k,
              {
                path: v.path,
                posterior: v.posterior,
                expected_hamming: v.expected_hamming,
              },
            ]),
          ),
          Object.fromEntries(
            Object.entries(c.report.decoders).map(([k, v]) => [
              k,
              {
                path: v.path,
                posterior: v.posterior,
                expected_hamming: v.expected_hamming,
              },
            ]),
          ),
        );
        near(r.entropy.path, c.report.entropy.path);
        assert.deepEqual(
          await js(
            `Array.from(document.querySelectorAll('#states button')).map(b=>b.dataset.exact)`,
          ),
          c.report.state.flat(),
        );
        stateCells += c.report.state.flat().length;
        const clicked = await js(
          `Array.from(document.querySelectorAll('#states button')).every(b=>{b.click();return document.getElementById('exact').value.endsWith(b.dataset.exact)})`,
        );
        assert.ok(clicked);
        for (let t = 0; t < r.edge.length; t++) {
          await js(
            `document.getElementById('edge-position').value='${t}';document.getElementById('edge-position').dispatchEvent(new Event('change'))`,
          );
          const expected = [
            ...c.report.edge[t].flat(),
            ...c.report.conditional[t].flat(),
          ].map((q) => (q === null ? "undefined" : q));
          assert.deepEqual(
            await js(
              `Array.from(document.querySelectorAll('#edges button,#conditionals button')).map(b=>b.dataset.exact)`,
            ),
            expected,
          );
          edgeCells += expected.length;
          assert.ok(
            await js(
              `Array.from(document.querySelectorAll('#edges button,#conditionals button')).every(b=>{b.click();return b.dataset.exact==='undefined'?document.getElementById('exact').value.includes('Undefined'):document.getElementById('exact').value.endsWith(b.dataset.exact)})`,
            ),
          );
        }
        for (const name of ["map", "supported_matches", "supported_product"]) {
          await js(
            `document.getElementById('objective').value='${name}';document.getElementById('objective').dispatchEvent(new Event('change'))`,
          );
          const expected = c.report.trellises[name]
            .flat()
            .filter(Boolean)
            .map((v) => v.score);
          assert.deepEqual(
            await js(
              `Array.from(document.querySelectorAll('#trellis button')).map(b=>b.dataset.exact)`,
            ),
            expected,
          );
          trellisCells += expected.length;
          assert.ok(
            await js(
              `Array.from(document.querySelectorAll('#trellis button')).every(b=>{b.click();return document.getElementById('exact').value.endsWith(b.dataset.exact)})`,
            ),
          );
        }
        for (const width of [320, 390, 1280]) {
          await call("Emulation.setDeviceMetricsOverride", {
            width,
            height: 1000,
            deviceScaleFactor: 1,
            mobile: width < 700,
          });
          assert.ok(
            await js(`document.documentElement.scrollWidth<=innerWidth`),
            "page overflow " + c.name + width,
          );
          assert.ok(
            await js(
              `Array.from(document.querySelectorAll('button,select,summary,textarea')).every(e=>e.getBoundingClientRect().height===0||e.getBoundingClientRect().height>=44)`,
            ),
            "touch target " + c.name + width,
          );
          layouts++;
        }
      }
      await js(
        `document.getElementById('control').value='three-objectives';PATH_APP.restore();document.querySelector('.input-panel details').open=true;document.getElementById('input').focus()`,
      );
      await key("a", "KeyA", 65, 2);
      await call("Input.insertText", {
        text: '{"model":1,"model":2,"sequence":[]}',
      });
      assert.equal(await js("PATH_APP.getReport()"), null);
      assert.match(
        await js(`document.getElementById('error').textContent`),
        /Duplicate/,
      );
      assert.equal(
        await js(`document.getElementById('path-entropy').textContent`),
        "-",
      );
      assert.equal(
        await js(`document.getElementById('download').disabled`),
        true,
      );
      const edited = structuredClone(controls[0].input);
      edited.model.states = ["user-edited"];
      edited.sequence = ["y", "y"];
      await key("a", "KeyA", 65, 2);
      await call("Input.insertText", { text: JSON.stringify(edited) });
      assert.deepEqual(await js("PATH_APP.getReport().input"), edited);
      await js(
        `Object.defineProperty(navigator,'clipboard',{value:{writeText:async()=>{throw Error('denied')}},configurable:true});document.getElementById('copy').click()`,
      );
      await delay(20);
      assert.match(
        await js(`document.getElementById('copy-status').textContent`),
        /Copy was blocked/,
      );
      await js(
        `Object.defineProperty(navigator,'clipboard',{value:{writeText:async s=>globalThis.copied=s},configurable:true});document.getElementById('copy').click()`,
      );
      await delay(20);
      assert.deepEqual(await js("JSON.parse(copied).input"), edited);
      await call("Browser.setDownloadBehavior", {
        behavior: "allow",
        downloadPath: downloads,
      });
      for (const name of ["three-objectives", "rare-evidence"]) {
        await js(
          `document.getElementById('control').value='${name}';PATH_APP.restore();document.getElementById('download').click()`,
        );
        for (
          let i = 0;
          i < 200 &&
          !fs.existsSync(path.join(downloads, "chainlab-path-report.json"));
          i++
        )
          await delay(25);
        const file = path.join(downloads, "chainlab-path-report.json");
        assert.ok(fs.existsSync(file), "real Blob download");
        const out = spawnSync("python", ["-m", "chainlab.path_verify", file], {
          cwd: path.resolve(__dirname, ".."),
          encoding: "utf8",
        });
        assert.equal(out.status, 0, out.stderr);
        if (process.env.CHAIN_PATH_ARCHIVES) {
          fs.mkdirSync(process.env.CHAIN_PATH_ARCHIVES, { recursive: true });
          fs.copyFileSync(
            file,
            path.join(process.env.CHAIN_PATH_ARCHIVES, name + ".json"),
          );
        }
        fs.unlinkSync(file);
      }
      console.log(
        JSON.stringify({
          controls: controls.length,
          stateCells,
          edgeCells,
          trellisCells,
          layouts,
          completedNativeVerifiedDownloads: 2,
        }),
      );
      if (process.env.CHAIN_PATH_SCREENSHOTS) {
        fs.mkdirSync(process.env.CHAIN_PATH_SCREENSHOTS, { recursive: true });
        for (const name of ["three-objectives", "locked", "large-support"])
          for (const width of [390, 1280]) {
            await js(
              `document.getElementById('control').value='${name}';PATH_APP.restore();document.querySelector('.input-panel details').open=false;document.getElementById('report-details').open=false;document.activeElement.blur();scrollTo(0,0)`,
            );
            await call("Emulation.setDeviceMetricsOverride", {
              width,
              height: 1200,
              deviceScaleFactor: 1,
              mobile: width < 700,
            });
            for (const [part, top] of [
              ["top", 0],
              ["decision", 650],
              ["entropy", 1200],
            ]) {
              await js(`scrollTo(0,${top})`);
              const image = await call("Page.captureScreenshot", {
                format: "png",
              });
              fs.writeFileSync(
                path.join(
                  process.env.CHAIN_PATH_SCREENSHOTS,
                  name + "-" + width + "-" + part + ".png",
                ),
                Buffer.from(image.data, "base64"),
              );
            }
          }
      }
    } finally {
      if (ws) ws.close();
      if (browser.pid && browser.exitCode === null) {
        browser.kill("SIGTERM");
        await new Promise((resolve) => browser.once("exit", resolve));
      }
      await fs.promises.rm(profile, {
        recursive: true,
        force: true,
        maxRetries: 10,
        retryDelay: 200,
      });
    }
  },
);
