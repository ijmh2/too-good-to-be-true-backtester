"use client";

import { useEffect, useMemo, useState } from "react";
import EquityChart from "@/components/EquityChart";
import {
  API_URL,
  getConfig,
  getExampleCsv,
  getStrategies,
  runBacktest,
  type RunRequest,
  type RunResult,
  type Strategy,
} from "@/lib/api";

const VERDICT_CLASS: Record<string, string> = {
  "likely real edge": "good",
  inconclusive: "warning",
  "likely overfit": "critical",
};
const VERDICT_ICON: Record<string, string> = {
  "likely real edge": "✅",
  inconclusive: "⚠️",
  "likely overfit": "🚩",
};

function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

// A working example, editable in place — select-all and paste your own strategy over it.
const STRATEGY_CODE_TEMPLATE = `import pandas as pd
from tgtbt.strategies.base import Strategy

class MyStrategy(Strategy):
    """Example: hold the asset only when it's above its N-day moving average."""

    def __init__(self, window: int = 100):
        self.window = window
        self.name = f"my_strategy(window={window})"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        px = prices.iloc[:, 0]
        ma = px.rolling(self.window, min_periods=self.window).mean()
        w = (px > ma).astype(float).fillna(0.0)  # backward-looking only -> no look-ahead
        return w.to_frame(prices.columns[0]).reindex(columns=prices.columns).fillna(0.0)


def make(**params):
    return MyStrategy(**params)


# The UI reads exactly these three module-level names.
STRATEGY = MyStrategy(window=100)
FACTORY = make
GRID = {"window": [20, 50, 100, 150, 200]}
`;

export default function Page() {
  const [allowUploads, setAllowUploads] = useState(false);
  const [apiChecked, setApiChecked] = useState(false);
  const [apiOk, setApiOk] = useState(false);

  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [params, setParams] = useState<Record<string, number>>({});

  const [strategySource, setStrategySource] = useState<"builtin" | "upload">("builtin");
  const [strategyCode, setStrategyCode] = useState<string>(STRATEGY_CODE_TEMPLATE);
  const [strategyFileName, setStrategyFileName] = useState<string | null>(null);
  const [strategyCodeErr, setStrategyCodeErr] = useState<string | null>(null);

  const [dataSource, setDataSource] = useState<"tickers" | "upload">("tickers");
  const [tickers, setTickers] = useState("SPY");
  const [priceCsv, setPriceCsv] = useState<string>("");
  const [priceFileName, setPriceFileName] = useState<string | null>(null);
  const [priceCsvErr, setPriceCsvErr] = useState<string | null>(null);

  const [start, setStart] = useState("2010-01-01");
  const [end, setEnd] = useState("2024-12-31");
  const [split, setSplit] = useState("2021-12-31");
  const [costBps, setCostBps] = useState(5);
  const [fast, setFast] = useState(true);
  const [folds, setFolds] = useState(5);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [elapsedSec, setElapsedSec] = useState<number | null>(null);
  const [lastRequest, setLastRequest] = useState<RunRequest | null>(null);

  const selected = useMemo(
    () => strategies.find((s) => s.id === selectedId),
    [strategies, selectedId],
  );

  useEffect(() => {
    getConfig()
      .then((c) => {
        setAllowUploads(c.allow_uploads);
        setApiOk(true);
      })
      .catch(() => setAllowUploads(false)) // treat an unreachable /config as "uploads off"
      .finally(() => setApiChecked(true));
    getStrategies()
      .then((list) => {
        setStrategies(list);
        if (list.length) selectStrategy(list[0]);
      })
      .catch((e) => setLoadErr(String(e.message || e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function selectStrategy(s: Strategy) {
    setSelectedId(s.id);
    setParams(Object.fromEntries(s.params.map((p) => [p.name, p.default])));
    if (dataSource === "tickers") setTickers(s.default_tickers);
    setResult(null);
    setError(null);
  }

  async function onStrategyFile(file: File | undefined) {
    setStrategyCodeErr(null);
    if (!file) return;
    try {
      const text = await readFileAsText(file);
      setStrategyCode(text);
      setStrategyFileName(file.name);
    } catch (e: any) {
      setStrategyCodeErr(String(e.message || e));
    }
  }

  async function onPriceFile(file: File | undefined) {
    setPriceCsvErr(null);
    if (!file) return;
    try {
      const text = await readFileAsText(file);
      setPriceCsv(text);
      setPriceFileName(file.name);
    } catch (e: any) {
      setPriceCsvErr(String(e.message || e));
    }
  }

  async function downloadExample() {
    try {
      const { filename, content } = await getExampleCsv();
      const blob = new Blob([content], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      setPriceCsvErr(String(e.message || e));
    }
  }

  async function loadExampleIntoBox() {
    setPriceCsvErr(null);
    try {
      const { content } = await getExampleCsv();
      setPriceCsv(content);
      setPriceFileName(null);
    } catch (e: any) {
      setPriceCsvErr(String(e.message || e));
    }
  }

  const strategyReady = strategySource === "builtin" ? !!selected : !!strategyCode;
  const dataReady = dataSource === "tickers" ? tickers.trim().length > 0 : !!priceCsv;

  async function run() {
    if (!strategyReady || !dataReady) return;
    setLoading(true);
    setError(null);
    setElapsedSec(null);
    const body: RunRequest = {
      ...(strategySource === "builtin"
        ? { strategy_id: selected!.id, params }
        : { strategy_code: strategyCode }),
      ...(dataSource === "tickers" ? { tickers } : { price_csv: priceCsv }),
      start,
      end,
      split,
      cost_bps: costBps,
      fast,
      n_folds: folds,
    };
    setLastRequest(body);
    const t0 = performance.now();
    try {
      const res = await runBacktest(body);
      setResult(res);
      setElapsedSec((performance.now() - t0) / 1000);
    } catch (e: any) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="wrap">
      <div className="statusbar">
        <span className={`dot ${apiChecked ? (apiOk ? "ok" : "down") : ""}`} />
        <span>api: {apiOk ? "reachable" : apiChecked ? "unreachable" : "checking…"}</span>
        <span className="sep">|</span>
        <code>{API_URL}</code>
        <span className="sep">|</span>
        <span className={`pill ${allowUploads ? "upload" : ""}`}>
          {allowUploads ? "LOCAL UPLOAD MODE" : "PUBLIC MODE"}
        </span>
      </div>

      <header className="site">
        <h1>🚩 too-good-to-be-true-backtester</h1>
        <p>
          Pick a strategy and run the full overfit gauntlet — out-of-sample walk-forward,
          permutation nulls, Monte-Carlo, the Deflated Sharpe and CSCV probability of backtest
          overfitting — for one honest verdict.{" "}
          <a href="https://github.com/ijmh2/too-good-to-be-true-backtester" target="_blank" rel="noreferrer">
            Source →
          </a>
        </p>
      </header>

      {loadErr && (
        <div className="err" style={{ marginBottom: 16 }}>
          Couldn&apos;t reach the API ({loadErr}). Is the backend running at{" "}
          <code>{process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}</code>?
        </div>
      )}

      <div className="layout">
        {/* ---- Controls ---- */}
        <div className="card">
          <h2>Strategy</h2>
          {allowUploads && (
            <label className="field">
              <span>Source</span>
              <select
                value={strategySource}
                onChange={(e) => {
                  setStrategySource(e.target.value as "builtin" | "upload");
                  setResult(null);
                }}
              >
                <option value="builtin">Built-in</option>
                <option value="upload">Upload your own code</option>
              </select>
            </label>
          )}

          {strategySource === "builtin" ? (
            <>
              <label className="field">
                <span>Strategy</span>
                <select
                  value={selectedId}
                  onChange={(e) => {
                    const s = strategies.find((x) => x.id === e.target.value);
                    if (s) selectStrategy(s);
                  }}
                >
                  {strategies.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name}
                    </option>
                  ))}
                </select>
              </label>
              {selected && (
                <p className="caption" style={{ marginTop: -6, marginBottom: 14 }}>
                  {selected.description}
                </p>
              )}
              {selected?.params.map((p) => (
                <label className="field" key={p.name}>
                  <span>
                    {p.label} <small>({params[p.name]})</small>
                  </span>
                  <input
                    type="number"
                    value={params[p.name] ?? p.default}
                    min={p.min}
                    max={p.max}
                    step={p.step}
                    onChange={(e) =>
                      setParams({ ...params, [p.name]: Number(e.target.value) })
                    }
                  />
                </label>
              ))}
            </>
          ) : (
            <>
              <p className="caption" style={{ marginTop: -6 }}>
                Paste code defining module-level <code>STRATEGY</code>, <code>FACTORY</code>,{" "}
                <code>GRID</code> — edit the example below, or clear it and paste your own (see{" "}
                <a
                  href="https://github.com/ijmh2/too-good-to-be-true-backtester/blob/main/app/strategy_template.py"
                  target="_blank"
                  rel="noreferrer"
                >
                  strategy_template.py
                </a>
                ). Runs local-only, in-process — only paste code you trust.
              </p>
              <label className="field">
                <span>Strategy code</span>
                <textarea
                  className="code-textarea"
                  spellCheck={false}
                  value={strategyCode}
                  onChange={(e) => setStrategyCode(e.target.value)}
                />
              </label>
              <label className="field">
                <span>…or load from a file</span>
                <input
                  type="file"
                  accept=".py"
                  onChange={(e) => onStrategyFile(e.target.files?.[0])}
                />
              </label>
              {strategyFileName && !strategyCodeErr && (
                <p className="caption">Loaded {strategyFileName}</p>
              )}
              {strategyCodeErr && <div className="err">{strategyCodeErr}</div>}
            </>
          )}

          <h2 style={{ marginTop: 20 }}>Universe &amp; period</h2>
          {allowUploads && (
            <label className="field">
              <span>Data source</span>
              <select
                value={dataSource}
                onChange={(e) => {
                  setDataSource(e.target.value as "tickers" | "upload");
                  setResult(null);
                }}
              >
                <option value="tickers">Tickers (Yahoo Finance)</option>
                <option value="upload">Upload CSV</option>
              </select>
            </label>
          )}

          {dataSource === "tickers" ? (
            <label className="field">
              <span>Tickers <small>(comma-separated)</small></span>
              <input value={tickers} onChange={(e) => setTickers(e.target.value)} />
            </label>
          ) : (
            <>
              <p className="caption" style={{ marginTop: -6 }}>
                A date column (<code>date</code>/<code>datetime</code>/<code>timestamp</code>/
                <code>time</code>, or the first column) plus one numeric column per asset.
                Column headers become the asset names.
              </p>
              <label className="field">
                <span>Price data (CSV)</span>
                <textarea
                  className="code-textarea small"
                  spellCheck={false}
                  value={priceCsv}
                  onChange={(e) => setPriceCsv(e.target.value)}
                  placeholder={"date,AssetA,AssetB\n2020-01-01,100.0,50.0\n2020-01-02,101.2,49.8\n..."}
                />
              </label>
              <label className="field">
                <span>…or load from a file</span>
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => onPriceFile(e.target.files?.[0])}
                />
              </label>
              {priceFileName && !priceCsvErr && (
                <p className="caption">Loaded {priceFileName}</p>
              )}
              {priceCsvErr && <div className="err">{priceCsvErr}</div>}
              <div className="row" style={{ marginBottom: 14 }}>
                <button className="ghost" type="button" onClick={loadExampleIntoBox}>
                  Paste example
                </button>
                <button className="ghost" type="button" onClick={downloadExample}>
                  Download as file
                </button>
              </div>
            </>
          )}

          <div className="row">
            <label className="field">
              <span>Start</span>
              <input value={start} onChange={(e) => setStart(e.target.value)} />
            </label>
            <label className="field">
              <span>End</span>
              <input value={end} onChange={(e) => setEnd(e.target.value)} />
            </label>
          </div>
          <label className="field">
            <span>In/out-of-sample split</span>
            <input value={split} onChange={(e) => setSplit(e.target.value)} />
          </label>

          <h2 style={{ marginTop: 20 }}>Settings</h2>
          <div className="row">
            <label className="field">
              <span>Cost (bps)</span>
              <input
                type="number"
                value={costBps}
                min={0}
                max={25}
                step={0.5}
                onChange={(e) => setCostBps(Number(e.target.value))}
              />
            </label>
            <label className="field">
              <span>WF folds</span>
              <input
                type="number"
                value={folds}
                min={3}
                max={8}
                step={1}
                onChange={(e) => setFolds(Number(e.target.value))}
              />
            </label>
          </div>
          <label className="field" style={{ display: "flex", alignItems: "center" }}>
            <input type="checkbox" checked={fast} onChange={(e) => setFast(e.target.checked)} />
            <span style={{ margin: 0 }}>Fast mode (fewer resamples)</span>
          </label>

          <button className="run" onClick={run} disabled={loading || !strategyReady || !dataReady}>
            {loading ? "Running the gauntlet…" : "Run the gauntlet"}
          </button>
        </div>

        {/* ---- Results ---- */}
        <div className="card">
          <h2>Result</h2>
          {!result && !loading && !error && (
            <div className="spinner">Configure a strategy and run it to see the verdict.</div>
          )}
          {loading && (
            <div className="spinner">
              Running walk-forward, permutation, Monte-Carlo and CSCV… (~10–20s)
            </div>
          )}
          {error && <div className="err">{error}</div>}

          {result && !loading && (
            <div className="stack">
              <div>
                <div className="runmeta">
                  <span>data: {result.data_source}</span>
                  {elapsedSec != null && <span>completed in {elapsedSec.toFixed(1)}s</span>}
                  <span>n_folds: {folds}</span>
                  <span>cost: {costBps}bps</span>
                </div>
                <div className={`verdict ${VERDICT_CLASS[result.verdict] || ""}`}>
                  <span className="badge">
                    {VERDICT_ICON[result.verdict]} {result.verdict.toUpperCase()}
                  </span>
                </div>
                <div className="chips">
                  {Object.entries(result.flags).map(([k, v]) => (
                    <span className="chip" key={k}>
                      <span className={v ? "ok" : "no"}>{v ? "✓" : "✗"}</span>{" "}
                      {k.replace(/_/g, " ")}
                    </span>
                  ))}
                </div>
              </div>

              <div className="tiles">
                <Tile k="Full Sharpe" v={fmt(result.tiles.full_sharpe)} />
                <Tile k="Walk-fwd OOS Sharpe" v={fmt(result.tiles.wf_oos_sharpe)} />
                <Tile k="Permutation p" v={fmt(result.tiles.perm_p, 3)} />
                <Tile k="Deflated Sharpe" v={fmt(result.tiles.dsr)} />
                <Tile k="Overfit prob (PBO)" v={fmt(result.tiles.pbo)} />
                <Tile k="Max drawdown" v={fmt(result.tiles.max_drawdown_pct, 1, "%")} />
              </div>

              <div>
                <h2>Equity curve (net of costs)</h2>
                <EquityChart result={result} />
              </div>

              <div>
                <h2>Full scorecard</h2>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img className="scorecard-img" src={result.scorecard_png} alt="overfit scorecard" />
              </div>

              <p className="caption">Data: {result.data_source}</p>

              {lastRequest && (
                <details className="payload">
                  <summary>Request payload (POST {API_URL}/run)</summary>
                  <pre>{JSON.stringify(redactPayload(lastRequest), null, 2)}</pre>
                </details>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function fmt(v: number | null, nd = 2, suffix = ""): string {
  return v == null ? "—" : v.toFixed(nd) + suffix;
}

// Long code/CSV fields would otherwise dump kilobytes inline; show a preview + length instead.
function redactPayload(req: RunRequest): Record<string, unknown> {
  const truncate = (s: string, n = 200) =>
    s.length > n ? `${s.slice(0, n)}… (${s.length} chars total)` : s;
  return {
    ...req,
    ...(req.strategy_code ? { strategy_code: truncate(req.strategy_code) } : {}),
    ...(req.price_csv ? { price_csv: truncate(req.price_csv) } : {}),
  };
}

function Tile({ k, v }: { k: string; v: string }) {
  return (
    <div className="tile">
      <div className="k">{k}</div>
      <div className="v">{v}</div>
    </div>
  );
}
