"use client";

import { useEffect, useMemo, useState } from "react";
import EquityChart from "@/components/EquityChart";
import {
  getStrategies,
  runBacktest,
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

export default function Page() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [params, setParams] = useState<Record<string, number>>({});
  const [tickers, setTickers] = useState("SPY");
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

  const selected = useMemo(
    () => strategies.find((s) => s.id === selectedId),
    [strategies, selectedId],
  );

  useEffect(() => {
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
    setTickers(s.default_tickers);
    setResult(null);
    setError(null);
  }

  async function run() {
    if (!selected) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runBacktest({
        strategy_id: selected.id,
        params,
        tickers,
        start,
        end,
        split,
        cost_bps: costBps,
        fast,
        n_folds: folds,
      });
      setResult(res);
    } catch (e: any) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="wrap">
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

          <h2 style={{ marginTop: 20 }}>Universe &amp; period</h2>
          <label className="field">
            <span>Tickers <small>(comma-separated)</small></span>
            <input value={tickers} onChange={(e) => setTickers(e.target.value)} />
          </label>
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

          <button className="run" onClick={run} disabled={loading || !selected}>
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

function Tile({ k, v }: { k: string; v: string }) {
  return (
    <div className="tile">
      <div className="k">{k}</div>
      <div className="v">{v}</div>
    </div>
  );
}
