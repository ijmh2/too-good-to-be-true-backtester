// API client + shared types for the too-good-to-be-true-backtester backend.

export type Param = {
  name: string;
  label: string;
  type: "int" | "float";
  default: number;
  min: number;
  max: number;
  step: number;
  grid: number[];
};

export type Strategy = {
  id: string;
  name: string;
  description: string;
  default_tickers: string;
  params: Param[];
};

export type RunResult = {
  strategy_name: string;
  verdict: "likely real edge" | "inconclusive" | "likely overfit";
  flags: Record<string, boolean>;
  metrics: { label: string; value: string }[];
  tiles: {
    full_sharpe: number | null;
    wf_oos_sharpe: number | null;
    perm_p: number | null;
    dsr: number | null;
    pbo: number | null;
    max_drawdown_pct: number | null;
  };
  split_date: string;
  equity: { dates: string[]; strategy: number[]; benchmark?: number[] };
  scorecard_png: string;
  data_source: string;
  params_used: Record<string, number>;
};

export type RunRequest = {
  strategy_id?: string;
  params?: Record<string, number>;
  tickers?: string;
  strategy_code?: string;
  price_csv?: string;
  start: string;
  end: string;
  split: string;
  cost_bps: number;
  fast: boolean;
  n_folds: number;
};

export type Config = { allow_uploads: boolean };

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function getConfig(): Promise<Config> {
  const res = await fetch(`${API_URL}/config`);
  if (!res.ok) throw new Error(`Could not load config (${res.status})`);
  return res.json();
}

export async function getExampleCsv(): Promise<{ filename: string; content: string }> {
  const res = await fetch(`${API_URL}/example-csv`);
  if (!res.ok) throw new Error(`Could not load example CSV (${res.status})`);
  return res.json();
}

export async function getStrategies(): Promise<Strategy[]> {
  const res = await fetch(`${API_URL}/strategies`);
  if (!res.ok) throw new Error(`Could not load strategies (${res.status})`);
  return res.json();
}

export async function runBacktest(body: RunRequest): Promise<RunResult> {
  const res = await fetch(`${API_URL}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Run failed (${res.status})`);
  }
  return res.json();
}
