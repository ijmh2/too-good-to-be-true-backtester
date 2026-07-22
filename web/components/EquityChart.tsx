"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { RunResult } from "@/lib/api";

export default function EquityChart({ result }: { result: RunResult }) {
  const { dates, strategy, benchmark } = result.equity;
  const data = dates.map((d, i) => ({
    date: d,
    Strategy: strategy[i],
    ...(benchmark ? { "Buy & hold": benchmark[i] } : {}),
  }));
  const splitLabel = dates.find((d) => d >= result.split_date);

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 8, right: 14, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="date" tick={{ fontSize: 11, fill: "var(--muted)" }} minTickGap={64} />
        <YAxis
          scale="log"
          domain={["auto", "auto"]}
          tick={{ fontSize: 11, fill: "var(--muted)" }}
          width={44}
          tickFormatter={(v: number) => (typeof v === "number" ? v.toFixed(1) : v)}
        />
        <Tooltip
          contentStyle={{
            background: "var(--card)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            fontSize: 12,
          }}
          formatter={(v: number) => (v as number).toFixed(2) + "x"}
        />
        {splitLabel && (
          <ReferenceLine
            x={splitLabel}
            stroke="var(--muted)"
            strokeDasharray="4 4"
            label={{ value: "OOS", fontSize: 10, fill: "var(--muted)", position: "top" }}
          />
        )}
        {benchmark && (
          <Line
            type="monotone"
            dataKey="Buy & hold"
            stroke="var(--muted)"
            dot={false}
            strokeWidth={1.6}
            isAnimationActive={false}
          />
        )}
        <Line
          type="monotone"
          dataKey="Strategy"
          stroke="var(--blue)"
          dot={false}
          strokeWidth={2}
          isAnimationActive={false}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}
