import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "too-good-to-be-true-backtester",
  description:
    "Run the overfit gauntlet on a trading strategy: walk-forward, permutation nulls, " +
    "Monte-Carlo, Deflated Sharpe and CSCV probability-of-backtest-overfitting, in one verdict.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
