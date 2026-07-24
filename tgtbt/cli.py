"""Command-line interface: the full overfit gauntlet from one command.

No server, no Node.js, no browser required — `pip install -e .` and go:

    tgtbt list
    tgtbt run --strategy trend --ticker SPY
    tgtbt run --strategy-file my_strategy.py --ticker SPY
    tgtbt run --strategy trend --price-csv my_data.csv
    tgtbt batch --strategy bollinger --tickers TSLA,NVDA,AAPL,SPY,QQQ

Run `tgtbt <subcommand> --help` for the full flag list on any subcommand.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from tgtbt.batch import run_batch
from tgtbt.costs import CostModel
from tgtbt.data import get_prices_or_fallback
from tgtbt.loaders import exec_strategy_code, parse_uploaded_prices
from tgtbt.registry import REGISTRY, coerce, grid_of
from tgtbt.reporting.scorecard import run_scorecard
from tgtbt.strategies import BuyAndHold

MAX_HISTORY_START = "1970-01-01"  # Yahoo clips this to whatever a ticker actually has


def _parse_params(pairs: list[str] | None) -> dict[str, float]:
    """Turn ["window=20", "k=2.0"] into {"window": 20, "k": 2.0} (int where it parses as one)."""
    out: dict[str, float] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--param must be key=value, got: {pair!r}")
        key, val = pair.split("=", 1)
        try:
            out[key.strip()] = int(val)
        except ValueError:
            out[key.strip()] = float(val)
    return out


def _resolve_strategy(args: argparse.Namespace):
    """Return (strategy_instance, factory, grid, default_tickers_str)."""
    if args.strategy_file:
        code = Path(args.strategy_file).read_text()
        mod = exec_strategy_code(code)
        params = _parse_params(args.param)
        strategy = mod.FACTORY(**params) if params else mod.STRATEGY
        return strategy, mod.FACTORY, mod.GRID, getattr(mod, "TICKERS", "SPY")

    if not args.strategy:
        raise SystemExit("provide --strategy <id> or --strategy-file <path>. See `tgtbt list`.")
    entry = REGISTRY.get(args.strategy)
    if entry is None:
        raise SystemExit(f"unknown --strategy '{args.strategy}'. Known: {', '.join(REGISTRY)}.")
    schema = entry["params"]
    params = coerce(schema, _parse_params(args.param))
    return entry["factory"](**params), entry["factory"], grid_of(schema), entry["default_tickers"]


def _resolve_prices(args: argparse.Namespace, default_tickers: str) -> tuple[pd.DataFrame, str]:
    start = MAX_HISTORY_START if args.max_history else args.start
    end = pd.Timestamp.today().strftime("%Y-%m-%d") if args.max_history else args.end

    if getattr(args, "price_csv", None):
        full = parse_uploaded_prices(Path(args.price_csv).read_text())
        prices = full.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        if len(prices) < 30:
            raise SystemExit(
                f"only {len(prices)} rows between {start} and {end} in {args.price_csv}"
            )
        return prices, f"uploaded CSV ({args.price_csv})"

    tickers = getattr(args, "ticker", None) or default_tickers
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    return get_prices_or_fallback(ticker_list, start=start, end=end)


def _add_common_args(p: argparse.ArgumentParser, *, price_csv: bool) -> None:
    p.add_argument("--strategy", help="Built-in strategy id (see `tgtbt list`)")
    p.add_argument("--strategy-file", help="Path to a .py defining STRATEGY/FACTORY/GRID")
    p.add_argument("--param", action="append", metavar="KEY=VALUE",
                   help="Override a headline parameter, e.g. --param window=30 (repeatable)")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--max-history", action="store_true",
                    help="Ignore --start/--end and fetch everything Yahoo has for the ticker")
    p.add_argument("--split", default=None, help="In/out-of-sample split date (default: ~60%% in)")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--folds", type=int, default=5, help="Walk-forward fold count")
    p.add_argument("--thorough", action="store_true",
                    help="1000 permutation/bootstrap resamples instead of 300 (slower)")
    if price_csv:
        p.add_argument("--price-csv", help="Path to a CSV of your own price data")


def cmd_list(_args: argparse.Namespace) -> None:
    print("Built-in strategies (use with --strategy <id>):\n")
    for sid, e in REGISTRY.items():
        params = ", ".join(
            f"{p['name']}={p['default']} (grid: {p['grid']})" for p in e["params"]
        )
        print(f"  {sid:10s} {e['name']}")
        print(f"             {e['description']}")
        print(f"             default ticker(s): {e['default_tickers']}")
        print(f"             params: {params}\n")
    print("Or bring your own: --strategy-file path/to/my_strategy.py")
    print("(see app/strategy_template.py for the STRATEGY/FACTORY/GRID contract)")


def cmd_run(args: argparse.Namespace) -> None:
    strategy, factory, grid, default_tickers = _resolve_strategy(args)
    prices, source = _resolve_prices(args, default_tickers)
    print(f"data: {source}, {len(prices)} rows, {prices.index[0].date()} -> {prices.index[-1].date()}")

    cost_model = CostModel(args.cost_bps)
    benchmark = BuyAndHold().backtest(prices, cost_model=cost_model).net_returns
    n = 1000 if args.thorough else 300

    n_combos = 1
    for values in grid.values():
        n_combos *= len(values)
    print(f"running the gauntlet: {n_combos} configs x {args.folds}-fold walk-forward, "
          f"{n} permutation resamples, {n} bootstrap resamples, CSCV...")

    t0 = time.perf_counter()
    card = run_scorecard(
        strategy, factory, grid, prices, benchmark=benchmark, split_date=args.split,
        cost_model=cost_model, perm_n=n, boot_n=n, n_folds=args.folds,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 60}\nVERDICT: {card.verdict.upper()}   ({elapsed:.1f}s)\n{'=' * 60}\n")
    for k, v in card.flags.items():
        print(f"  {'PASS' if v else 'FAIL':4s}  {k}")
    print()
    print(card.to_markdown())

    out = args.out or f"scorecard_{strategy.name.split('(')[0]}.png"
    card.figure().savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n[saved] {out}")


def cmd_batch(args: argparse.Namespace) -> None:
    strategy, factory, grid, default_tickers = _resolve_strategy(args)
    tickers = [t.strip().upper() for t in (args.tickers or default_tickers).split(",") if t.strip()]
    headline_params = {k: v for k, v in vars(strategy).items() if k in grid}

    n = 1000 if args.thorough else 300
    df = run_batch(
        factory, headline_params, grid, tickers,
        start=MAX_HISTORY_START if args.max_history else args.start,
        end=pd.Timestamp.today().strftime("%Y-%m-%d") if args.max_history else args.end,
        split_date=args.split, cost_model=CostModel(args.cost_bps),
        perm_n=n, boot_n=n, n_folds=args.folds,
    )
    print("\n" + df.to_string(float_format=lambda x: f"{x:.3f}"))
    if args.csv_out:
        df.to_csv(args.csv_out)
        print(f"\n[saved] {args.csv_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tgtbt", description="Run the overfit-detection gauntlet on a trading strategy."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List built-in strategies")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="Run the full gauntlet on one strategy/ticker")
    _add_common_args(p_run, price_csv=True)
    p_run.add_argument("--ticker", help="Comma-separated ticker(s), e.g. SPY or SPY,TLT,EFA")
    p_run.add_argument("--out", help="Scorecard PNG output path")
    p_run.set_defaults(func=cmd_run)

    p_batch = sub.add_parser("batch", help="Run one strategy across many tickers")
    _add_common_args(p_batch, price_csv=False)
    p_batch.add_argument("--tickers", help="Comma-separated tickers, e.g. TSLA,NVDA,AAPL,SPY,QQQ")
    p_batch.add_argument("--csv-out", help="Save the comparison table to this CSV path")
    p_batch.set_defaults(func=cmd_batch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a clean message, not a traceback, by default
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
