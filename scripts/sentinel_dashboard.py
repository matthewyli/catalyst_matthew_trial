from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.cli import build_pipeline  # noqa: E402
from run_strategy_with_dummy_trade import run_strategy  # noqa: E402
from sentinel_loop import (  # noqa: E402
    DEFAULT_ASSET,
    DEFAULT_INTERVAL_SEC,
    DEFAULT_PROMPT,
    DEFAULT_MAX_ERRORS,
    decide_trade,
    derive_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive terminal dashboard for monitoring a continuous strategy sentinel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Strategy prompt to analyze each cycle.")
    parser.add_argument("--asset", default=DEFAULT_ASSET, help="Primary asset ticker.")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC, help="Seconds between runs.")
    parser.add_argument("--no-strict", action="store_true", help="Disable strict I/O if you prefer raw payloads.")
    parser.add_argument("--history", type=int, default=8, help="Number of previous cycles to display.")
    parser.add_argument("--max-errors", type=int, default=DEFAULT_MAX_ERRORS, help="Consecutive errors before exit.")
    parser.add_argument(
        "--min-sentiment",
        type=float,
        help="Override inferred BUY sentiment threshold.",
    )
    parser.add_argument("--min-tvl", type=float, help="Override inferred BUY TVL threshold.")
    parser.add_argument("--max-vol", type=float, help="Override inferred BUY volatility ceiling.")
    parser.add_argument("--sell-sentiment", type=float, help="Override inferred SELL sentiment threshold.")
    parser.add_argument("--sell-tvl", type=float, help="Override inferred SELL TVL threshold.")
    parser.add_argument("--min-score", type=float, help="Override inferred aggregate fallback score.")
    parser.add_argument("--show-raw", action="store_true", help="Display raw Mobula payload excerpts.")
    parser.add_argument("--verbose", action="store_true", help="Include strict value diagnostics every cycle.")
    return parser.parse_args()


def clear_screen() -> None:
    print("\033[2J\033[H", end="")


def format_float(value: Optional[float], none_text: str = "-") -> str:
    if value is None:
        return none_text
    return f"{value:.4f}"


def render_dashboard(
    iteration: int,
    prompt: str,
    asset: str,
    thresholds: Dict[str, float],
    threshold_source: str,
    last_result: Dict[str, object],
    history: list[Dict[str, object]],
    show_raw: bool,
    verbose: bool,
) -> None:
    clear_screen()
    print("=" * 100)
    print(f" Strategy Sentinel Dashboard | asset={asset} | iteration={iteration} | source={threshold_source}")
    print("=" * 100)
    print(f"Prompt: {prompt}")
    print("-" * 100)
    print(
        "Thresholds → "
        f"min_sentiment={format_float(thresholds.get('min_sentiment'))} | "
        f"min_tvl={format_float(thresholds.get('min_tvl'))} | "
        f"max_vol={format_float(thresholds.get('max_vol'))} | "
        f"sell_sentiment={format_float(thresholds.get('sell_sentiment'))} | "
        f"sell_tvl={format_float(thresholds.get('sell_tvl'))} | "
        f"min_score={format_float(thresholds.get('min_score'))}"
    )
    print("-" * 100)

    if last_result:
        timestamp = last_result.get("timestamp")
        aggregate = format_float(last_result.get("aggregate"))
        price = format_float(last_result.get("price"), none_text="n/a")
        decision = last_result.get("decision", "HOLD")
        strict_map = last_result.get("strict_map", {}) or {}
        router = last_result.get("router", {})
        ranking = router.get("tool_ranking")
        executed = router.get("executed_tools")
        rationale = last_result.get("rationale")

        print(f"Last run :: timestamp={timestamp} | decision={decision} | aggregate={aggregate} | price={price}")
        if verbose and strict_map:
            strict_line = ", ".join(f"{k}={format_float(v)}" for k, v in strict_map.items())
            print(f"Strict signals :: {strict_line}")
        if verbose and ranking:
            print(f"Router ranking :: {ranking}")
        if verbose and executed:
            print(f"Executed tools :: {executed}")
        if rationale:
            print(f"Rationale :: {rationale}")
        if show_raw and last_result.get("price_raw"):
            raw_str = str(last_result["price_raw"])
            print(f"Mobula raw (truncated) :: {raw_str[:200]}")
    else:
        print("No runs executed yet.")

    print("-" * 100)
    print("Recent history (most recent first):")
    if not history:
        print("  <no history yet>")
    else:
        print(
            "  {ts:<26} | {decision:<5} | agg={agg:<8} | price={price:<9} | sentiment={sent:<8} | tvl={tvl:<8} | vol={vol:<8}".format(
                ts="timestamp",
                decision="dec",
                agg="aggregate",
                price="price",
                sent="sent",
                tvl="tvl",
                vol="vol",
            )
        )
        for entry in history[:]:
            strict_map = entry.get("strict_map", {}) or {}
            print(
                "  {ts:<26} | {decision:<5} | agg={agg:<8} | price={price:<9} | sentiment={sent:<8} | tvl={tvl:<8} | vol={vol:<8}".format(
                    ts=entry.get("timestamp", "")[:26],
                    decision=str(entry.get("decision", "")),
                    agg=format_float(entry.get("aggregate")),
                    price=format_float(entry.get("price"), none_text="n/a"),
                    sent=format_float(strict_map.get("asknews_impact")),
                    tvl=format_float(strict_map.get("tvl_growth")),
                    vol=format_float(strict_map.get("volatility_percentile")),
                )
            )

    print("=" * 100)
    print("Press Ctrl+C to stop.")


def main() -> None:
    args = parse_args()
    strict_mode = not args.no_strict

    inferred_thresholds, source = derive_thresholds(args.prompt, args.asset)
    thresholds = dict(inferred_thresholds)
    overrides = {
        "min_sentiment": args.min_sentiment,
        "min_tvl": args.min_tvl,
        "max_vol": args.max_vol,
        "sell_sentiment": args.sell_sentiment,
        "sell_tvl": args.sell_tvl,
        "min_score": args.min_score,
    }
    for key, override in overrides.items():
        if override is not None:
            thresholds[key] = override

    pipeline = build_pipeline()
    history: list[Dict[str, object]] = []
    consecutive_errors = 0
    iteration = 0

    last_result: Dict[str, object] = {}

    try:
        while True:
            iteration += 1
            loop_start = time.time()
            try:
                result = run_strategy(
                    prompt=args.prompt,
                    asset=args.asset,
                    strict=strict_mode,
                    pipeline=pipeline,
                )
                router_meta = result.get("output").metadata.get("router", {}) if result.get("output") else {}  # type: ignore[assignment]

                decision, rationale = decide_trade(
                    result.get("strict_map", {}) or {},
                    float(result["aggregate"]),
                    thresholds,
                )
                result["decision"] = decision
                result["rationale"] = rationale
                result["router"] = router_meta

                history.insert(0, result)
                if len(history) > max(1, args.history):
                    history.pop()
                last_result = result
                consecutive_errors = 0
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # pragma: no cover - operational
                consecutive_errors += 1
                last_result = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "decision": "ERROR",
                    "aggregate": None,
                    "price": None,
                    "rationale": [f"exception {exc!r}"],
                    "strict_map": {},
                    "router": {},
                }
                history.insert(0, last_result)
                if len(history) > max(1, args.history):
                    history.pop()
                if consecutive_errors >= max(1, args.max_errors):
                    render_dashboard(
                        iteration,
                        args.prompt,
                        args.asset,
                        thresholds,
                        source,
                        last_result,
                        history,
                        args.show_raw,
                        args.verbose,
                    )
                    print("[dashboard] maximum consecutive errors reached; exiting.")
                    break

            render_dashboard(
                iteration,
                args.prompt,
                args.asset,
                thresholds,
                source,
                last_result,
                history,
                args.show_raw,
                args.verbose,
            )

            elapsed = time.time() - loop_start
            sleep_for = args.interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        clear_screen()
        print("[dashboard] interrupted by user; shutting down.")


if __name__ == "__main__":
    main()
