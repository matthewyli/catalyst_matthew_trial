from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from catalyst_matthew_trial.pipeline.cli import build_pipeline  # noqa: E402
from run_strategy_with_dummy_trade import run_strategy  # noqa: E402

DEFAULT_PROMPT = (
    "Continuously monitor SOL for news shocks, rising TVL momentum, and subdued volatility to guide swing trades."
)
DEFAULT_ASSET = "SOL"
DEFAULT_INTERVAL_SEC = 300.0
DEFAULT_BUY_SENTIMENT = 0.20
DEFAULT_BUY_TVL = 0.02
DEFAULT_MAX_VOL = 0.50
DEFAULT_SELL_SENTIMENT = -0.10
DEFAULT_SELL_TVL = 0.00
DEFAULT_MIN_SCORE = 0.0
DEFAULT_MAX_ERRORS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the catalyst strategy pipeline in a continuous sentinel loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Strategy prompt to evaluate each cycle.")
    parser.add_argument("--asset", default=DEFAULT_ASSET, help="Primary asset ticker.")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC, help="Seconds between runs.")
    parser.add_argument("--min-sentiment", type=float, help="Minimum sentiment required to BUY.")
    parser.add_argument("--min-tvl", type=float, help="Minimum TVL momentum required to BUY.")
    parser.add_argument("--max-vol", type=float, help="Maximum volatility allowed to BUY.")
    parser.add_argument("--sell-sentiment", type=float, help="Sentiment threshold to SELL.")
    parser.add_argument("--sell-tvl", type=float, help="TVL threshold to SELL.")
    parser.add_argument("--min-score", type=float, help="Fallback aggregate score threshold.")
    parser.add_argument("--max-errors", type=int, default=DEFAULT_MAX_ERRORS, help="Consecutive errors before exit.")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable strict I/O (numeric values) if you prefer raw payloads.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed diagnostics each cycle.")
    return parser.parse_args()


def decide_trade(
    strict_map: dict[str, float],
    aggregate: float,
    thresholds: dict[str, float],
) -> tuple[str, list[str]]:
    sentiment = strict_map.get("asknews_impact")
    tvl_growth = strict_map.get("tvl_growth")
    volatility = strict_map.get("volatility_percentile")

    reasons: list[str] = []
    decision = "HOLD"

    buy_conditions = (
        sentiment is not None
        and tvl_growth is not None
        and sentiment >= thresholds["min_sentiment"]
        and tvl_growth >= thresholds["min_tvl"]
        and (volatility is None or volatility <= thresholds["max_vol"])
    )
    sell_conditions = (
        sentiment is not None
        and tvl_growth is not None
        and sentiment <= thresholds["sell_sentiment"]
        and tvl_growth <= thresholds["sell_tvl"]
    )

    if buy_conditions:
        decision = "BUY"
        reasons.append("sentiment/tvl thresholds met")
        if volatility is not None:
            reasons.append(f"volatility={volatility:.4f}")
    elif sell_conditions:
        decision = "SELL"
        reasons.append("bearish thresholds met")
    elif aggregate >= thresholds["min_score"]:
        decision = "BUY" if aggregate > 0 else "SELL" if aggregate < 0 else "HOLD"
        reasons.append("aggregate fallback")
    else:
        reasons.append("no thresholds met")

    return decision, reasons


def _call_openai_for_thresholds(prompt: str, asset: str) -> Optional[Dict[str, float]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You convert trading strategy prompts into numeric thresholds. "
                    "Return strict JSON with keys: min_sentiment, min_tvl, max_vol, sell_sentiment, sell_tvl, min_score. "
                    "Values should be floats between -1 and 1 for sentiment, 0-1 for TVL and volatility. "
                    "If the prompt is subjective (e.g. 'safe'), choose sensible conservative numbers."
                ),
            },
            {
                "role": "user",
                "content": f"Strategy prompt: {prompt}\nPrimary asset: {asset}",
            },
        ],
        "max_output_tokens": 200,
    }
    try:
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("output", [{}])[0].get("content", [{}])[0].get("text")  # type: ignore[index]
        if not text:
            text = data.get("output_text")
        if not text:
            return None
        extracted = json.loads(text.splitlines()[0])
        return {
            "min_sentiment": float(extracted["min_sentiment"]),
            "min_tvl": float(extracted["min_tvl"]),
            "max_vol": float(extracted["max_vol"]),
            "sell_sentiment": float(extracted["sell_sentiment"]),
            "sell_tvl": float(extracted["sell_tvl"]),
            "min_score": float(extracted["min_score"]),
        }
    except Exception:
        return None


def _heuristic_thresholds(prompt: str) -> Dict[str, float]:
    text = prompt.lower()
    thresholds = {
        "min_sentiment": DEFAULT_BUY_SENTIMENT,
        "min_tvl": DEFAULT_BUY_TVL,
        "max_vol": DEFAULT_MAX_VOL,
        "sell_sentiment": DEFAULT_SELL_SENTIMENT,
        "sell_tvl": DEFAULT_SELL_TVL,
        "min_score": DEFAULT_MIN_SCORE,
    }

    def clamp(v: float, min_v: float, max_v: float) -> float:
        return max(min_v, min(max_v, v))

    if "very safe" in text or "ultra safe" in text:
        thresholds.update(
            {
                "min_sentiment": 0.85,
                "min_tvl": 0.06,
                "max_vol": 0.25,
                "sell_sentiment": -0.05,
                "sell_tvl": 0.02,
            }
        )
    elif "safe" in text or "conservative" in text:
        thresholds.update(
            {
                "min_sentiment": 0.8,
                "min_tvl": 0.05,
                "max_vol": 0.3,
                "sell_sentiment": -0.05,
                "sell_tvl": 0.01,
            }
        )
    elif "aggressive" in text or "fast" in text or "high risk" in text:
        thresholds.update(
            {
                "min_sentiment": 0.15,
                "min_tvl": 0.01,
                "max_vol": 0.7,
                "sell_sentiment": -0.2,
                "sell_tvl": -0.02,
                "min_score": -0.05,
            }
        )

    sentiment_matches = re.findall(r"sentiment[^0-9]*(\d+(?:\.\d+)?)", text)
    if sentiment_matches:
        value = float(sentiment_matches[0])
        if value > 1:
            value = value / 100.0
        thresholds["min_sentiment"] = clamp(value, 0.0, 1.0)
    tvl_matches = re.findall(r"tvl[^0-9]*(\d+(?:\.\d+)?)", text)
    if tvl_matches:
        value = float(tvl_matches[0])
        if value > 1:
            value = value / 100.0
        thresholds["min_tvl"] = clamp(value, 0.0, 1.0)
    vol_matches = re.findall(r"vol(?:atility)?[^0-9]*(\d+(?:\.\d+)?)", text)
    if vol_matches:
        value = float(vol_matches[0])
        if value > 1:
            value = value / 100.0
        thresholds["max_vol"] = clamp(value, 0.05, 1.0)

    return thresholds


def derive_thresholds(prompt: str, asset: str) -> tuple[dict[str, float], str]:
    llm_thresholds = _call_openai_for_thresholds(prompt, asset)
    if llm_thresholds:
        return llm_thresholds, "openai"
    return _heuristic_thresholds(prompt), "heuristic"


def main() -> None:
    args = parse_args()
    strict_mode = not args.no_strict
    auto_thresholds, source = derive_thresholds(args.prompt, args.asset)
    thresholds = dict(auto_thresholds)

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
    consecutive_errors = 0
    iteration = 0

    print(
        "[sentinel] starting with prompt={prompt!r}, asset={asset}, interval={interval}s".format(
            prompt=args.prompt,
            asset=args.asset,
            interval=args.interval,
        )
    )
    print(f"[sentinel] threshold_source={source}")
    print("[sentinel] thresholds:", thresholds)
    print("[sentinel] strict_io={}".format(strict_mode))

    try:
        while True:
            iteration += 1
            loop_start = time.time()
            print(
                "[sentinel] iteration={i} timestamp={ts}".format(
                    i=iteration,
                    ts=datetime.now(timezone.utc).isoformat(),
                )
            )
            try:
                result = run_strategy(
                    prompt=args.prompt,
                    asset=args.asset,
                    strict=strict_mode,
                    pipeline=pipeline,
                )
                consecutive_errors = 0

                strict_map = result.get("strict_map", {}) or {}
                if result.get("output"):
                    router_meta = result["output"].metadata.get("router", {})  # type: ignore[assignment]
                    ranking = router_meta.get("tool_ranking")
                    if ranking:
                        print(f"[sentinel] router_ranking={ranking}")
                    executed = router_meta.get("executed_tools")
                    if executed:
                        print(f"[sentinel] executed_tools={executed}")
                aggregate = float(result["aggregate"])
                decision, rationale = decide_trade(strict_map, aggregate, thresholds)
                price = result["price"]

                if args.verbose:
                    tool_dump = ", ".join(f"{k}={v:.4f}" for k, v in strict_map.items())
                    print(f"[sentinel][debug] strict_values: {tool_dump}")
                    raw = result.get("price_raw")
                    if raw is not None:
                        raw_str = str(raw)
                        print(f"[sentinel][debug] mobula_raw: {raw_str[:200]}")

                print(
                    "[sentinel] trade_decision={decision} aggregate={aggregate:.4f} price={price}".format(
                        decision=decision,
                        aggregate=aggregate,
                        price=f"{price:.4f}" if price is not None else "unavailable",
                    )
                )
                print("[sentinel] rationale:", "; ".join(rationale))
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # pragma: no cover - operational
                consecutive_errors += 1
                print(f"[sentinel] ERROR {exc!r} (consecutive={consecutive_errors})")
                if consecutive_errors >= args.max_errors:
                    print("[sentinel] maximum consecutive errors reached; exiting.")
                    break

            elapsed = time.time() - loop_start
            sleep_for = args.interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("[sentinel] interrupted by user; shutting down.")


if __name__ == "__main__":
    main()
