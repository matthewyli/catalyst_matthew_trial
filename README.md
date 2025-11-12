pipeline walkthrough

generic idea of the pipeline:
we want to read a natural language prompt like "scale into SOL when sentiment is positive and volatility is calm" and turn it into a structured plan. the steps are: parse the prompt, pick the right tools, run them in order, capture their outputs, normalize the data, and give an actionable recommendation

tldr:
- prompts go into pipeline/prompt_parser.py to extract goals, assets, indicators
- tool selection happens in pipeline/keyword_detector.py (literal keywords + usage + optional llm hints)
- pipeline/strategy_pipeline.py orchestrates phases and writes everything to runs/<timestamp>
- run_strategy_with_dummy_trade.py prints strict values and simulates a trade
- sentinel_loop.py keeps running the pipeline forever
- sentinel_dashboard.py shows a terminal ui with history and thresholds
- generate_sentinel.py spits out new sentinel scripts per strategy
- train_keyword_weights.py learns better keyword weights from labeled prompts

dir:

pipeline package:
  strategy_pipeline.py -> main
  keyword_detector.py -> tool scoring
  keyword_config.py -> static keyword data
  prompt_parser.py -> simple regex extraction
  execution_context.py -> shared metadata store
  usage_tracker.py -> exponential decay weights
  data_contracts.py -> output validators

tools package:
  textql_context_tool.py -> TextQL primer/runtime research tools
  asknews_tool.py -> news impact score
  tvl_tool.py -> defillama tvl momentum
  volatility_tool.py -> mobula vol percentile
  execution_tool.py -> paper/live execution adapter
  backtest_utils.py -> standardizes backtest payloads
  base.py -> base classes, errors

scripts:
  run_strategy_with_dummy_trade.py -> single shot + simulated trade output
  sentinel_loop.py -> continuous automation with threshold inference
  sentinel_dashboard.py -> live terminal dashboard
  generate_sentinel.py -> creates uniquely named sentinel scripts
  train_keyword_weights.py -> fits keyword weight overrides

supporting files:
  prompts/router_training.jsonl -> labeled prompts for trainer
  prompts/library.json -> regression prompts
  runs/ -> summary.json, tool_runs.json, report.txt per run
  cache/ -> tvl and mobula cache pickles
  tests/test_pipeline_smoke.py -> smoke tests
  .env.example -> copy to .env and fill in keys




EXAMPLE prompt flow step by step - ie how its supposed to go through the pipeline
- user runs a script with a natural language prompt
- prompt_parser extracts assets, timeframes, goals, indicators
- keyword_detector scores each tool, applying llm hints when available
- strategy_pipeline orchestrates phases (data_gather, feature_engineering, signal_generation, risk_sizing, execution)
- each phase runs the selected tools, validates payloads, logs outputs, and saves telemetry
- pipeline returns a structured PipelineOutput object
- run_strategy_with_dummy_trade.py (and the sentinels) convert payloads into strict numeric signals, fetch prices, and produce final guidance

env
required keys: ASKNEWS_API_ID, ASKNEWS_API_KEY, OPENAI_API_KEY (for llm assist and threshold inference), MOBULA_API_KEY (price + vol data). optional toggles: PIPELINE_STRICT_IO=true (default), PIPELINE_DEBUG, PIPELINE_KEYWORD_ALPHA, PIPELINE_BLEND_MODE, PIPELINE_KEYWORD_WEIGHTS_PATH. TextQL context planning uses the `TEXTQL_*` block from `.env.example`: drop in your TextQL API key, point `TEXTQL_RPC_URL` at the QueryOneShot endpoint, and leave the Universal paradigm flags enabled if you want web + python support (set `TEXTQL_SQL_CONNECTOR_ID` and switch the paradigm to `TYPE_SQL` only if you need warehouse access). Tune request reliability with `TEXTQL_TIMEOUT_SEC` (default 120s), `TEXTQL_MAX_RETRIES`, and `TEXTQL_RETRY_DELAY_SEC`, and control how long we wait for `GetAPIChatAnswer` via `TEXTQL_POLL_INTERVAL_SEC`, `TEXTQL_POLL_MAX_ATTEMPTS`, `TEXTQL_POLL_MAX_DURATION_SEC`, and `TEXTQL_POLL_BACKOFF_MULTIPLIER`.

keyword trainer (IMPORTANT!!!!!!!!)
A. edit prompts/router_training.jsonl (or your own json/jsonl) so each entry has a prompt and the tool ids it should pick.
B. run
  python scripts/train_keyword_weights.py --dataset prompts/router_training.jsonl --output pipeline/keyword_weights.json
C. restart the pipeline scripts; keyword_detector will load the new weights automatically (or from PIPELINE_KEYWORD_WEIGHTS_PATH if set).

commands and sample output
single run:
  python scripts/run_strategy_with_dummy_trade.py --prompt "Run a SOL swing strategy that trades only when news sentiment is strongly positive, TVL shows a solid week-over-week climb, and volatility stays in a safe regime." --asset SOL
output looks like:
  [strategy] timestamp=...
  [strategy] aggregate_score=0.1821 decision=BUY
  [strategy] latest_price=194.1023
  [strategy] strict_tool_values=asknews_impact=0.2563, tvl_growth=0.0239, volatility_percentile=0.4125
  [strategy] router_tool_ranking=['tvl_growth', 'asknews_impact', 'volatility_percentile', 'execution_adapter']
  [trade] tool=asknews_impact value=0.2563 raw={...}
  [trade] tool=tvl_growth value=0.0239 raw={...}
  [trade] tool=volatility_percentile value=0.4125 raw={...}
  [trade] tool=execution_adapter value=0.0000 raw={...}
  [trade] Executed simulated BUY at price 194.1023 using aggregate value 0.1821

continuous sentinel:
  python scripts/sentinel_loop.py --prompt "Conservative SOL swing strategy with strong sentiment and rising TVL" --asset SOL --interval 180
per iteration you will see:
  [sentinel] iteration=4 timestamp=...
  [sentinel] threshold_source=openai
  [sentinel] thresholds: {'min_sentiment': 0.80, 'min_tvl': 0.05, 'max_vol': 0.30, 'sell_sentiment': -0.05, 'sell_tvl': 0.01, 'min_score': 0.0}
  [sentinel] router_ranking=['tvl_growth', 'asknews_impact', 'execution_adapter']
  [sentinel] trade_decision=BUY aggregate=0.1734 price=194.2245
  [sentinel] rationale: sentiment/tvl thresholds met; volatility=0.27

live dashboard:
  python scripts/sentinel_dashboard.py --prompt "Aggressive SOL breakout strategy" --asset SOL --interval 120 --verbose
this clears the terminal and prints a live board with thresholds, latest decision, strict values, router ranking, and history.

generate a dedicated sentinel:
  python scripts/generate_sentinel.py --prompt "ETH funding bias scalper" --asset ETH
this emits something like eth_funding_sentinel.py with the prompt baked in.

threshold inference logic
- sentinel_loop.py calls the openai responses api when OPENAI_API_KEY is set. the response should include min_sentiment, min_tvl, max_vol, sell_sentiment, sell_tvl, min_score.
- if no key or the call fails, heuristics look for words like safe, aggressive, or explicit percentages. safe pushes thresholds higher, aggressive lowers them, explicit numbers get normalized.
- users can still override via command line flags (need to figure out how to train llm to pick these if in prompt and not hardcoded into cl?)
- each iteration prints threshold_source openai or heuristic so you can sanity check what happened.

on extending the system (since there will defo be more tools, training, etc.)
- adding a tool: subclass BaseTool, fill out __TOOL_META__, register it, update keyword_config, consider strict value.
- adding a phase: touch strategy_pipeline, data_contracts, and create at least one tool advertising that phase.
- training routing: gather data, run the trainer, deploy the new keyword_weights.json.
- running multiple strategies: use generate_sentinel.py so each has its own script and config.

for testing:
- run python -m pytest for smoke tests (strict mode, caching, run persistence).
- operational steps: set env vars, optionally train keywords, run single test, deploy sentinel loop or generated scripts, use dashboard for monitoring, archive runs/, check thresholds and strict values.
