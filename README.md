a volatility regime percentile that takes an assets current realized vol and compares it to historical vol. Then using mobula prices, compute the rolling vol over various windows and percentile over the past year for any regime triggers (a little more automated of a tool)

a cross corr growth metric detecting synchronized growth between to metrics e.g. defi TVL, eth TVL, … and then computes the n-day cagr for each series when both exceed a set threshold.




FUNC1:

python3 vol_pctile.py \
  --asset ethereum \
  --blockchain ethereum \
  --windows 1,3,7,14,21,30,45,60,90 \
  --lookback-days 365 \
  --freq d \
  --mobula-api-key "$MOBULA_API_KEY" \
  --json-output --pretty

NOTE: i removed some fallbacks so it is dependent that you enter in the names precisely as documented by mobula

args:

asset – Mobula asset id or symbol
windows – rolling windows in days (default is `1,3,7,14,21,30,45,60,90`)
lookback-days – Percentile lookback horizon in days (default is 365)
freq – Price interval (`d` for daily, `h` for hourly)
mobula-api-key – the key
json-output / `--pretty` – Gives a structured JSON payload; add pretty for indentation

out:

default:
prints the blended percentile and window-weighted confidence


JSON:
results: realized vol, percentile, and per-window confidence for each window
score_confidence: confidence of the blended percentile based on window coverage
confidence, confidence_breakdown: four-part geometric mean that accounts for coverage, window success, cross-window agreement, and price freshness
meta: observation counts, provider info, optional blockchain