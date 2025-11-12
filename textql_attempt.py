import json
from tools.textql_context_tool import TextQLPrimerTool
from tools.base import ToolContext
from pathlib import Path

prompt = "Need a textql research brief for BTC with web search plan before other tools"
metadata = {
    "inputs": {
        "symbols": ["BTC"],
        "primary_symbol": "BTC",
        "timeframe_minutes": 240,
    },
    "params": {
        "prompt": prompt,
        "detected_keywords": ["textql"],
        "router": {},
        "parse": {
            "assets": ["BTC"],
            "timeframe_minutes": 240,
            "indicators": [],
            "goals": [],
        },
        "options": {"strict_io": False},
    },
}
context = ToolContext(
    asset="BTC",
    assets=("BTC",),
    detected_keywords=("textql",),
    metadata=metadata,
    usage_counts={},
    weights={}
)

output_path = Path("runs/textql_direct_sample.json")
for attempt in range(1, 6):
    print(f"Attempt {attempt}")
    tool = TextQLPrimerTool()
    result = tool.execute(prompt, context)
    warnings = result.payload.get("warnings")
    if warnings:
        print(" -> warnings:", warnings)
        continue
    print(" -> success; writing to", output_path)
    output_path.write_text(json.dumps(result.payload, indent=2))
    break
else:
    print("No successful attempts during script run")
