from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEMPLATE_PATH = Path(__file__).resolve().parent / "sentinel_loop.py"


def _sanitize_name(raw: str, asset: str) -> str:
    stripped = re.sub(r"[^a-zA-Z0-9]+", "_", raw.lower()).strip("_")
    if not stripped:
        stripped = f"{asset.lower()}_sentinel"
    if not stripped.endswith("_sentinel"):
        stripped = f"{stripped}_sentinel"
    return stripped


def _openai_filename(prompt: str, asset: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": "You generate concise snake_case filenames for trading strategy sentinels. Respond with only the filename, no extension.",
            },
            {
                "role": "user",
                "content": f"Strategy prompt: {prompt}\nPrimary asset: {asset}",
            },
        ],
        "max_output_tokens": 64,
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
            # fallback to aggregated text field
            text = data.get("output_text")
        if not text:
            return None
        candidate = text.splitlines()[0].strip()
        candidate = candidate.replace(".py", "")
        return _sanitize_name(candidate, asset)
    except Exception:
        return None


def generate_script_name(prompt: str, asset: str) -> str:
    name = _openai_filename(prompt, asset)
    if not name:
        slug = "_".join(re.findall(r"[A-Za-z0-9]+", prompt.lower())[:3]) or asset.lower()
        name = _sanitize_name(slug, asset)
    path = Path(__file__).resolve().parent / f"{name}.py"
    if path.exists():
        name = f"{name}_{int(time.time())}"
    return name


def customise_template(template: str, prompt: str, asset: str) -> str:
    prompt_literal = json.dumps(prompt, ensure_ascii=False)
    template = re.sub(
        r'DEFAULT_PROMPT = \(\s*"[^"]*"\s*\)',
        f"DEFAULT_PROMPT = (\n    {prompt_literal}\n)",
        template,
        count=1,
    )
    template = re.sub(
        r'DEFAULT_ASSET = "[^"]+"',
        f'DEFAULT_ASSET = "{asset}"',
        template,
        count=1,
    )
    return template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a named sentinel loop script tailored to a specific strategy.",
    )
    parser.add_argument("--prompt", required=True, help="Strategy prompt to bake into the sentinel.")
    parser.add_argument("--asset", default="SOL", help="Primary asset ticker (default: SOL).")
    parser.add_argument("--interval", type=float, default=300.0, help="Default interval in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    template = template.replace(
        "DEFAULT_INTERVAL_SEC = 300.0",
        f"DEFAULT_INTERVAL_SEC = {float(args.interval)}",
        1,
    )
    customised = customise_template(template, args.prompt, args.asset)
    script_name = generate_script_name(args.prompt, args.asset)
    output_path = Path(__file__).resolve().parent / f"{script_name}.py"
    output_path.write_text(customised, encoding="utf-8")

    print(f"[generate] Created {output_path.name}")
    print(f"[generate] Default prompt: {args.prompt!r}")
    print(f"[generate] Default asset: {args.asset}")
    print(f"[generate] Run it with: python scripts/{output_path.name}")


if __name__ == "__main__":
    main()
