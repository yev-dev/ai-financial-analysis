"""CLI tool for querying LLM providers via the factory.

Usage::

    python scripts/litellm_query_v2.py --target ollama --format markdown
    python scripts/litellm_query_v2.py --target github --prompt "Hello"
"""

import argparse
import sys
import os

from dotenv import load_dotenv

# Ensure src/ is on sys.path
_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from fin_ai.core.request import create_llm_client, RequestPayload  # noqa: E402
from fin_ai.core.response import ResponseFactory  # noqa: E402


def main() -> None:
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="LiteLLM query via create_llm_client")
    parser.add_argument(
        "--target",
        choices=["ollama", "github", "deepseek", "all"],
        default="all",
        help="Which backend to query.",
    )
    parser.add_argument(
        "--format",
        choices=ResponseFactory.available(),
        default="console",
        help="Response rendering format.",
    )
    parser.add_argument(
        "--prompt",
        default="Give me one short investment risk management tip.",
        help="Prompt text to send to the model.",
    )
    args = parser.parse_args()
    response_class = ResponseFactory.get(args.format)

    payload = RequestPayload(prompt=args.prompt, temperature=0.2)

    targets = ["ollama", "github"] if args.target == "all" else [args.target]

    for target in targets:
        print(f"\n=== {target.upper()} ===")
        try:
            client = create_llm_client(target)
            response = client.send(payload, response_class=response_class)
            response.to_console()
        except Exception as exc:
            print(f"Failed to query {target}: {exc}")


if __name__ == "__main__":
    main()
