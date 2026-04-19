import argparse

from dotenv import load_dotenv
from payload import ModelRequest, RequestPayload, ResponseFactory


def main() -> None:
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="LiteLLM query for Ollama and GitHub Models")
    parser.add_argument(
        "--target",
        choices=["ollama", "github", "all"],
        default="all",
        help="Which backend to query.",
    )
    parser.add_argument(
        "--format",
        choices=ResponseFactory.available(),
        default="console",
        help="Response rendering format: console (default), text, or markdown.",
    )
    parser.add_argument(
        "--prompt",
        default="Give me one short investment risk management tip.",
        help="Prompt text to send to the model.",
    )
    args = parser.parse_args()

    request = RequestPayload(prompt=args.prompt, temperature=0.2)

    if args.target in ("ollama", "all"):
        print("\n=== OLLAMA ===")
        try:
            response = ModelRequest("ollama", format=args.format).request(request)
            response.to_console()
        except Exception as exc:
            raise RuntimeError(
                "Failed to query Ollama. Verify OLLAMA_ENDPOINT and OLLAMA_MODEL configuration."
            ) from exc

    if args.target in ("github", "all"):
        print("\n=== GITHUB MODELS ===")
        try:
            response = ModelRequest("github", format=args.format).request(request)
            response.to_console()
        except Exception as exc:
            raise RuntimeError(
                "Failed to query GitHub Models. Verify GITHUB_TOKEN, GITHUB_MODEL, and GITHUB_ENDPOINT."
            ) from exc


if __name__ == "__main__":
    main()
