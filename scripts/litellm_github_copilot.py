"""LiteLLM GitHub Copilot example using OAuth device flow authentication.

On first use, LiteLLM will prompt for GitHub device authentication by showing:
- a device code
- a verification URL

After authentication, LiteLLM stores the credentials locally for reuse.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from litellm import completion


DEFAULT_MODEL = "github_copilot/gpt-4o"
TOKEN_FILE_NAME = "access-token"


def run_github_copilot(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prompt to GitHub Copilot via LiteLLM and return the response text."""
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def check_auth(token_dir: str) -> None:
    """Check if the current authentication token is valid and offer to reauthenticate if not."""
    token_file = Path(token_dir) / TOKEN_FILE_NAME
    
    if not token_file.exists():
        print(f"Token file not found at {token_file}. You are not authenticated.", file=sys.stderr)
        ans = input("Would you like to authenticate now? (y/N): ").strip().lower()
        if ans == "y":
            os.makedirs(token_dir, exist_ok=True)
            try:
                run_github_copilot("Hello", model=DEFAULT_MODEL)
                print("Authentication successful.")
            except Exception as e:
                print(f"Authentication failed: {e}", file=sys.stderr)
        return

    try:
        with open(token_file, "r") as f:
            content = f.read().strip()
            if content.startswith("{"):
                data = json.loads(content)
                token = data.get("access_token") or data.get("token") or (data.get("github.com", {}).get("oauth_token"))
            else:
                token = content
            
            if not token:
                print("Could not find a valid token in the configuration file. It may be corrupt.", file=sys.stderr)
                # Force reauth by removing the bad file maybe?
                token = ""

        if token:
            req = urllib.request.Request(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
            )
            try:
                with urllib.request.urlopen(req) as response:
                    if response.status == 200:
                        user_data = json.loads(response.read().decode("utf-8"))
                        print(f"Authentication is valid. Authenticated as: {user_data.get('login')}")
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    print("Authentication is invalid or expired.", file=sys.stderr)
                else:
                    print(f"HTTP error checking authentication: {e}", file=sys.stderr)

    except Exception as e:
        print(f"Error checking authentication: {e}", file=sys.stderr)

    ans = input("Would you like to reauthenticate now? (y/N): ").strip().lower()
    if ans == "y":
        try:
            if token_file.exists():
                token_file.unlink() # remove the invalid token so litellm triggers oauth again
            run_github_copilot("Hello", model=DEFAULT_MODEL)
            print("Authentication successful.")
        except Exception as e:
            print(f"Authentication failed: {e}", file=sys.stderr)


def check_available_models(token_dir: str) -> None:
    """Check and display available models directly from the GitHub Copilot API."""
    token_file = Path(token_dir) / TOKEN_FILE_NAME
    
    if not token_file.exists():
        print(f"Token file not found at {token_file}. Initiating authentication...", file=sys.stderr)
        os.makedirs(token_dir, exist_ok=True)
        try:
            run_github_copilot("Hello", model=DEFAULT_MODEL)
        except Exception as e:
            print(f"Authentication failed: {e}", file=sys.stderr)
            return

    try:
        with open(token_file, "r") as f:
            content = f.read().strip()
            # The token format might vary, attempt to grab the auth token wrapper
            if content.startswith("{"):
                data = json.loads(content)
                token = data.get("access_token") or data.get("token") or (data.get("github.com", {}).get("oauth_token"))
            else:
                token = content
            
            if not token:
                print("Could not find a valid token in the configuration file.", file=sys.stderr)
                return

        req = urllib.request.Request(
            "https://api.githubcopilot.com/models",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            models = result.get("data", [])
            print("Available GitHub Copilot Models:")
            for m in models:
                print(f" - {m.get('id')} (Version: {m.get('version')})")

    except Exception as e:
        print(f"Failed to fetch available models: {e}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the GitHub Copilot request."""
    parser = argparse.ArgumentParser(
        description="Send a prompt to GitHub Copilot through the LiteLLM Python SDK.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Prompt text. If omitted, the prompt is read from stdin.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LiteLLM model name to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Check and display available GitHub Copilot models.",
    )
    parser.add_argument(
        "--check-auth",
        action="store_true",
        help="Check if the current authentication is valid and offer to reauthenticate.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for the GitHub Copilot LiteLLM example."""
    token_dir = os.environ.get(
        "GITHUB_COPILOT_TOKEN_DIR",
        str(Path.home() / ".config" / "litellm" / "github_copilot")
    )
    os.environ["GITHUB_COPILOT_TOKEN_DIR"] = token_dir
    os.makedirs(token_dir, exist_ok=True)

    args = parse_args()

    if args.check_auth:
        check_auth(token_dir)
        return 0

    if args.list_models:
        check_available_models(token_dir)
        return 0

    prompt = args.prompt.strip()

    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("A prompt is required. Provide it via --prompt or stdin.", file=sys.stderr)
        return 1

    print(
        "If this is your first run, LiteLLM will display a GitHub device code and verification URL for OAuth authentication.",
        file=sys.stderr,
    )

    try:
        result = run_github_copilot(prompt=prompt, model=args.model)
    except Exception as exc:
        print(f"GitHub Copilot request failed: {exc}", file=sys.stderr)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
