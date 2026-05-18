import os
from dataclasses import asdict
import json

import streamlit as st

# Import dashboard first so dashboard/__init__.py runs and bootstraps paths.
from dashboard import DEFAULT_GITHUB_MODEL
from fin_ai.core.request import ModelRequest, RequestPayload
from fin_ai.core.response import ResponseFactory, ResponseMetadata
from fin_ai.agents.tools import YAHOO_FINANCE_TOOLS, LITELLM_TOOL_FUNCTIONS


st.set_page_config(page_title="LiteLLM Chat", layout="wide")
st.title("LiteLLM Provider Chat")
st.caption("Query either GitHub Models or local Ollama using fin_ai.core request/response wrappers.")

provider_label_to_key = {
    "GitHub Models": "github",
    "Local Ollama": "ollama",
}


def _build_tool_aware_system_prompt(base_prompt: str | None) -> str:
    tool_names = [
        tool.get("function", {}).get("name", "")
        for tool in YAHOO_FINANCE_TOOLS
        if tool.get("type") == "function"
    ]
    tool_names = [name for name in tool_names if name]
    available_tools = ", ".join(tool_names) if tool_names else "none"

    tool_guidance = (
        "You have access to function tools via YAHOO_FINANCE_TOOLS. "
        f"Available tools: {available_tools}. "
        "When the user asks for stock prices, company facts, dividends, financial statements, "
        "or analyst recommendations, call the most appropriate tool instead of guessing. "
        "Use the exact tool arguments required by the schema. "
        "After tool output is returned, summarize clearly and reference the returned data."
    )

    base = (base_prompt or "").strip()
    if tool_guidance in base:
        return base
    if not base:
        return tool_guidance
    return f"{base}\n\n{tool_guidance}"


def _extract_tool_calls(message: object) -> list[dict]:
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return []

    extracted: list[dict] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            function_payload = tool_call.get("function", {})
            name = function_payload.get("name")
            arguments_text = function_payload.get("arguments", "{}")
            call_id = tool_call.get("id")
        else:
            function_payload = getattr(tool_call, "function", None)
            name = getattr(function_payload, "name", None)
            arguments_text = getattr(function_payload, "arguments", "{}")
            call_id = getattr(tool_call, "id", None)

        try:
            arguments = json.loads(arguments_text or "{}")
        except json.JSONDecodeError:
            arguments = {}

        extracted.append(
            {
                "id": call_id or f"call_{len(extracted)}",
                "name": name,
                "arguments": arguments,
                "arguments_text": arguments_text or "{}",
            }
        )

    return extracted


def _execute_tool(name: str | None, arguments: dict) -> dict:
    if not name:
        return {"error": "Missing tool name"}

    tool_function = LITELLM_TOOL_FUNCTIONS.get(name)
    if tool_function is None:
        return {"error": f"Unsupported tool: {name}"}

    try:
        return tool_function(**arguments)
    except TypeError as exc:
        return {"error": f"Invalid arguments for {name}: {exc}"}
    except Exception as exc:
        return {"error": str(exc)}

with st.sidebar:
    st.header("Settings")
    provider_label = st.selectbox("Provider", list(provider_label_to_key.keys()), index=0)
    provider = provider_label_to_key[provider_label]

    response_format = st.selectbox(
        "Response Format",
        ResponseFactory.available(),
        index=ResponseFactory.available().index("markdown") if "markdown" in ResponseFactory.available() else 0,
        help="Rendering style implemented by fin_ai.core.response wrappers.",
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
    max_tokens = st.number_input("Max Tokens (optional)", min_value=0, value=0, step=32)
    auto_truncate_prompt = st.checkbox(
        "Auto-truncate prompt for strict-input models (e.g. gpt-5)",
        value=True,
        help="When enabled, oversized prompts are clipped before sending to strict-input models like gpt-5.",
    )
    st.text_input("Proxy Port (optional)", value=os.getenv("PX_PROXY_PORT", ""), key="proxy_port_input")

    if provider == "github":
        st.text_input("GitHub Model", value=os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL), key="github_model")
        st.text_input(
            "GitHub Endpoint",
            value=os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference"),
            key="github_endpoint",
        )
        st.text_input("GitHub Token", value=os.getenv("GITHUB_TOKEN", ""), type="password", key="github_token")
    else:
        st.text_input("Ollama Model", value=os.getenv("OLLAMA_MODEL", "llama3.1"), key="ollama_model")
        st.text_input("Ollama Endpoint", value=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"), key="ollama_endpoint")

system_prompt = st.text_area(
    "System Prompt",
    value="You are a helpful financial analysis assistant.",
    height=100,
)
user_prompt = st.text_area(
    "User Prompt",
    value="Use tools to summarize company info and latest analyst recommendations for AAPL.",
    height=140,
)

effective_system_prompt = _build_tool_aware_system_prompt(system_prompt)
with st.expander("Effective System Prompt (with tool guidance)", expanded=False):
    st.code(effective_system_prompt)

if st.button("Send", type="primary"):
    if not user_prompt.strip():
        st.error("User Prompt cannot be empty.")
    else:
        if provider == "github":
            os.environ["GITHUB_MODEL"] = st.session_state["github_model"]
            os.environ["GITHUB_ENDPOINT"] = st.session_state["github_endpoint"]
            os.environ["GITHUB_TOKEN"] = st.session_state["github_token"]
        else:
            os.environ["OLLAMA_MODEL"] = st.session_state["ollama_model"]
            os.environ["OLLAMA_ENDPOINT"] = st.session_state["ollama_endpoint"]

        raw_proxy_port = st.session_state.get("proxy_port_input", "").strip()
        proxy_port = None
        if raw_proxy_port:
            try:
                proxy_port = int(raw_proxy_port)
            except ValueError:
                st.error("Proxy Port must be a valid integer.")
                st.stop()

        payload = RequestPayload(
            prompt=user_prompt.strip(),
            system_prompt=effective_system_prompt,
            temperature=float(temperature),
            max_tokens=int(max_tokens) if max_tokens > 0 else None,
            proxy_port=proxy_port,
            auto_truncate_prompt=bool(auto_truncate_prompt),
            tools=YAHOO_FINANCE_TOOLS,
        )

        try:
            requester = ModelRequest(provider=provider, format=response_format)
            response = requester.request(payload)

            first_raw = response.get_metadata().raw_response
            first_message = first_raw.choices[0].message
            tool_calls = _extract_tool_calls(first_message)

            if tool_calls:
                follow_up_messages = []
                if payload.system_prompt:
                    follow_up_messages.append({"role": "system", "content": payload.system_prompt})
                follow_up_messages.append({"role": "user", "content": payload.prompt})
                follow_up_messages.append(
                    {
                        "role": "assistant",
                        "content": first_message.content or "",
                        "tool_calls": [
                            {
                                "id": tool_call["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_call["name"],
                                    "arguments": tool_call["arguments_text"],
                                },
                            }
                            for tool_call in tool_calls
                        ],
                    }
                )

                for tool_call in tool_calls:
                    tool_result = _execute_tool(tool_call["name"], tool_call["arguments"])
                    follow_up_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_call["name"] or "unknown_tool",
                            "content": json.dumps(tool_result),
                        }
                    )

                follow_up_payload = RequestPayload(
                    prompt=payload.prompt,
                    system_prompt=payload.system_prompt,
                    temperature=payload.temperature,
                    max_tokens=payload.max_tokens,
                    proxy_port=payload.proxy_port,
                    auto_truncate_prompt=payload.auto_truncate_prompt,
                    tools=YAHOO_FINANCE_TOOLS,
                    messages=follow_up_messages,
                )
                response = requester.client.send(
                    follow_up_payload,
                    response_class=requester.response_class,
                )
        except Exception as exc:
            st.exception(exc)
        else:
            st.subheader("Response")
            if response_format == "markdown":
                st.markdown(response.render())
            else:
                st.text(response.render())

            metadata = response.get_metadata()
            if metadata.prompt_truncated:
                before = metadata.prompt_tokens_before_guard
                after = metadata.prompt_tokens_after_guard
                st.warning(
                    f"Prompt was truncated to fit gpt-5 input limits"
                    f" ({before} -> {after} tokens before sending)."
                )

            metadata_dict = asdict(metadata)
            metadata_dict.pop("raw_response", None)

            st.subheader("Metadata")
            st.json(metadata_dict)

            with st.expander("Raw Response", expanded=False):
                st.text(str(metadata.raw_response))
