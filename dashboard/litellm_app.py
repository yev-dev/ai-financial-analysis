import os
from dataclasses import asdict

import streamlit as st

# Import dashboard first so dashboard/__init__.py runs and bootstraps paths.
from dashboard import DEFAULT_GITHUB_MODEL
from fin_ai.core.request import ModelRequest, RequestPayload
from fin_ai.core.response import ResponseFactory


st.set_page_config(page_title="LiteLLM Chat", layout="wide")
st.title("LiteLLM Provider Chat")
st.caption("Query either GitHub Models or local Ollama using fin_ai.core request/response wrappers.")

provider_label_to_key = {
    "GitHub Models": "github",
    "Local Ollama": "ollama",
}

with st.sidebar:
    st.header("Settings")
    provider_label = st.selectbox("Provider", list(provider_label_to_key.keys()), index=0)
    provider = provider_label_to_key[provider_label]

    response_format = st.selectbox(
        "Response Format",
        ResponseFactory.available(),
        index=0,
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
    value="What are the top three indicators of financial health for a public company?",
    height=140,
)

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
            system_prompt=system_prompt.strip() or None,
            temperature=float(temperature),
            max_tokens=int(max_tokens) if max_tokens > 0 else None,
            proxy_port=proxy_port,
            auto_truncate_prompt=bool(auto_truncate_prompt),
        )

        try:
            response = ModelRequest(provider=provider, format=response_format).request(payload)
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
