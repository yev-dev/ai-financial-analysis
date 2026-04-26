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

    st.text_input("HTTP Proxy (optional)", value=os.getenv("HTTP_PROXY", ""), key="http_proxy")
    st.text_input("HTTPS Proxy (optional)", value=os.getenv("HTTPS_PROXY", ""), key="https_proxy")

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
            github_token = st.session_state.get("github_token", "").strip()
            if github_token:
                os.environ["GITHUB_TOKEN"] = github_token
            else:
                os.environ.pop("GITHUB_TOKEN", None)
        else:
            os.environ["OLLAMA_MODEL"] = st.session_state["ollama_model"]
            os.environ["OLLAMA_ENDPOINT"] = st.session_state["ollama_endpoint"]

        http_proxy = st.session_state.get("http_proxy", "").strip()
        https_proxy = st.session_state.get("https_proxy", "").strip()
        if http_proxy:
            os.environ["HTTP_PROXY"] = http_proxy
            os.environ["http_proxy"] = http_proxy
        else:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("http_proxy", None)

        if https_proxy:
            os.environ["HTTPS_PROXY"] = https_proxy
            os.environ["https_proxy"] = https_proxy
        else:
            os.environ.pop("HTTPS_PROXY", None)
            os.environ.pop("https_proxy", None)

        payload = RequestPayload(
            prompt=user_prompt.strip(),
            system_prompt=system_prompt.strip() or None,
            temperature=float(temperature),
            max_tokens=int(max_tokens) if max_tokens > 0 else None,
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
            metadata_dict = asdict(metadata)
            metadata_dict.pop("raw_response", None)

            st.subheader("Metadata")
            st.json(metadata_dict)

            with st.expander("Raw Response", expanded=False):
                st.text(str(metadata.raw_response))
