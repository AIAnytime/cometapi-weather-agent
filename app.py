import os

import requests
import streamlit as st
from dotenv import load_dotenv

from agent import BASE_URL, build_agent

load_dotenv()

MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "claude-haiku-4-5-20251001",
    "deepseek-chat",
    "glm-4.7",
]

EXAMPLES = [
    "Is it good weather for a run in Bengaluru today?",
    "Should I carry an umbrella in London tomorrow?",
    "Compare the air quality in Delhi and Mumbai right now.",
]

st.set_page_config(page_title="Weather Agent", page_icon="🌤️")

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont,
        "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
      .footer { color: #888; font-size: 0.85rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600)
def model_count(api_key: str) -> int:
    r = requests.get(
        f"{BASE_URL}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=15
    )
    return len(r.json().get("data", []))


api_key = os.environ.get("COMET_API_KEY", "")

with st.sidebar:
    st.subheader("Settings")
    if not api_key:
        api_key = st.text_input("CometAPI key", type="password")
    model = st.selectbox("Model", MODELS)
    st.caption("Same agent code, any model — swap the model id.")
    if api_key:
        try:
            st.caption(f"{model_count(api_key)} models available on CometAPI.")
        except Exception:
            pass
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("Weather Agent")
st.caption(
    "A LangGraph agent with three tools — place lookup, forecast and air quality "
    "(Open-Meteo, no key needed). The model runs through CometAPI."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("steps"):
            with st.expander("Agent steps"):
                st.code(m["steps"], language="text")
        st.markdown(m["content"])

cols = st.columns(len(EXAMPLES))
picked = None
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        picked = example

prompt = st.chat_input("Ask about the weather anywhere...") or picked

if prompt:
    if not api_key:
        st.error("Add COMET_API_KEY to your .env file, or paste a key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [(m["role"], m["content"]) for m in st.session_state.messages]

    with st.chat_message("assistant"):
        steps: list[str] = []
        box = st.status(f"Thinking with {model}...", expanded=True)
        try:
            app = build_agent(model, api_key)
            answer = ""
            for update in app.stream({"messages": history}, stream_mode="updates"):
                for node, payload in update.items():
                    for msg in payload["messages"]:
                        if node == "model" and msg.tool_calls:
                            for call in msg.tool_calls:
                                line = f"call {call['name']}({call['args']})"
                                steps.append(line)
                                box.write(line)
                        elif node == "tools":
                            steps.append(f"{msg.name} -> {msg.content}")
                            box.write(f"{msg.name} returned data")
                        elif node == "model":
                            answer = msg.content
            box.update(label=f"Answered with {model}", state="complete", expanded=False)
        except Exception as exc:
            box.update(label="Failed", state="error")
            st.error(str(exc))
            st.stop()

        answer = answer or "The model returned an empty answer. Try another model."
        st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "steps": "\n".join(steps)}
        )

st.markdown(
    '<div class="footer">Built by AI Anytime with ❤️</div>', unsafe_allow_html=True
)
