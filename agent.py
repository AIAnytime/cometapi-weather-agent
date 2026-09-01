"""A small LangGraph agent that answers weather questions using free, key-less APIs.

The model is served through CometAPI, which exposes many providers behind one
OpenAI-compatible endpoint, so the same agent code runs on any model id.
"""

import os
from typing import Annotated, TypedDict

import requests
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

BASE_URL = "https://api.cometapi.com/v1"
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

SYSTEM_PROMPT = (
    "You are a weather assistant. Use the tools to look up real data before answering. "
    "Always call find_place first to turn a place name into coordinates. "
    "Answer in at most four short sentences, mention the numbers you used, "
    "and give one concrete recommendation."
)


@tool
def find_place(name: str) -> str:
    """Find the latitude, longitude, country and timezone of a place by name."""
    r = requests.get(GEO_URL, params={"name": name, "count": 1}, timeout=15)
    results = r.json().get("results")
    if not results:
        return f"No place found for {name!r}."
    p = results[0]
    return (
        f"{p['name']}, {p.get('country', '')} | lat={p['latitude']} lon={p['longitude']} "
        f"| timezone={p.get('timezone')}"
    )


@tool
def get_weather(latitude: float, longitude: float) -> str:
    """Get current weather and a 3-day forecast for a latitude/longitude."""
    r = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 3,
            "timezone": "auto",
        },
        timeout=15,
    )
    d = r.json()
    c, day = d["current"], d["daily"]
    lines = [
        f"now: {c['temperature_2m']}C, humidity {c['relative_humidity_2m']}%, "
        f"precip {c['precipitation']}mm, wind {c['wind_speed_10m']}km/h"
    ]
    for i, date in enumerate(day["time"]):
        lines.append(
            f"{date}: {day['temperature_2m_min'][i]}-{day['temperature_2m_max'][i]}C, "
            f"rain chance {day['precipitation_probability_max'][i]}%"
        )
    return "\n".join(lines)


@tool
def get_air_quality(latitude: float, longitude: float) -> str:
    """Get the current air quality (PM2.5, PM10, European AQI) for a latitude/longitude."""
    r = requests.get(
        AIR_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "pm2_5,pm10,european_aqi",
            "timezone": "auto",
        },
        timeout=15,
    )
    c = r.json()["current"]
    return f"european_aqi={c['european_aqi']}, pm2_5={c['pm2_5']}, pm10={c['pm10']}"


TOOLS = [find_place, get_weather, get_air_quality]


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_agent(model: str, api_key: str | None = None, temperature: float = 0.2):
    """Build the graph: the model decides, the tool node fetches, repeat until done."""
    llm = ChatOpenAI(
        model=model,
        base_url=BASE_URL,
        api_key=api_key or os.environ["COMET_API_KEY"],
        temperature=temperature,
        max_tokens=500,
        max_retries=1,
    ).bind_tools(TOOLS)

    def call_model(state: State):
        messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(messages)]}

    def should_continue(state: State):
        return "tools" if state["messages"][-1].tool_calls else END

    graph = StateGraph(State)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue, ["tools", END])
    graph.add_edge("tools", "model")
    return graph.compile()


if __name__ == "__main__":
    app = build_agent("gpt-4o-mini")
    out = app.invoke({"messages": [("user", "Is it good weather for a run in Bengaluru today?")]})
    for m in out["messages"]:
        m.pretty_print()
