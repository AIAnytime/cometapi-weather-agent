"""Render the high-level architecture diagram to architecture.jpg."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]

INK = "#1b1b1b"
MUTED = "#6f6f6f"
LINE = "#b0b0b0"

fig, ax = plt.subplots(figsize=(10, 7.8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 80)
ax.axis("off")


def box(x, y, w, h, title, sub=None, fill="#ffffff"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=1.6",
            facecolor=fill, edgecolor="#c9c9c9", linewidth=1.2,
        )
    )
    ty = y + h / 2 + (2.2 if sub else 0)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=12, color=INK)
    if sub:
        ax.text(
            x + w / 2, y + h / 2 - 3.0, sub,
            ha="center", va="center", fontsize=9, color=MUTED, linespacing=1.6,
        )


def arrow(p1, p2):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=11,
            color=LINE, linewidth=1.2, shrinkA=0, shrinkB=0,
        )
    )


def label(x, y, text, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=8.5, color=MUTED)


# top: the model gateway
box(40, 55, 44, 13, "CometAPI",
    "one OpenAI-compatible endpoint\n500+ models — GPT · Gemini · Claude · GLM")

# middle row: user, ui, agent
box(2, 30, 16, 12, "User", "a weather\nquestion", fill="#fafafa")
box(21, 30, 16, 12, "Streamlit UI", "chat +\nmodel picker", fill="#fafafa")
box(40, 28, 44, 16, "LangGraph agent", "model  ↔  tools\nloops until it can answer")

# bottom: the data source
box(40, 4, 44, 13, "Open-Meteo",
    "free, no API key\ngeocoding · forecast · air quality")

arrow((18, 36), (21, 36))
arrow((37, 36), (40, 36))

arrow((54, 44), (54, 55))
label(52, 49.5, "prompt +\ntool schemas", ha="right")
arrow((70, 55), (70, 44))
label(72, 49.5, "answer or\ntool call", ha="left")

arrow((54, 28), (54, 17))
label(52, 22.5, "tool call", ha="right")
arrow((70, 17), (70, 28))
label(72, 22.5, "live data", ha="left")

ax.text(2, 77, "Weather Agent — architecture", ha="left", va="center", fontsize=15, color=INK)
ax.text(2, 72.5,
        "One agent, any model: swap the model id and the same graph and tools keep running.",
        ha="left", va="center", fontsize=9.5, color=MUTED)
ax.text(2, 0.5, "Built by AI Anytime with ❤️", ha="left", va="center", fontsize=8.5, color=MUTED)

fig.savefig("architecture.jpg", dpi=200, format="jpg", bbox_inches="tight",
            pad_inches=0.35, facecolor="white")
print("wrote architecture.jpg")
