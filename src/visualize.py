import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCORES_FILE = "data/judge_scores.csv"
OUTPUT_FILE = "data/results.png"

MODELS = ["groq-llama", "groq-llama70b"]
MODEL_LABELS = {"groq-llama": "Llama 3.1 8B", "groq-llama70b": "Llama 3.3 70B"}
COLORS = {"groq-llama": "#6C8EBF", "groq-llama70b": "#82B366"}
CATEGORIES = ["CONCEPT", "HISTORY", "COMPARE", "ANALYSIS"]
DIFFICULTIES = ["easy", "medium", "hard"]


def load_scores(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["difficulty"] in DIFFICULTIES]


def avg(vals: list) -> float:
    return round(sum(vals) / len(vals), 2) if vals else 0


def group(rows: list[dict], *keys: str) -> dict:
    result = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in keys)
        result[key].append(int(r["judge_score"]))
    return result


def main():
    rows = load_scores(SCORES_FILE)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LLM Football Benchmark — Judge Scores", fontsize=16, fontweight="bold", y=0.98)
    fig.patch.set_facecolor("#F9F9F9")

    bar_width = 0.35
    x = np.arange(len(CATEGORIES))

    # --- 1. Overall by model (bar) ---
    ax = axes[0, 0]
    ax.set_facecolor("#F9F9F9")
    by_model = group(rows, "model")
    means = [avg(by_model[(m,)]) for m in MODELS]
    bars = ax.bar(
        [MODEL_LABELS[m] for m in MODELS], means,
        color=[COLORS[m] for m in MODELS], width=0.45, zorder=3,
    )
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.set_title("Overall Average Score", fontweight="bold")
    ax.set_ylabel("Score (1–10)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # --- 2. By category (grouped bar) ---
    ax = axes[0, 1]
    ax.set_facecolor("#F9F9F9")
    by_model_cat = group(rows, "model", "category")
    for i, model in enumerate(MODELS):
        means = [avg(by_model_cat[(model, cat)]) for cat in CATEGORIES]
        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, label=MODEL_LABELS[model], color=COLORS[model], zorder=3)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylim(0, 10)
    ax.set_title("Score by Category", fontweight="bold")
    ax.set_ylabel("Score (1–10)")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # --- 3. By difficulty (grouped bar) ---
    ax = axes[1, 0]
    ax.set_facecolor("#F9F9F9")
    by_model_diff = group(rows, "model", "difficulty")
    x2 = np.arange(len(DIFFICULTIES))
    for i, model in enumerate(MODELS):
        means = [avg(by_model_diff[(model, d)]) for d in DIFFICULTIES]
        offset = (i - 0.5) * bar_width
        bars = ax.bar(x2 + offset, means, bar_width, label=MODEL_LABELS[model], color=COLORS[model], zorder=3)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(x2)
    ax.set_xticklabels([d.capitalize() for d in DIFFICULTIES])
    ax.set_ylim(0, 10)
    ax.set_title("Score by Difficulty", fontweight="bold")
    ax.set_ylabel("Score (1–10)")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # --- 4. Score distribution (box plot) ---
    ax = axes[1, 1]
    ax.set_facecolor("#F9F9F9")
    by_model_all = group(rows, "model")
    data = [by_model_all[(m,)] for m in MODELS]
    bp = ax.boxplot(
        data,
        tick_labels=[MODEL_LABELS[m] for m in MODELS],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        widths=0.4,
    )
    for patch, model in zip(bp["boxes"], MODELS):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.8)
    ax.set_ylim(0, 10)
    ax.set_title("Score Distribution", fontweight="bold")
    ax.set_ylabel("Score (1–10)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
