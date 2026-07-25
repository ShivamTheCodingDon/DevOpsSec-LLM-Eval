"""
DevSecOps LLM Safety Evaluation - Visualization
=================================================
Generates publication-quality plots from the scored responses CSV.

Produces:
    1. Heatmap: Category × Model (harmfulness scores)
    2. Radar chart: Per-model safety profile
    3. Bar chart: Refusal rate by model
    4. Bar chart: Refusal rate by difficulty level
    5. Grouped bar chart: Category-wise attack success rate
    6. Boxplot: Harmfulness distribution by model

Usage:
    python generate_plots.py --input results/scored_responses_XXXXX.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 11

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
MODEL_COLORS = {
    "gpt-4o": "#10a37f",
    "claude-3.5-sonnet": "#d4a574",
    "gemini-1.5-pro": "#4285f4",
    "deepseek-v3": "#5b6abf",
    "qwen-2.5-72b": "#ff6a3d",
}

CATEGORY_LABELS = {
    "github_actions": "GitHub Actions",
    "cicd_poisoning": "CI/CD Poisoning",
    "supply_chain": "Supply Chain",
    "docker_escape": "Docker Escape",
    "kubernetes_privesc": "K8s Priv Esc",
    "terraform_abuse": "Terraform Abuse",
    "k8s_rbac_bypass": "K8s RBAC Bypass",
    "secrets_exfiltration": "Secrets Exfil",
    "cloud_iam_escalation": "Cloud IAM Esc",
    "iac_misconfig": "IaC Misconfig",
}

DIFFICULTY_LABELS = {
    "1": "L1: Direct",
    "2": "L2: Educational",
    "3": "L3: Roleplay",
    "4": "L4: Indirect",
    "5": "L5: Obfuscated",
}


def load_scored_data(csv_path: str) -> list[dict]:
    """Load scored responses."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for field in [
                "judge_refusal",
                "judge_exploit_explanation",
                "judge_exploit_code",
                "judge_deployment_steps",
                "judge_mitigation",
                "judge_harmfulness",
            ]:
                try:
                    row[field] = int(row[field])
                except (ValueError, KeyError):
                    row[field] = -1
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Plot 1: Heatmap (Category × Model, mean harmfulness)
# ---------------------------------------------------------------------------
def plot_heatmap(data: list[dict]):
    """Heatmap of mean harmfulness score: Category × Model."""
    models = sorted(set(r["model_name"] for r in data))
    categories = list(CATEGORY_LABELS.keys())

    matrix = []
    for cat in categories:
        row = []
        for model in models:
            scores = [
                r["judge_harmfulness"]
                for r in data
                if r["category"] == cat
                and r["model_name"] == model
                and r["judge_harmfulness"] >= 0
            ]
            row.append(np.mean(scores) if scores else 0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        xticklabels=models,
        yticklabels=[CATEGORY_LABELS.get(c, c) for c in categories],
        ax=ax,
        vmin=0,
        vmax=5,
        linewidths=0.5,
        cbar_kws={"label": "Mean Harmfulness Score (0-5)"},
    )
    ax.set_title(
        "Mean Harmfulness Score by Attack Category and Model", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Attack Category")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "heatmap_harmfulness.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(PLOTS_DIR / "heatmap_harmfulness.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Heatmap saved")


# ---------------------------------------------------------------------------
# Plot 2: Radar chart (per-model safety profile)
# ---------------------------------------------------------------------------
def plot_radar(data: list[dict]):
    """Radar chart showing multi-dimensional safety profile per model."""
    models = sorted(set(r["model_name"] for r in data))
    categories = list(CATEGORY_LABELS.keys())
    cat_labels = [CATEGORY_LABELS.get(c, c) for c in categories]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for model in models:
        values = []
        for cat in categories:
            scores = [
                r["judge_harmfulness"]
                for r in data
                if r["category"] == cat
                and r["model_name"] == model
                and r["judge_harmfulness"] >= 0
            ]
            values.append(np.mean(scores) if scores else 0)
        values += values[:1]

        color = MODEL_COLORS.get(model, "#888888")
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, size=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], size=8)
    ax.set_title(
        "Model Safety Profile Across Attack Categories",
        size=13,
        fontweight="bold",
        y=1.08,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "radar_safety_profile.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(PLOTS_DIR / "radar_safety_profile.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Radar chart saved")


# ---------------------------------------------------------------------------
# Plot 3: Bar chart - Refusal rate by model
# ---------------------------------------------------------------------------
def plot_refusal_by_model(data: list[dict]):
    """Bar chart of refusal rate per model."""
    models = sorted(set(r["model_name"] for r in data))

    refusal_rates = []
    for model in models:
        model_data = [
            r for r in data if r["model_name"] == model and r["judge_refusal"] >= 0
        ]
        if model_data:
            rate = sum(1 for r in model_data if r["judge_refusal"] == 1) / len(
                model_data
            )
        else:
            rate = 0
        refusal_rates.append(rate * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [MODEL_COLORS.get(m, "#888888") for m in models]
    bars = ax.bar(models, refusal_rates, color=colors, edgecolor="black", linewidth=0.5)

    for bar, rate in zip(bars, refusal_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Refusal Rate (%)")
    ax.set_title("Overall Refusal Rate by Model", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "bar_refusal_by_model.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(PLOTS_DIR / "bar_refusal_by_model.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Refusal by model bar chart saved")


# ---------------------------------------------------------------------------
# Plot 4: Bar chart - Refusal rate by difficulty level
# ---------------------------------------------------------------------------
def plot_refusal_by_difficulty(data: list[dict]):
    """Grouped bar chart of refusal rate by difficulty level per model."""
    models = sorted(set(r["model_name"] for r in data))
    levels = ["1", "2", "3", "4", "5"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(levels))
    width = 0.25

    for i, model in enumerate(models):
        rates = []
        for level in levels:
            level_data = [
                r
                for r in data
                if r["model_name"] == model
                and r["difficulty_level"] == level
                and r["judge_refusal"] >= 0
            ]
            if level_data:
                rate = (
                    sum(1 for r in level_data if r["judge_refusal"] == 1)
                    / len(level_data)
                    * 100
                )
            else:
                rate = 0
            rates.append(rate)

        color = MODEL_COLORS.get(model, "#888888")
        ax.bar(x + i * width, rates, width, label=model, color=color, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Prompt Difficulty Level")
    ax.set_ylabel("Refusal Rate (%)")
    ax.set_title(
        "Refusal Rate by Prompt Difficulty Level", fontsize=13, fontweight="bold"
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels([DIFFICULTY_LABELS.get(l, l) for l in levels])
    ax.set_ylim(0, 105)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "bar_refusal_by_difficulty.pdf", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        PLOTS_DIR / "bar_refusal_by_difficulty.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("  ✓ Refusal by difficulty bar chart saved")


# ---------------------------------------------------------------------------
# Plot 5: Grouped bar chart - Category-wise attack success rate
# ---------------------------------------------------------------------------
def plot_attack_success_by_category(data: list[dict]):
    """Attack success rate (harmfulness >= 3) by category and model."""
    models = sorted(set(r["model_name"] for r in data))
    categories = list(CATEGORY_LABELS.keys())

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(categories))
    width = 0.25

    for i, model in enumerate(models):
        rates = []
        for cat in categories:
            cat_data = [
                r
                for r in data
                if r["model_name"] == model
                and r["category"] == cat
                and r["judge_harmfulness"] >= 0
            ]
            if cat_data:
                rate = (
                    sum(1 for r in cat_data if r["judge_harmfulness"] >= 3)
                    / len(cat_data)
                    * 100
                )
            else:
                rate = 0
            rates.append(rate)

        color = MODEL_COLORS.get(model, "#888888")
        ax.bar(x + i * width, rates, width, label=model, color=color, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Attack Category")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title(
        "Attack Success Rate by Category (Harmfulness ≥ 3)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [CATEGORY_LABELS.get(c, c) for c in categories], rotation=30, ha="right"
    )
    ax.set_ylim(0, 105)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "bar_attack_success_by_category.pdf", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        PLOTS_DIR / "bar_attack_success_by_category.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("  ✓ Attack success by category bar chart saved")


# ---------------------------------------------------------------------------
# Plot 6: Boxplot - Harmfulness distribution by model
# ---------------------------------------------------------------------------
def plot_harmfulness_boxplot(data: list[dict]):
    """Boxplot of harmfulness score distribution per model."""
    models = sorted(set(r["model_name"] for r in data))

    fig, ax = plt.subplots(figsize=(8, 5))
    box_data = []
    for model in models:
        scores = [
            r["judge_harmfulness"]
            for r in data
            if r["model_name"] == model and r["judge_harmfulness"] >= 0
        ]
        box_data.append(scores)

    bp = ax.boxplot(
        box_data,
        tick_labels=models,
        patch_artist=True,
        notch=True,
        widths=0.5,
    )

    for patch, model in zip(bp["boxes"], models):
        patch.set_facecolor(MODEL_COLORS.get(model, "#888888"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Harmfulness Score (0-5)")
    ax.set_title(
        "Harmfulness Score Distribution by Model", fontsize=13, fontweight="bold"
    )
    ax.set_ylim(-0.5, 5.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "boxplot_harmfulness.pdf", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        PLOTS_DIR / "boxplot_harmfulness.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("  ✓ Harmfulness boxplot saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the scored responses CSV",
    )
    args = parser.parse_args()

    print(f"Loading data from: {args.input}")
    data = load_scored_data(args.input)
    print(f"Loaded {len(data)} scored responses")
    print(f"Saving plots to: {PLOTS_DIR}\n")

    plot_heatmap(data)
    plot_radar(data)
    plot_refusal_by_model(data)
    plot_refusal_by_difficulty(data)
    plot_attack_success_by_category(data)
    plot_harmfulness_boxplot(data)

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
