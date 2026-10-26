"""
Visualization module for the Embedding Bias Detector.

Renders a colour-coded dashboard as a PNG:
  - ranked horizontal bar chart of per-concept association scores
  - heatmap of similarity to each group centroid
  - status summary panel with the overall disparity percentage

Run from anywhere in the project:
    python visualization/report.py

Output:
    visualization/bias_report.png
"""

from pathlib import Path
import sys

import numpy as np
from sympy import group
import yaml
import matplotlib

matplotlib.use("Agg")  # headless-safe backend; writes files without a display
import matplotlib.pyplot as plt
import seaborn as sns

# --- Make the project root importable regardless of the working directory ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.detector import BiasDetector
from core.metrics import disparity_percentage

# --- Configuration -----------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
CONFIG_PATH = ROOT / "config" / "groups.yaml"
OUTPUT_PATH = ROOT / "visualization" / "bias_report.png"

# Direction colours: which group a concept leans toward
COLOUR_GROUP_A = "#1f77b4"  # blue
COLOUR_GROUP_B = "#d62728"  # red

# Status bands: (upper bound exclusive, label, colour)
STATUS_BANDS = (
    (20.0, "Low bias", "#2e7d32"),
    (40.0, "Moderate bias", "#f9a825"),
    (float("inf"), "High bias", "#c62828"),
)


def status_for(percentage):
    """Map an overall disparity percentage onto a (label, colour) status."""
    for threshold, label, colour in STATUS_BANDS:
        if percentage < threshold:
            return label, colour
    return "High bias", "#c62828"


def load_config(path=CONFIG_PATH):
    """Read the YAML group definitions."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compute_rows(config, detector):
    """
    Embed every target concept and measure its similarity to each group centroid.

    Returns (rows, group) where each row is a dict with:
        target  - the concept word
        sim_a   - cosine similarity to group A's centroid
        sim_b   - cosine similarity to group B's centroid
        score   - sim_a - sim_b  (positive leans toward group A)
    """
    group = config["gender"]

    centroid_a = detector.group_centroid(detector.embed(group["group_a"]["terms"]))
    centroid_b = detector.group_centroid(detector.embed(group["group_b"]["terms"]))

    targets = config["attributes"]["terms"]
    vectors = detector.embed(targets)

    rows = []
    for target, vector in zip(targets, vectors):
        sim_a = float(np.dot(vector, centroid_a))
        sim_b = float(np.dot(vector, centroid_b))
        rows.append(
            {
                "target": target,
                "sim_a": sim_a,
                "sim_b": sim_b,
                "score": sim_a - sim_b,
            }
        )

    # Sort ascending so the most group-B-leaning concept sits at the bottom
    rows.sort(key=lambda row: row["score"])
    return rows, group


def render(rows, group, overall, output=OUTPUT_PATH):
    """Draw the dashboard and save it to disk."""
    label_a = group["group_a"]["label"]
    label_b = group["group_b"]["label"]
    status_label, status_colour = status_for(overall)

    targets = [row["target"] for row in rows]
    scores = [row["score"] for row in rows]
    bar_colours = [COLOUR_GROUP_A if s > 0 else COLOUR_GROUP_B for s in scores]

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(14, 8.5))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.35, 1.0],
        height_ratios=[1.0, 0.8],
        wspace=0.35,
        hspace=0.38,
    )

    # --- Panel 1: ranked horizontal bar chart --------------------------------
    ax_bar = fig.add_subplot(grid[:, 0])
    y_pos = np.arange(len(targets))
    ax_bar.barh(y_pos, scores, color=bar_colours, height=0.66)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(targets)
    ax_bar.axvline(0, color="#444444", linewidth=0.9)
    ax_bar.set_xlabel(
        f"Association score   (negative \u2192 {label_b}   |   positive \u2192 {label_a})"
    )
    ax_bar.set_title("Per-concept association", loc="left", fontweight="bold")
    ax_bar.grid(axis="y", visible=False)

    # --- Panel 2: similarity heatmap -----------------------------------------
    ax_heat = fig.add_subplot(grid[0, 1])
    matrix = np.array([[row["sim_a"], row["sim_b"]] for row in rows])
    centre = float(matrix.mean())
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=centre,
        xticklabels=[label_a, label_b],
        yticklabels=targets,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "cosine similarity"},
        ax=ax_heat,
    )
    ax_heat.set_title("Similarity to each group centroid", loc="left", fontweight="bold")

    # --- Panel 3: status summary ---------------------------------------------
    ax_sum = fig.add_subplot(grid[1, 1])
    ax_sum.axis("off")

    leans_a = sum(1 for s in scores if s > 0)
    leans_b = sum(1 for s in scores if s < 0)

    ax_sum.text(
        0.5, 0.84, f"{overall:.1f}%",
        fontsize=46, fontweight="bold", color=status_colour,
        ha="center", va="center",
    )
    ax_sum.text(
        0.5, 0.63, "overall disparity",
        fontsize=11, color="#666666", ha="center", va="center",
    )
    ax_sum.text(
        0.5, 0.39, status_label.upper(),
        fontsize=13, fontweight="bold", color=status_colour,
        ha="center", va="center",
    )
    ax_sum.text(
        0.5, 0.15,
        f"{leans_a} concept(s) lean {label_a}  \u00b7  {leans_b} lean {label_b}",
        fontsize=10, color="#666666", ha="center", va="center",
    )

    # --- Titles and footer ---------------------------------------------------
    fig.suptitle(
        "Embedding Bias Report \u2014 Gender \u00d7 Occupations",
        fontsize=17, fontweight="bold", x=0.010, ha="left", y=0.985,
    )
    fig.text(
        0.990, 0.015, f"model: {MODEL_NAME}",
        fontsize=9, color="#888888", ha="right",
    )
    fig.text(
        0.010, 0.015, "Descriptive measure only \u2014 not a compliance artefact.",
        fontsize=9, color="#888888", ha="left",
    )

    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main():
    print("Loading configuration...")
    config = load_config()

    print(f"Loading embedding model ({MODEL_NAME})...")
    detector = BiasDetector(MODEL_NAME)

    print("Computing association scores...")
    rows, group = compute_rows(config, detector)
    overall = disparity_percentage([row["score"] for row in rows])

    status_label, _ = status_for(overall)

    print("\n" + "=" * 55)
    print("EMBEDDING BIAS REPORT \u2014 Gender x Occupations")
    print("=" * 55)
    print(f"{'Concept':<15} {'Score':>9}   Leans toward")
    print("-" * 55)
    for row in rows:
        lean = group["group_a"]["label"] if row["score"] > 0 else group["group_b"]["label"]
        print(f"{row['target']:<15} {row['score']:>+9.4f}   {lean}")
    print("-" * 55)
    print(f"Overall disparity: {overall:.1f}%  [{status_label}]")

    path = render(rows, group, overall)
    print(f"\nDashboard saved to: {path}")


if __name__ == "__main__":
    main()
def compute_rows(config, detector):
    ...
    rows = []
    for target, vector in zip(targets, vectors):
        ...
        rows.append(...)
    
    # Sort ascending so the most group-B-leaning concept sits at the bottom
    rows.sort(key=lambda row: row["score"], reverse=True)  # <-- THIS LINE (line ~76)
    return rows, group
