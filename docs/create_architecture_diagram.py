from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTPUT_PATH = Path(__file__).parent / "architecture_diagram.png"


def add_box(ax, x, y, width, height, title, details):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        fill=False,
    )
    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height * 0.67,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        x + width / 2,
        y + height * 0.32,
        details,
        ha="center",
        va="center",
        fontsize=8,
        wrap=True,
    )


def add_arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.5,
        },
    )


def create_architecture_diagram():
    _fig, ax = plt.subplots(figsize=(16, 10))

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        8,
        9.55,
        "Customer Behavior Prediction - System Architecture",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    # Data layer
    add_box(
        ax,
        0.5,
        7.5,
        2.5,
        1.2,
        "Raw Data",
        "marketing_campaign.csv",
    )

    add_box(
        ax,
        3.6,
        7.5,
        2.8,
        1.2,
        "Data Pipeline",
        "Loading\nCleaning\nFeature Engineering",
    )

    add_box(
        ax,
        7.0,
        7.5,
        2.8,
        1.2,
        "Processed Features",
        "customer_features.csv\n38 prediction features",
    )

    add_arrow(ax, (3.0, 8.1), (3.6, 8.1))
    add_arrow(ax, (6.4, 8.1), (7.0, 8.1))

    # Model layer
    add_box(
        ax,
        1.0,
        5.2,
        3.0,
        1.3,
        "Response Prediction",
        "LightGBM\nThreshold = 0.20",
    )

    add_box(
        ax,
        4.7,
        5.2,
        3.0,
        1.3,
        "Customer Value",
        "Random Forest Regression\nHistorical CLV Proxy",
    )

    add_box(
        ax,
        8.4,
        5.2,
        3.0,
        1.3,
        "Segmentation",
        "K-Means\n2 Business Segments",
    )

    add_box(
        ax,
        12.1,
        5.2,
        3.0,
        1.3,
        "Explainability",
        "SHAP\nLIME\nPartial Dependence",
    )

    add_arrow(ax, (8.4, 7.5), (2.5, 6.5))
    add_arrow(ax, (8.4, 7.5), (6.2, 6.5))
    add_arrow(ax, (8.4, 7.5), (9.9, 6.5))
    add_arrow(ax, (9.8, 8.1), (13.6, 6.5))

    # Application layer
    add_box(
        ax,
        2.0,
        2.8,
        4.0,
        1.3,
        "FastAPI Service",
        "Single Prediction\nBatch CSV\nMetadata\nHealth",
    )

    add_box(
        ax,
        7.0,
        2.8,
        4.0,
        1.3,
        "Streamlit Dashboard",
        "Response Prediction\nSegmentation\nCLV Proxy\nSHAP + LIME",
    )

    add_box(
        ax,
        12.0,
        2.8,
        3.0,
        1.3,
        "PostgreSQL",
        "Predictions\nCustomer Segments",
    )

    add_arrow(ax, (2.5, 5.2), (4.0, 4.1))
    add_arrow(ax, (6.2, 5.2), (8.5, 4.1))
    add_arrow(ax, (9.9, 5.2), (9.5, 4.1))
    add_arrow(ax, (13.6, 5.2), (10.0, 4.1))

    add_arrow(ax, (6.0, 3.45), (7.0, 3.45))
    add_arrow(ax, (6.0, 3.15), (12.0, 3.15))

    # Deployment layer
    add_box(
        ax,
        3.0,
        0.6,
        10.0,
        1.1,
        "Docker Compose Deployment",
        "FastAPI Container  |  Streamlit Container  |  PostgreSQL Container",
    )

    add_arrow(ax, (4.0, 2.8), (5.0, 1.7))
    add_arrow(ax, (9.0, 2.8), (8.0, 1.7))
    add_arrow(ax, (13.5, 2.8), (11.0, 1.7))

    ax.text(
        15.4,
        0.25,
        "Customer Behavior Prediction",
        ha="right",
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Architecture diagram saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_architecture_diagram()