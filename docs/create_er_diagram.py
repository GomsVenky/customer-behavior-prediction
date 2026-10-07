from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTPUT_PATH = Path(__file__).parent / "er_diagram.png"


def add_table(ax, x, y, width, height, title, fields):
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
        y + height - 0.45,
        title,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    separator_y = y + height - 0.8

    ax.plot(
        [x, x + width],
        [separator_y, separator_y],
        linewidth=1.2,
    )

    field_y = separator_y - 0.45

    for field in fields:
        ax.text(
            x + 0.25,
            field_y,
            field,
            ha="left",
            va="center",
            fontsize=10,
        )
        field_y -= 0.48


def create_er_diagram():
    _fig, ax = plt.subplots(figsize=(13, 7))

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(
        6.5,
        6.55,
        "Customer Behavior Prediction - Database ER Diagram",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )

    prediction_fields = [
        "PK  id : Integer",
        "customer_id : Integer (nullable)",
        "response_probability : Float",
        "predicted_response : Integer",
        "prediction_label : String",
        "created_at : DateTime",
    ]

    segment_fields = [
        "PK  id : Integer",
        "customer_id : Integer",
        "segment_id : Integer",
        "segment_name : String",
        "created_at : DateTime",
    ]

    add_table(
        ax,
        0.8,
        2.0,
        4.8,
        4.0,
        "predictions",
        prediction_fields,
    )

    add_table(
        ax,
        7.4,
        2.0,
        4.8,
        4.0,
        "customer_segments",
        segment_fields,
    )

    # Conceptual customer relationship.
    ax.annotate(
        "",
        xy=(7.4, 3.75),
        xytext=(5.6, 3.75),
        arrowprops={
            "arrowstyle": "<->",
            "linewidth": 1.5,
        },
    )

    ax.text(
        6.5,
        4.1,
        "customer_id",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    ax.text(
        6.5,
        3.4,
        "Conceptual customer reference\n(no database foreign key)",
        ha="center",
        va="center",
        fontsize=9,
    )

    ax.text(
        6.5,
        0.9,
        "PostgreSQL persistence layer",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        6.5,
        0.5,
        "Prediction results and customer segment assignments are "
        "stored independently.",
        ha="center",
        va="center",
        fontsize=9,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"ER diagram saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_er_diagram()