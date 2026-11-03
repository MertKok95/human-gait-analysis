"""
A side-by-side bar graph comparing the mean absolute error (MAE) 
of the model using the reference camera method.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLE = Path("../data/outputs/thesis_tables/Eval_Table3_Baseline_vs_Model.csv")
OUT   = Path("../data/outputs/figures/")

# 4 spatial parameters to be shown in the graph (the group showing the most pronounced improvement)
ROWS = ["Step Length - usual", "Stride Length - usual",
        "Step Length - fast",  "Stride Length - fast"]
LABELS = ["Step\n(usual)", "Stride\n(usual)", "Step\n(fast)", "Stride\n(fast)"]

BASE_C  = "#C0504D"   # reference -> red
MODEL_C = "#1F63B0"   # model    -> blue


def main():
    df = pd.read_csv(TABLE).set_index("Parameter")
    base  = [float(df.loc[r, "Baseline_MAE"]) for r in ROWS]
    model = [float(df.loc[r, "Model_MAE"])    for r in ROWS]
    gain  = [float(df.loc[r, "Improvement_Percent"]) for r in ROWS]
    n     = int(df.loc[ROWS[0], "N"])

    x = np.arange(len(ROWS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 6.2))

    b1 = ax.bar(x - w/2, base,  w, label="Reference camera method",
                color=BASE_C, edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + w/2, model, w, label="The proposed model",
                color=MODEL_C, edgecolor="black", linewidth=0.8)

    for bar, v in list(zip(b1, base)) + list(zip(b2, model)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.6, f"{v:.1f}",
                ha="center", fontsize=11, fontweight="bold")

    # improvement percentage
    top = max(base)
    for i, g in enumerate(gain):
        ax.annotate(f"−%{g:.0f}", xy=(x[i], top * 1.09), ha="center",
                    fontsize=13, fontweight="bold", color="#1a7a3c")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=12)
    ax.set_ylabel("Mean absolute error (cm)", fontsize=12)
    ax.set_title(f"Error reduction in spatial parameters  (n = {n})",
                 fontsize=14, fontweight="bold", pad=34)
    ax.set_ylim(0, top * 1.34)
    ax.legend(fontsize=11, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.005), framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "error_reduction.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "error_reduction.svg",
                bbox_inches="tight", facecolor="white")
    print("[OK] error_reduction.png / .svg producted")
    for r, b, m, g in zip(ROWS, base, model, gain):
        print(f"   {r:24s} {b:7.3f} -> {m:6.3f} cm   (%{g:.1f})")


if __name__ == "__main__":
    main()