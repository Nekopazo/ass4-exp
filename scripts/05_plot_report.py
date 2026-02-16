#!/usr/bin/env python3
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import ensure_dir
from src.config import PLOTS_DIR, REPORTS_DIR


def load_metrics():
    rows = []
    with (REPORTS_DIR / "metrics.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def barplot(rows, key: str, title: str, out_name: str):
    labels = [r["mode"] for r in rows]
    vals = [float(r[key]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel(key)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / out_name, dpi=200)
    plt.close(fig)


def detection_matrix_plot(row: dict, out_name: str):
    # Detection-style 2x2 confusion matrix (Predicted x True):
    # [[TP, FP],
    #  [FN, TN(NA->0)]]
    tp = int(float(row.get("tp", 0)))
    fp = int(float(row.get("fp", row.get("fp_count", 0))))
    fn = int(float(row.get("fn", 0)))
    tn = 0
    mat = np.array([[tp, fp], [fn, tn]], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pothole", "background"], rotation=90)
    ax.set_yticklabels(["pothole", "background"])
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Confusion Matrix - {row['mode']}")
    for i in range(2):
        for j in range(2):
            txt = "-" if (i == 1 and j == 1) else f"{mat[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / out_name, dpi=220)
    plt.close(fig)


def main():
    ensure_dir(PLOTS_DIR)
    rows = load_metrics()
    barplot(rows, "map50_95", "mAP@0.5:0.95 Comparison", "map_comparison.png")
    barplot(rows, "f1", "F1 Comparison", "f1_comparison.png")
    barplot(rows, "fp_count", "FP Count Comparison", "fp_comparison.png")
    for r in rows:
        detection_matrix_plot(r, f"matrix_{r['mode']}.png")
    print("[done] reports/plots generated")


if __name__ == "__main__":
    main()
