#!/usr/bin/env python3
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt

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


def main():
    ensure_dir(PLOTS_DIR)
    rows = load_metrics()
    barplot(rows, "map50_95", "mAP@0.5:0.95 Comparison", "map_comparison.png")
    barplot(rows, "f1", "F1 Comparison", "f1_comparison.png")
    barplot(rows, "fp_count", "FP Count Comparison", "fp_comparison.png")
    print("[done] reports/plots generated")


if __name__ == "__main__":
    main()
