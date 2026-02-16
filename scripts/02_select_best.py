#!/usr/bin/env python3
import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import ensure_dir
from src.config import BEST_MODEL_FILE, RUNS_DIR, TRAIN_MODELS


def read_map5095(results_csv: Path) -> float:
    with results_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty results.csv: {results_csv}")
    last = rows[-1]
    candidates = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95",
        "map50-95",
        "mAP50-95",
    ]
    for k in candidates:
        if k in last:
            return float(last[k])
    for k, v in last.items():
        if "50-95" in k:
            return float(v)
    raise RuntimeError(f"mAP50-95 column not found: {results_csv}")


def main() -> None:
    best = None
    for model_name, _ in TRAIN_MODELS:
        model_dir = RUNS_DIR / model_name
        results_csv = model_dir / "results.csv"
        best_pt = model_dir / "weights" / "best.pt"
        if not results_csv.exists():
            raise FileNotFoundError(results_csv)
        if not best_pt.exists():
            raise FileNotFoundError(best_pt)
        score = read_map5095(results_csv)
        print(f"{model_name}: mAP50-95={score:.6f}")
        row = (score, model_name, best_pt)
        if best is None or row[0] > best[0]:
            best = row

    assert best is not None
    ensure_dir(BEST_MODEL_FILE.parent)
    BEST_MODEL_FILE.write_text(str(best[2]), encoding="utf-8")
    print(f"[best] {best[1]} -> {best[2]}")
    print(f"[write] {BEST_MODEL_FILE}")


if __name__ == "__main__":
    main()
