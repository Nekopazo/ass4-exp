#!/usr/bin/env python3
from pathlib import Path
import sys

from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import ensure_dir, set_seed
from src.config import BATCH, DATA_YAML, EPOCHS, IMGSZ, RUNS_DIR, SEED, TRAIN_MODELS, PATIENCE


def main() -> None:
    set_seed(SEED)
    ensure_dir(RUNS_DIR)
    for model_name, model_weight in TRAIN_MODELS:
        print(f"[train] {model_name} <- {model_weight}")
        model = YOLO(model_weight)
        model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            seed=SEED,
            deterministic=True,
            project=str(RUNS_DIR),
            name=model_name,
            exist_ok=True,
            verbose=True,
            patience=PATIENCE
        )
        print(f"[done] outputs: {RUNS_DIR / model_name}")


if __name__ == "__main__":
    main()
