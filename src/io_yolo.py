from pathlib import Path

import numpy as np


def parse_yolo_txt(path: Path, width: int, height: int) -> list[dict]:
    boxes = []
    if not path.exists():
        return boxes
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) > 5 else 1.0
            bw = w * width
            bh = h * height
            x1 = (xc * width) - bw / 2.0
            y1 = (yc * height) - bh / 2.0
            x2 = x1 + bw
            y2 = y1 + bh
            boxes.append(
                {
                    "cls": cls_id,
                    "conf": conf,
                    "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
                }
            )
    return boxes


def write_yolo_txt(path: Path, boxes: list[dict], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for b in boxes:
            x1, y1, x2, y2 = b["xyxy"]
            conf = float(b.get("conf", 1.0))
            cls_id = int(b.get("cls", 0))
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            xc = x1 + bw / 2.0
            yc = y1 + bh / 2.0
            xc /= width
            yc /= height
            bw /= width
            bh /= height
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.6f}\n")

