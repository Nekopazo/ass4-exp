#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import ensure_dir, write_csv, write_json
from src.config import EXTERNAL_FRAMES_DIR, EXTERNAL_LABELS_DIR, HYBRID_DIR, NO_HYBRID_DIR, REPORTS_DIR
from src.io_yolo import parse_yolo_txt


@dataclass
class EvalResult:
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float
    fp_count: int


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_one_image(preds: list[dict], gts: list[dict], iou_thres: float) -> tuple[int, int, int]:
    preds = sorted(preds, key=lambda x: x.get("conf", 1.0), reverse=True)
    used = [False] * len(gts)
    tp = 0
    fp = 0
    for p in preds:
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gts):
            if used[j]:
                continue
            if int(p["cls"]) != int(g["cls"]):
                continue
            v = iou_xyxy(p["xyxy"], g["xyxy"])
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= iou_thres:
            used[best_j] = True
            tp += 1
        else:
            fp += 1
    fn = sum(1 for u in used if not u)
    return tp, fp, fn


def ap_for_iou(pred_map: dict[str, list[dict]], gt_map: dict[str, list[dict]], iou_thres: float) -> float:
    flat = []
    total_gt = 0
    for k, gts in gt_map.items():
        total_gt += len(gts)
        for p in pred_map.get(k, []):
            flat.append((k, p))
    if total_gt == 0:
        return 0.0
    flat.sort(key=lambda x: x[1].get("conf", 1.0), reverse=True)

    used = {k: [False] * len(v) for k, v in gt_map.items()}
    tp = np.zeros(len(flat), dtype=np.float32)
    fp = np.zeros(len(flat), dtype=np.float32)

    for i, (k, p) in enumerate(flat):
        gts = gt_map.get(k, [])
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gts):
            if used[k][j]:
                continue
            if int(p["cls"]) != int(g["cls"]):
                continue
            v = iou_xyxy(p["xyxy"], g["xyxy"])
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= iou_thres:
            used[k][best_j] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(total_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def load_map(label_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for img_path in sorted(EXTERNAL_FRAMES_DIR.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        txt = label_dir / f"{img_path.stem}.txt"
        out[img_path.stem] = parse_yolo_txt(txt, w, h)
    return out


def eval_mode(pred_dir: Path, gt_dir: Path) -> EvalResult:
    pred_map = load_map(pred_dir)
    gt_map = load_map(gt_dir)

    tp = fp = fn = 0
    for k in gt_map.keys():
        tpi, fpi, fni = match_one_image(pred_map.get(k, []), gt_map.get(k, []), iou_thres=0.5)
        tp += tpi
        fp += fpi
        fn += fni
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    ious = np.arange(0.5, 0.96, 0.05)
    aps = [ap_for_iou(pred_map, gt_map, float(t)) for t in ious]
    map50 = aps[0] if aps else 0.0
    map50_95 = float(np.mean(aps)) if aps else 0.0
    return EvalResult(precision, recall, f1, map50, map50_95, fp)


def main() -> None:
    gt_dir = EXTERNAL_LABELS_DIR
    no_h = eval_mode(NO_HYBRID_DIR / "labels", gt_dir)
    hyb = eval_mode(HYBRID_DIR / "labels", gt_dir)

    result = {
        "no_hybrid": no_h.__dict__,
        "hybrid": hyb.__dict__,
    }
    ensure_dir(REPORTS_DIR)
    write_json(REPORTS_DIR / "metrics.json", result)
    rows = [
        {"mode": "no_hybrid", **no_h.__dict__},
        {"mode": "hybrid", **hyb.__dict__},
    ]
    write_csv(
        REPORTS_DIR / "metrics.csv",
        rows,
        fieldnames=["mode", "precision", "recall", "f1", "map50", "map50_95", "fp_count"],
    )
    print("[done] reports/metrics.json and reports/metrics.csv")


if __name__ == "__main__":
    main()
