#!/usr/bin/env python3
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import ensure_dir, set_seed
from src.config import (
    BEST_MODEL_FILE,
    BG_VALUE,
    CONF_THRES,
    EXTERNAL_FRAMES_DIR,
    HYBRID_DIR,
    IMGSZ,
    IOU_THRES,
    NO_HYBRID_DIR,
    ROI_BOTTOM_LEFT_X_RATIO,
    ROI_BOTTOM_RIGHT_X_RATIO,
    ROI_BOTTOM_Y_RATIO,
    ROI_MID_Y_RATIO,
    ROI_TOP_LEFT_X_RATIO,
    ROI_TOP_RIGHT_X_RATIO,
    ROI_TOP_Y_RATIO,
    SEED,
)
from src.io_yolo import write_yolo_txt


def make_mask(width: int, height: int) -> np.ndarray:
    roi_polygon = np.array(
        [
            # top trapezoid edge
            [int(width * ROI_TOP_LEFT_X_RATIO), int(height * ROI_TOP_Y_RATIO)],
            [int(width * ROI_TOP_RIGHT_X_RATIO), int(height * ROI_TOP_Y_RATIO)],
            # connect to bottom rectangle
            [int(width * ROI_BOTTOM_RIGHT_X_RATIO), int(height * ROI_MID_Y_RATIO)],
            [int(width * ROI_BOTTOM_RIGHT_X_RATIO), int(height * ROI_BOTTOM_Y_RATIO)],
            [int(width * ROI_BOTTOM_LEFT_X_RATIO), int(height * ROI_BOTTOM_Y_RATIO)],
            [int(width * ROI_BOTTOM_LEFT_X_RATIO), int(height * ROI_MID_Y_RATIO)],
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon], 255)

    # Hard crop outside vertical ROI span:
    # everything above ROI_TOP_Y_RATIO and below ROI_BOTTOM_Y_RATIO is removed.
    y_top = int(np.clip(height * ROI_TOP_Y_RATIO, 0, height - 1))
    y_bottom = int(np.clip(height * ROI_BOTTOM_Y_RATIO, 0, height))
    if y_bottom <= y_top:
        y_bottom = min(height, y_top + 1)
    if y_top > 0:
        mask[:y_top, :] = 0
    if y_bottom < height:
        mask[y_bottom:, :] = 0
    return mask


def to_box_dicts(result) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    out = []
    for b, s, c in zip(xyxy, conf, cls):
        out.append({"cls": int(c), "conf": float(s), "xyxy": b.astype(np.float32)})
    return out


def draw_boxes(frame: np.ndarray, boxes: list[dict], color=(0, 255, 0)) -> np.ndarray:
    vis = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b["xyxy"])
        conf = float(b["conf"])
        cls = int(b["cls"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"{cls}:{conf:.2f}",
            (x1, max(y1 - 6, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return vis


def hybrid_filter(boxes: list[dict], mask: np.ndarray, width: int, height: int) -> list[dict]:
    kept = []
    for b in boxes:
        x1, y1, x2, y2 = map(float, b["xyxy"])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        cxi = int(np.clip(cx, 0, width - 1))
        cyi = int(np.clip(cy, 0, height - 1))
        if mask[cyi, cxi] == 0:
            continue
        kept.append(b)
    return kept


def run_mode(model: YOLO, mode_dir: Path, hybrid: bool) -> None:
    img_dir = ensure_dir(mode_dir / "images")
    lbl_dir = ensure_dir(mode_dir / "labels")

    frames = sorted(EXTERNAL_FRAMES_DIR.glob("*.jpg"))
    for i, p in enumerate(frames, start=1):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        if hybrid:
            mask = make_mask(w, h)
            bg = np.full_like(frame, BG_VALUE)
            infer_frame = np.where(mask[..., None] == 255, frame, bg)
            result = model.predict(
                source=infer_frame,
                conf=CONF_THRES,
                iou=IOU_THRES,
                imgsz=IMGSZ,
                verbose=False,
            )[0]
            boxes = hybrid_filter(to_box_dicts(result), mask, w, h)
            vis = draw_boxes(frame, boxes, color=(0, 255, 0))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
        else:
            result = model.predict(
                source=frame,
                conf=CONF_THRES,
                iou=IOU_THRES,
                imgsz=IMGSZ,
                verbose=False,
            )[0]
            boxes = to_box_dicts(result)
            vis = draw_boxes(frame, boxes, color=(0, 255, 0))

        write_yolo_txt(lbl_dir / f"{p.stem}.txt", boxes, w, h)
        cv2.imwrite(str(img_dir / p.name), vis)
        if i % 100 == 0:
            print(f"[{mode_dir.name}] {i}/{len(frames)}")


def main() -> None:
    set_seed(SEED)
    ensure_dir(NO_HYBRID_DIR)
    ensure_dir(HYBRID_DIR)
    model_path = Path(BEST_MODEL_FILE.read_text(encoding="utf-8").strip())
    model = YOLO(str(model_path))
    print(f"[model] {model_path}")
    run_mode(model, NO_HYBRID_DIR, hybrid=False)
    run_mode(model, HYBRID_DIR, hybrid=True)
    print("[done] external inference complete")


if __name__ == "__main__":
    main()
