#!/usr/bin/env python3
from collections import deque
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
    HORIZON_EMA_ALPHA,
    HORIZON_ENABLE,
    HORIZON_GATE_MAX_DELTA_RATIO,
    HORIZON_GATE_MIN_DELTA_RATIO,
    HORIZON_MAX_MISS,
    HORIZON_MAX_SLOPE_DEG,
    HORIZON_MAX_STEP_RATIO,
    HORIZON_MAX_Y_RATIO,
    HORIZON_MEDIAN_WIN,
    HORIZON_TOP_MAX_RATIO,
    HORIZON_TOP_MIN_RATIO,
    HORIZON_MIN_CONF,
    HORIZON_MIN_LINE_LEN_RATIO,
    HORIZON_RESIZE_W,
    HORIZON_TOP_MARGIN_RATIO,
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
    HORIZON_UPDATE_INTERVAL,
)
from src.io_yolo import write_yolo_txt


def make_mask(width: int, height: int, top_y_ratio: float) -> np.ndarray:
    roi_polygon = np.array(
        [
            # top trapezoid edge
            [int(width * ROI_TOP_LEFT_X_RATIO), int(height * top_y_ratio)],
            [int(width * ROI_TOP_RIGHT_X_RATIO), int(height * top_y_ratio)],
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
    # everything above dynamic top_y and below ROI_BOTTOM_Y_RATIO is removed.
    y_top = int(np.clip(height * top_y_ratio, 0, height - 1))
    y_bottom = int(np.clip(height * ROI_BOTTOM_Y_RATIO, 0, height))
    if y_bottom <= y_top:
        y_bottom = min(height, y_top + 1)
    if y_top > 0:
        mask[:y_top, :] = 0
    if y_bottom < height:
        mask[y_bottom:, :] = 0
    return mask


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    idx = np.argsort(values)
    v = values[idx]
    w = weights[idx]
    cdf = np.cumsum(w)
    cutoff = 0.5 * w.sum()
    return float(v[np.searchsorted(cdf, cutoff)])


def estimate_horizon_y(frame: np.ndarray) -> tuple[float, float, bool]:
    h, w = frame.shape[:2]
    if w <= 1 or h <= 1:
        return float(h * ROI_TOP_Y_RATIO), 0.0, False

    scale = HORIZON_RESIZE_W / float(w)
    nw = HORIZON_RESIZE_W
    nh = max(2, int(round(h * scale)))
    img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=max(10, int(nw * HORIZON_MIN_LINE_LEN_RATIO)),
        maxLineGap=20,
    )
    if lines is None or len(lines) == 0:
        return float(h * ROI_TOP_Y_RATIO), 0.0, False

    max_slope = np.tan(np.deg2rad(HORIZON_MAX_SLOPE_DEG))
    max_y = nh * HORIZON_MAX_Y_RATIO
    ys = []
    ws = []

    for ln in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, ln)
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-6:
            continue
        slope = dy / dx
        if abs(slope) > max_slope:
            continue
        y_mid = 0.5 * (y1 + y2)
        if y_mid > max_y:
            continue
        length = float(np.hypot(dx, dy))
        # Mean edge response near the segment midpoint as a weak quality term.
        xm = int(np.clip(round(0.5 * (x1 + x2)), 0, nw - 1))
        ym = int(np.clip(round(y_mid), 0, nh - 1))
        patch = edges[max(0, ym - 2) : min(nh, ym + 3), max(0, xm - 2) : min(nw, xm + 3)]
        edge_strength = float(patch.mean() / 255.0) if patch.size > 0 else 0.0
        weight = length * (0.7 + 0.3 * edge_strength)
        ys.append(y_mid)
        ws.append(weight)

    if not ys:
        return float(h * ROI_TOP_Y_RATIO), 0.0, False

    ys_arr = np.asarray(ys, dtype=np.float32)
    ws_arr = np.asarray(ws, dtype=np.float32)
    y_hat_small = weighted_median(ys_arr, ws_arr)
    y_hat = y_hat_small / scale

    y_std = float(np.std(ys_arr))
    line_count_term = min(1.0, len(ys_arr) / 20.0)
    spread_term = max(0.0, 1.0 - y_std / max(1.0, 0.08 * nh))
    conf = 0.6 * line_count_term + 0.4 * spread_term
    return float(y_hat), float(np.clip(conf, 0.0, 1.0)), True


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


def discover_frame_dirs() -> list[Path]:
    # Backward compatible: single clip path.
    dirs = []
    if EXTERNAL_FRAMES_DIR.exists():
        dirs.append(EXTERNAL_FRAMES_DIR)

    # Multi-clip mode: video-frame1/frames, video-frame2/frames, ...
    root = EXTERNAL_FRAMES_DIR.parent.parent
    for p in sorted(root.glob("video-frame*/frames")):
        if p.exists() and p not in dirs:
            dirs.append(p)
    return dirs


def run_mode_for_clip(model: YOLO, mode_dir: Path, clip_name: str, frames_dir: Path, hybrid: bool) -> None:
    img_dir = ensure_dir(mode_dir / clip_name / "images")
    lbl_dir = ensure_dir(mode_dir / clip_name / "labels")
    pre_dir = ensure_dir(mode_dir / clip_name / "pre_yolo") if hybrid else None

    frames = sorted(frames_dir.glob("*.jpg"))
    y_prev = None
    miss_count = 0
    y_hist = deque(maxlen=HORIZON_MEDIAN_WIN)
    for i, p in enumerate(frames, start=1):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        if hybrid:
            top_ratio = ROI_TOP_Y_RATIO
            if HORIZON_ENABLE:
                if y_prev is None:
                    y_prev = float(h * ROI_TOP_Y_RATIO)

                should_update = (i == 1) or (i % max(1, HORIZON_UPDATE_INTERVAL) == 0)
                if should_update:
                    y_raw, conf, ok = estimate_horizon_y(frame)
                    if ok and conf >= HORIZON_MIN_CONF:
                        y_hist.append(y_raw)
                        y_med = float(np.median(np.asarray(y_hist, dtype=np.float32)))
                        delta_ratio = abs(y_med - y_prev) / max(float(h), 1.0)

                        # Gate 1: deadband (too small changes are treated as noise)
                        # Gate 2: reject unrealistic jumps.
                        gated_in = (
                            delta_ratio >= HORIZON_GATE_MIN_DELTA_RATIO
                            and delta_ratio <= HORIZON_GATE_MAX_DELTA_RATIO
                        )
                        if gated_in:
                            y_ema = HORIZON_EMA_ALPHA * y_med + (1.0 - HORIZON_EMA_ALPHA) * y_prev
                            max_step = HORIZON_MAX_STEP_RATIO * h
                            y_ema = float(np.clip(y_ema, y_prev - max_step, y_prev + max_step))
                            y_prev = y_ema
                            miss_count = 0
                        else:
                            miss_count += 1
                    else:
                        miss_count += 1

                if miss_count > HORIZON_MAX_MISS:
                    y_prev = float(h * ROI_TOP_Y_RATIO)
                    miss_count = 0
                y_top = y_prev + HORIZON_TOP_MARGIN_RATIO * h
                top_ratio = float(
                    np.clip(
                        y_top / h,
                        HORIZON_TOP_MIN_RATIO,
                        min(HORIZON_TOP_MAX_RATIO, ROI_BOTTOM_Y_RATIO - 0.05),
                    )
                )

            mask = make_mask(w, h, top_ratio)
            bg = np.full_like(frame, BG_VALUE)
            infer_frame = np.where(mask[..., None] == 255, frame, bg)
            if pre_dir is not None:
                cv2.imwrite(str(pre_dir / p.name), infer_frame)
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
            if hybrid:
                print(
                    f"[{mode_dir.name}/{clip_name}] {i}/{len(frames)} "
                    f"top_y={top_ratio:.3f} miss={miss_count}"
                )
            else:
                print(f"[{mode_dir.name}/{clip_name}] {i}/{len(frames)}")


def main() -> None:
    set_seed(SEED)
    ensure_dir(NO_HYBRID_DIR)
    ensure_dir(HYBRID_DIR)
    model_path = Path(BEST_MODEL_FILE.read_text(encoding="utf-8").strip())
    model = YOLO(str(model_path))
    print(f"[model] {model_path}")
    frame_dirs = discover_frame_dirs()
    if not frame_dirs:
        raise FileNotFoundError("No clip frames directory found.")
    print("[clips]", ", ".join(str(p.parent.name) for p in frame_dirs))
    for frames_dir in frame_dirs:
        clip_name = frames_dir.parent.name
        run_mode_for_clip(model, NO_HYBRID_DIR, clip_name, frames_dir, hybrid=False)
        run_mode_for_clip(model, HYBRID_DIR, clip_name, frames_dir, hybrid=True)
    print("[done] external inference complete")


if __name__ == "__main__":
    main()
