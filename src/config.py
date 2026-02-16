from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = ROOT / "datasets" / "data.yaml"
EXTERNAL_FRAMES_DIR = ROOT / "video-frame" / "frames"
EXTERNAL_LABELS_DIR = ROOT / "video-frame" / "output"

OUTPUTS_DIR = ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
BEST_MODEL_FILE = OUTPUTS_DIR / "best_model.txt"
EXTERNAL_DIR = OUTPUTS_DIR / "external"
NO_HYBRID_DIR = EXTERNAL_DIR / "no_hybrid"
HYBRID_DIR = EXTERNAL_DIR / "hybrid"

REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

SEED = 42

EPOCHS = 200
IMGSZ = 640
BATCH = 16
CONF_THRES = 0.278
IOU_THRES = 0.5
PATIENCE = 30

TRAIN_MODELS = [
    ("yolov5s", "yolov5su.pt"),
    ("yolov8s", "yolov8s.pt"),
    ("yolo26s", "yolo26s.pt"),
]

# Fixed ROI parameters from user-provided notebook code.
ROI_TOP_Y_RATIO = 0.45
ROI_TOP_LEFT_X_RATIO = 0.45
ROI_TOP_RIGHT_X_RATIO = 0.55
# Hex ROI (top trapezoid + bottom rectangle).
ROI_MID_Y_RATIO = 0.7
ROI_BOTTOM_Y_RATIO = 0.9
ROI_BOTTOM_LEFT_X_RATIO = 0.0
ROI_BOTTOM_RIGHT_X_RATIO = 1.0
ROI_DILATE_PX = 50
INNER_ERODE_PX = 40
BG_VALUE = 114

MIN_AREA_RATIO = 0.00025
MAX_AREA_RATIO = 0.05
MIN_ASPECT = 0.30
MAX_ASPECT = 3.50
