# YOLO 训练与推理实施计划（精简版）

## 1. 固定配置（先锁定，不再改动）

### 步骤 1.1：固定全局训练参数
- 指令：统一使用 `seed=42`、`epochs=200`、`imgsz=800`、`batch=16`、`conf=0.35`、`iou=0.5`。
- 验证测试：检查训练记录中 3 个模型的参数完全一致。
- 通过标准：无任意一个模型出现参数漂移。

### 步骤 1.2：固定 Hybrid ROI 参数（整套流程只用这组）
- 指令：使用归一化梯形 ROI 顶点（按左上、右上、右下、左下顺序）：
  - `(0.48, 0.55)`
  - `(0.52, 0.55)`
  - `(1.00, 0.85)`
  - `(0.00, 0.85)`
  同时固定：
  - ROI 膨胀：`50 px`
  - 内层 ROI 腐蚀：`40 px`
  - 外部填充值：`114`
  - 面积比例阈值：`0.00025 ~ 0.05`
  - 长宽比阈值：`0.30 ~ 3.50`
- 验证测试：抽检 20 张外部帧，确认 ROI 覆盖道路主体且不过度吞噬天空/路肩区域。
- 通过标准：20 张中至少 18 张覆盖合理。

## 2. 数据与目录准备

### 步骤 2.1：确认输入目录
- 指令：训练数据使用 `datasets`；外部推理输入使用 `video-frame/frames`。
- 验证测试：确认 `datasets/train|valid|test` 的 `images` 与 `labels` 都存在，且 `video-frame/frames` 非空。
- 通过标准：全部目录可访问且存在有效文件。

### 步骤 2.2：准备输出目录
- 指令：仅保留必要输出目录：`outputs/runs`、`outputs/external/no_hybrid`、`outputs/external/hybrid`、`reports/plots`。
- 验证测试：检查 4 个目录是否创建成功。
- 通过标准：目录齐全，后续阶段无需再创建新目录。

## 3. 三模型训练与选优

### 步骤 3.1：训练 YOLOv5s
- 指令：用固定参数训练 YOLOv5s，输出到 `outputs/runs/yolov5s`。
- 验证测试：核验 `best.pt`、`results.csv`、`results.png`。
- 通过标准：3 个文件都存在且非空。

### 步骤 3.2：训练 YOLOv8s
- 指令：用同一固定参数训练 YOLOv8s，输出到 `outputs/runs/yolov8s`。
- 验证测试：核验 `best.pt`、`results.csv`、`results.png`。
- 通过标准：3 个文件都存在且非空。

### 步骤 3.3：训练 yolo26s
- 指令：用同一固定参数训练 yolo26s，输出到 `outputs/runs/yolo26s`。
- 验证测试：核验 `best.pt`、`results.csv`、`results.png`。
- 通过标准：3 个文件都存在且非空。

### 步骤 3.4：选最优模型
- 指令：只按验证集 `mAP50-95` 选最大值模型，把对应权重路径写入 `outputs/best_model.txt`。
- 验证测试：读取 `best_model.txt`，确认路径指向真实存在的 `best.pt`。
- 通过标准：路径有效且与最高 `mAP50-95` 模型一致。

## 4. 外部推理（No-Hybrid 与 Hybrid）

### 步骤 4.1：No-Hybrid 全量推理
- 指令：使用最佳模型对 `video-frame/frames` 做标准推理，输出到 `outputs/external/no_hybrid`。
- 验证测试：对比输入帧数与输出预测文件数。
- 通过标准：覆盖率达到 100%（或有明确失败清单且失败率小于 1%）。

### 步骤 4.2：Hybrid 全量推理（固定 ROI）
- 指令：按固定 ROI 参数执行：ROI 生成 → 膨胀 50px → 内层腐蚀 40px → ROI 外填充 114 → YOLO 推理 → 中心点过滤 → 面积比过滤 → 长宽比过滤，输出到 `outputs/external/hybrid`。
- 验证测试：抽检 30 帧，确认每帧都执行了全部过滤环节且输出文件命名与 No-Hybrid 可一一对应。
- 通过标准：30 帧全部通过流程检查，无步骤漏跑。

## 5. 外部评估与报告

### 步骤 5.1：评估两种模式
- 指令：用同一套真值标签，分别评估 No-Hybrid 与 Hybrid，指标仅保留：Precision、Recall、F1、mAP@0.5、mAP@0.5:0.95、FP count。
- 验证测试：检查两组评估使用的样本集合与评价阈值完全一致。
- 通过标准：两组结果可直接横向比较。

### 步骤 5.2：输出指标文件
- 指令：将最终结果保存为 `reports/metrics.json` 和 `reports/metrics.csv`。
- 验证测试：核对两文件字段名一致、数值一致。
- 通过标准：JSON 与 CSV 可互相校验，无缺字段。

### 步骤 5.3：输出最小图表集
- 指令：只生成 3 张图：mAP 对比、F1 对比、FP 对比，保存到 `reports/plots`。
- 验证测试：逐图核对数值来源与 `metrics` 文件一致。
- 通过标准：3 张图全部存在且数值正确。

## 6. 最终验收（精简版）

### 步骤 6.1：交付检查
- 指令：最终仅检查 6 项：`best.pt`（最佳模型）、`best_model.txt`、No-Hybrid 输出目录、Hybrid 输出目录、`reports/metrics.*`、`reports/plots`。
- 验证测试：逐项点验存在性和可读性。
- 通过标准：6 项全部通过。
