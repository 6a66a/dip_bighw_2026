# Dataset Processing Log

> 记录时间：2026-06-14  
> 注意：本文记录的是当前工作区中的数据集处理结果。若后续重新运行脚本、替换源数据、修改阈值或改动划分方式，下面的数量、阈值和文件地址都可能改变。

## 1. 原始 YOLO 数据集

源数据集：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_data`

结构与数量：

| Split | Images | Labels |
| --- | ---: | ---: |
| train | 419 | 419 |
| val | 87 | 87 |
| test | 132 | 132 |

对应配置文件：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new.yaml`

## 2. HSI 强度减 0.3 数据集

处理含义：

将 RGB 图像转为 HSI，取其中 `I` 即 intensity，对 normalized intensity 执行 `I - 0.3`，再转回 RGB。

输出数据集：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_data_0.3`

对应配置文件：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_0.3.yaml`

当前结构：

| Split | Images | Labels |
| --- | ---: | ---: |
| train | 419 | 419 |
| val | 87 | 87 |
| test | 132 | 132 |

生成脚本：

`D:\Pythonfiles\dip\Project\Big_hw\tools\rebuild_new_data_variants.py`

该脚本复用：

`D:\Pythonfiles\dip\Project\Big_hw\tools\darken_test_hsi.py`

## 3. HSI 强度减 0.3 + Gamma 0.8 数据集

处理含义：

先基于 `new_data_0.3`，再对全部 `train/val/test` 图像执行 gamma 变换，`gamma=0.8`。

输出数据集：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_data_0.3_gamma_0.8`

对应配置文件：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_0.3_gamma_0.8.yaml`

当前结构：

| Split | Images | Labels |
| --- | ---: | ---: |
| train | 419 | 419 |
| val | 87 | 87 |
| test | 132 | 132 |

生成脚本：

`D:\Pythonfiles\dip\Project\Big_hw\tools\rebuild_new_data_variants.py`

该脚本复用：

`D:\Pythonfiles\dip\Project\Big_hw\tools\image_deal_tools.py`

## 4. HSI-I 均值最低 25% 暗图子集

筛选依据：

对 `new_data` 中全部 638 张图片计算 HSI 中 `I` 强度均值，按均值从小到大排序，取最低 25%。

当前筛选结果：

- 总图片数：638
- 选中图片数：160
- 当前阈值：`HSI-I mean <= 0.33353803`

输出数据集：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_data_hsi_i_darkest_25`

对应配置文件：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_hsi_i_darkest_25.yaml`

排序报告：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_data_hsi_i_darkest_25\hsi_i_darkest_25_report.csv`

当前结构：

| Split | Images | Labels |
| --- | ---: | ---: |
| train | 104 | 104 |
| val | 14 | 14 |
| test | 42 | 42 |

生成脚本：

`D:\Pythonfiles\dip\Project\Big_hw\tools\select_darkest_hsi_i_subset.py`

已核对：

- 图片与标签同名对应。
- 缺失标签数为 `0`。
- 多余标签数为 `0`。

## 5. 训练脚本相关说明

训练脚本：

`D:\Pythonfiles\dip\Project\Big_hw\scripts\run_yolo_script.bat`

该脚本已增加训练后测试集评估逻辑：训练成功后会调用 YOLOv5 的 `val.py --task test`，在对应 run 目录下生成测试集指标图。

注意：

- 若要训练不同数据集，需要修改脚本中的 `DATA_YAML`。
- `new_0.3.yaml`、`new_0.3_gamma_0.8.yaml`、`new_hsi_i_darkest_25.yaml` 当前都包含 `train/val/test` 三个 split。
- YOLOv5 训练时可能重新生成 `.cache` 文件，这属于正常行为。

## 6. Gap20 Clean 数据集的 Gamma 2 + Gaussian Noise 暗环境模拟

初始记录：2026-06-14；链路恢复与验证：2026-06-24。

注意：本节记录的是当前工作区中已生成的数据版本。若后续重新运行脚本、替换源数据、修改随机种子、修改 gamma/sigma 或重新划分数据集，下面的数量、阈值、均值和文件地址都可能改变。

本次源数据集只使用：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_data_gap20_valtest_clean`

没有使用旧的派生数据集，例如：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_data_gap20_valtest_clean_i_gamma1p5_sigma0p03`

处理含义：

- 将 RGB 图像转换为 HSI。
- 对 normalized intensity `I` 执行 `I' = I^gamma + N(0, sigma^2)`。
- 本次参数为 `gamma = 2.0`，`sigma = 0.02`。
- 这里的 `sigma = 0.02` 是在 `I` 范围为 `0-1` 时使用的高斯噪声标准差。
- 处理后将 HSI 转回 RGB，并保留 YOLO 标签结构。

生成脚本：

`D:\Pythonfiles\dip\Project\Big_hw\code\build_i_gamma2_sigma0p02_from_gap20_clean.py`

随机种子：

`20260614`

### 6.1 全量暗环境模拟数据集

输出数据集：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_data_gap20_valtest_clean_i_gamma2_sigma0p02`

对应配置文件：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_gap20_valtest_clean_i_gamma2_sigma0p02.yaml`

当前结构：

| Split | Images | Labels |
| --- | ---: | ---: |
| train | 397 | 397 |
| val | 77 | 77 |
| test | 132 | 132 |

核验结果：

- 图片和标签数量一致。
- 平均 HSI-I 从源数据约 `0.41334248` 变为处理后约 `0.20409195`，符合 gamma=2 暗化效果。

### 6.2 基于源数据亮度最低 25% 的暗环境模拟子集

筛选依据：

对 `new_data_gap20_valtest_clean` 中全部 `606` 张源图像计算原始 HSI-I 均值，按均值从低到高排序，取前 `ceil(606 * 0.25) = 152` 张。注意：筛选依据是源 clean 数据的原始亮度，不是 gamma/noise 处理后的亮度。

当前阈值：

`source HSI-I mean <= 0.33337256`

输出数据集：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_data_gap20_valtest_clean_i_gamma2_sigma0p02_source_darkest25`

对应配置文件：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_gap20_valtest_clean_i_gamma2_sigma0p02_source_darkest25.yaml`

当前结构：

| Split | Images | Labels |
| --- | ---: | ---: |
| train | 106 | 106 |
| val | 16 | 16 |
| test | 30 | 30 |

报告文件：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_data_gap20_valtest_clean_i_gamma2_sigma0p02_source_darkest25\i_gamma2_sigma0p02_source_darkest25_summary.txt`

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_data_gap20_valtest_clean_i_gamma2_sigma0p02_source_darkest25\i_gamma2_sigma0p02_source_darkest25_report.csv`

核验结果：

- 子集总数为 `152 / 606`。
- CSV 中 `selected_darkest25_from_source=True` 的样本 rank 为 `1-152`。
- 第一个未选中样本 rank 为 `153`。
- 图片和标签数量一致。

## 7. Detect/Val 阶段可选图像增强预处理

记录时间：2026-06-14

本节记录近期对 YOLOv5 推理和评估链路做的改动。核心目标是：不生成新数据集，而是在运行 `detect.py` / `val.py` 时通过参数选择是否对输入图像做预处理，默认 `none` 保持原始 YOLOv5 行为，作为对照组。

### 7.1 YOLOv5 路径调整

当前 YOLOv5 工程目录已统一为：

`D:\Pythonfiles\dip\Project\Big_hw\yolov5`

相关脚本已改为调用该目录：

- `D:\Pythonfiles\dip\Project\Big_hw\detect_with_model.py`
- `D:\Pythonfiles\dip\Project\Big_hw\scripts\run_detect.bat`
- `D:\Pythonfiles\dip\Project\Big_hw\scripts\run_detect_human.bat`

当前可用训练权重示例：

`D:\Pythonfiles\dip\Project\Big_hw\runs\train_yolov5s_gap20_valtest_clean\weights\best.pt`

### 7.2 运行时预处理参数

已在以下文件中加入统一的预处理参数链路：

- `D:\Pythonfiles\dip\Project\Big_hw\yolov5\utils\dataloaders.py`
- `D:\Pythonfiles\dip\Project\Big_hw\yolov5\detect.py`
- `D:\Pythonfiles\dip\Project\Big_hw\yolov5\val.py`
- `D:\Pythonfiles\dip\Project\Big_hw\detect_with_model.py`
- `D:\Pythonfiles\dip\Project\Big_hw\scripts\run_detect.bat`
- `D:\Pythonfiles\dip\Project\Big_hw\scripts\run_detect_human.bat`

当前 `--preprocess` 可选值：

| 模式 | 含义 |
| --- | --- |
| `none` | 不做预处理，原始对照组 |
| `gamma` | 对 BGR 图像所有通道做 gamma 变换 |
| `hsi_equalize` | 转 RGB -> HSI，只对 I 强度通道做全局直方图均衡，再转回 BGR |
| `yuv_equalize` | 转 BGR -> YUV，只对 Y 亮度通道做全局直方图均衡，再转回 BGR |
| `yuv_clahe` | 转 BGR -> YUV，只对 Y 亮度通道做 CLAHE 局部自适应直方图均衡，再转回 BGR |
| `gamma_hsi_equalize` | 先做 gamma，再对 HSI-I 做全局直方图均衡 |

相关参数：

- `--preprocess-gamma`：默认 `0.8`，用于 `gamma` 和 `gamma_hsi_equalize`。
- `--preprocess-clahe-clip-limit`：默认 `2.0`，用于 `yuv_clahe`。
- `--preprocess-clahe-tile-size`：默认 `8`，用于 `yuv_clahe`，表示 CLAHE 的方形网格尺寸。

注意：

- `detect.py` 的预处理只作用于普通图片/图片文件夹输入；不主动处理 webcam、stream、screenshot、video。
- `val.py` 已同步支持同样预处理，因此当 `RUN_EVAL=1` 时，检测图和评估指标可以使用同一种预处理。
- 不会改写原始数据集，也不会生成新的数据集。

### 7.3 CLAHE 增强说明

新增的 CLAHE 方法采用 YUV 空间的 Y 通道处理：

`BGR -> YUV -> CLAHE(Y) -> BGR`

选择该方式的原因：

- 只增强亮度通道，尽量减少直接处理 RGB 三通道造成的颜色偏移。
- 相比全局直方图均衡，CLAHE 对局部暗区域更敏感，增强幅度更可控。
- 对水下图像仍需谨慎，`clipLimit` 过大时会放大悬浮物、海底纹理和噪声。

建议初始配置：

```bat
set PREPROCESS=yuv_clahe
set PREPROCESS_CLAHE_CLIP_LIMIT=2.0
set PREPROCESS_CLAHE_TILE_SIZE=8
```

如果噪声或误检明显增多，优先尝试：

```bat
set PREPROCESS_CLAHE_CLIP_LIMIT=1.5
```

### 7.4 当前脚本用法

普通测试脚本：

`D:\Pythonfiles\dip\Project\Big_hw\scripts\run_detect.bat`

human/gamma2 数据测试脚本：

`D:\Pythonfiles\dip\Project\Big_hw\scripts\run_detect_human.bat`

脚本中主要修改项：

- `WEIGHTS`：模型权重路径。
- `SOURCE`：用于 detect 的图片或图片文件夹。
- `RESULT_DIR`：检测图和评估结果输出目录。
- `DATA_YAML`：用于 `val.py` 计算 P/R/mAP 的数据集配置。
- `PREPROCESS`：选择图像预处理方法。
- `PREPROCESS_GAMMA`：gamma 参数。
- `PREPROCESS_CLAHE_CLIP_LIMIT`：CLAHE 对比度限制。
- `PREPROCESS_CLAHE_TILE_SIZE`：CLAHE 网格大小。

### 7.5 对比实验观察

曾对以下两组结果做过一次干净复跑对比：

- 原图：`PREPROCESS=none`
- Gamma 增强：`PREPROCESS=gamma`，`PREPROCESS_GAMMA=0.8`

测试数据：

`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_data_gap20_valtest_clean_hsi_i_darkest25\images\test`

权重：

`D:\Pythonfiles\dip\Project\Big_hw\runs\train_yolov5s_gap20_valtest_clean\weights\best.pt`

干净检测输出：

- `D:\Pythonfiles\dip\Project\Big_hw\runs\detect\analysis_darkest25_fresh\original`
- `D:\Pythonfiles\dip\Project\Big_hw\runs\detect\analysis_darkest25_fresh\gamma0p8`

`val.py` 指标对比：

| 方法 | Images | Instances | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原图 | 30 | 36 | 0.757 | 0.694 | 0.829 | 0.530 |
| gamma=0.8 | 30 | 36 | 0.832 | 0.689 | 0.851 | 0.535 |

观察：

- gamma 后整体 mAP 略有提升，但可视化效果图的主检测框变化不大。
- gamma 后框总数从 `43` 增至 `45`，多出的主要是低置信度框。
- 典型样例中，`obj1054_frame0000035.jpg` 和 `obj1111_frame0000050.jpg` 各多出一个低置信度检测框，结合标签看更像额外误检。

结论：

- 单独在推理阶段做图像增强，不一定稳定提升检测效果。
- 人眼看起来更清楚的图像，不一定更符合模型训练时学到的输入分布。
- 水下图像增强容易同时增强目标和背景噪声，因此误检可能增加。
- 更稳妥的实验路线是：将增强加入训练数据分布，或训练原图 + 增强图混合数据，而不是只在 detect 阶段临时增强 test 图像。

### 7.6 额外工具函数

通用图像处理工具文件：

`D:\Pythonfiles\dip\Project\Big_hw\code\color_enhance_denoise_tools.py`

近期加入/整理的方法包括：

- 彩色图像 gamma 变换。
- HSI-I 直方图均衡。
- YUV-Y 全局直方图均衡。
- YUV-Y CLAHE 局部自适应直方图均衡。
- 高斯降噪。
- 中值滤波。
- 双边滤波。
- 非局部均值降噪。

该工具文件可用于后续离线生成数据集版本，和 YOLOv5 detect/val 阶段的运行时预处理保持方法含义一致。

### 7.7 2026-06-24 链路恢复与验证

检查发现，此前 `detect.py` 曾被覆盖回未扩展预处理参数的官方版本，而 `val.py`、`dataloaders.py` 和调用脚本仍保留部分预处理链路，导致 `run_detect.bat` 传入 `--preprocess` 时出现 `unrecognized arguments` 错误。

现已恢复并核验以下完整链路：

- `detect.py` 接收 `--preprocess`、`--preprocess-gamma`、`--preprocess-clahe-clip-limit` 和 `--preprocess-clahe-tile-size`，并传给图片加载器。
- `dataloaders.py` 统一实现并校验六种模式：`none`、`gamma`、`hsi_equalize`、`yuv_equalize`、`yuv_clahe`、`gamma_hsi_equalize`。
- `val.py` 通过验证集数据加载器执行与 `detect.py` 相同的预处理，保证检测可视化和 P/R/mAP 评估使用一致输入。
- 普通图片/图片文件夹会执行预处理；webcam、stream、screenshot 和 video 保持 YOLOv5 原有行为。

已使用以下配置完成实际验证：

- 权重：`D:\Pythonfiles\dip\Project\Big_hw\runs\train_yolov5s_gap20_valtest_clean\weights\best.pt`
- 数据集：`D:\Pythonfiles\dip\Project\Big_hw\Datasets\new_gap20_valtest_clean_hsi_i_darkest25.yaml`
- split：`test`，共 `30` 张图、`36` 个实例。
- 预处理：`PREPROCESS=yuv_clahe`、`clip limit=2.0`、`tile size=8`。
- 验证环境：CPU、`batch-size=4`、`workers=0`。

本次 `yuv_clahe` 验证结果：

| Images | Instances | P | R | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 36 | 0.748 | 0.861 | 0.834 | 0.518 |

用于链路验证的临时输出目录：

- `D:\Pythonfiles\dip\Project\Big_hw\runs\detect\_codex_wrapper_smoke\yuv_clahe`
- `D:\Pythonfiles\dip\Project\Big_hw\runs\val\_codex_smoke\yuv_clahe`

以上数值用于确认当前代码链路可运行，不应直接替代第 7.5 节的正式对比实验；如需比较方法效果，应固定相同权重、数据集、split、置信度阈值与输出配置，分别重跑各模式。
