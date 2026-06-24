## 小组成员

- 姓名：王仲秋　学号：23339108
- 姓名：陶科宇　学号：23339098
- 姓名：叶朋真　学号：23339134
## 环境配置与安装

首次使用时，先创建 Python 3.10 环境：

```bat
conda create -n YOLOv5s python=3.10 -y
conda activate YOLOv5s
```

然后切换到项目根目录（即本 README 所在目录），安装依赖：

```bat
pip install -r requirements.txt
```

## 关于数据集
下载链接：
通过网盘分享的文件：`Datasets.zip`
链接: `https://pan.baidu.com/s/19ACydzlfxgUoYkssv7-jBg` 提取码: `dvx4`
`Datasets/new_data_gap20_valtest_clean` 是训练用的数据集。

`Datasets/new_data_gap20_valtest_clean_hsi_i_darkest25` 是按层将每个集合中亮度最低的 25% 抽取出来构成的低光照子集。

注：数据集都放在 `Datasets/` 中。其中 `Datasets/trash_ICRA19` 是最原始的数据集，`Datasets/new_data` 是我们从中筛选并重新整理标签得到的数据集。后来发现这个数据集有明显的相邻帧泄露风险，所以我们限制不同集合中的两张图片至少相隔 20 帧，得到 `Datasets/new_data_gap20_valtest_clean`，这就是训练用的数据集。

## 代码放在哪里

处理数据集和图像处理涉及的代码放在 `code/` 目录。

只有运行脚本要用的关键文件 `detect_with_model.py` 放在外面，可能比较乱 `o(╥﹏╥)o`。

## 训练 YOLOv5s

注意先把命令行路径切到当前项目根目录。

运行：

```bat
.\scripts\run_yolo_script.bat
```

训练出来的结果和模型会放在 `runs/` 目录下。

## 使用训练好的模型进行检测

要复现我们的检测结果，需要手动修改 `scripts/run_detect.bat` 中和 `PREPROCESS` 有关的参数，选择使用哪种处理方法。可选的方法在脚本里有注释，包括：

- `none`：不处理。
- `gamma`：做 Gamma 校正。
- `hsi_equalize`：做 HSI 全局直方图均衡。
- `yuv_equalize`：做 YUV 全局直方图均衡。
- `yuv_clahe`：做 YUV CLAHE 局部直方图均衡。

其中：

- `baseline0`：在 `new_data_gap20_valtest_clean` 上检测，不进行图像预处理。
- `baseline1`：在低光照子集上检测，不进行图像预处理（`PREPROCESS=none`）。
- `method`：在低光照子集上先执行某种图像预处理，再进行检测。

在项目根目录运行：

```bat
.\scripts\run_detect.bat
```

训练出来的检测结果会放在 `runs/detect/` 目录下。

`runs/` 中已经有我们跑出来的一些数据；如果想自己复现一下，可以直接清空 `runs/` 再跑。

`DATASET_PROCESSING_LOG.md` 是在完成项目过程中让 AI 帮忙把项目改动总结出来的项目摘要，时效性比较差，仅供参考，主要是帮助我们自己以后快速读懂项目用的。我们跑项目的时候似乎根本用不到 `━━∑(￣□￣*|||━━)`。
