## 环境配置与安装
首次使用时，先创建 Python 3.10 的环境：
## bat
conda create -n YOLOv5s python=3.10 -y
conda activate YOLOv5s
##
然后切换到本项目根目录（即本 README 所在的目录）
## bat
pip install -r requirements.txt
==============================================
##
关于数据集
Datasets\new_data_gap20_valtest_clean是训练用的数据集
Datasets\new_data_gap20_valtest_clean_hsi_i_darkest25是按层将每个集中亮度最低的25%抽取出来构成的低光照子集
##
注:数据集全部放在Datasets文件中，其中Datasets\trash_ICRA19是最原始的数据集，Datasets\new_data是我们从中筛选并重新整理标签得到的数据集,由于发现该数据集有明显的相邻帧泄露风险，我们限制处于不同集的两张图片间距至少大于20帧，得到了Datasets\new_data_gap20_valtest_clean，这是训练用的数据集
============================================
##
处理数据集和图像处理的涉及到的代码放到了D:\Pythonfiles\dip\Project\Big_hw\code中
只有用于运行脚本的关键.py文件detect_with_model.py放在了外面
可能比较乱o(╥﹏╥)o
=============================================
##
YOLOv5s训练
(注意要先将命令行的路径切到当前根目录)
运行以下代码：
.\scripts\run_yolo_script.bat
训练出来的结果和模型会放在\runs目录下
=============================================
##
使用训练好的模型做检测
要复现我们的检测结果需要手动修改scripts\run_detect.bat中PREPROCESS相关参数来选择使用那种处理方法，可供选择的方法在对应位置已经写了注释，包括
none不处理               
gamma 做gamma校正              
hsi_equalize 做hsi全局直方图均衡  
yuv_equalize 做yuv全局直方图均衡
yuv_clahe    做yuv clahe局部直方图均衡
等
##
baseline0是对new_data_gap20_valtest_clean检测前不做图像处理
baseline1是对低光照子集检测前不做图像处理（对应PREPROCESS：none）
method是使用各种图像处理方法在检测前处理低光照子集
在命令行运行以下代码：
.\scripts\run_detect.bat
##
我们使用
训练出来的结果会放在\runs\detect目录下
==========================================
runs中已经有我们跑出来的一些数据了，如果你想自己复现一下，可以直接把runs清空再跑
==========================================
DATASET_PROCESSING_LOG.md这是在完成项目过程中用ai帮忙,把我们做的项目改动总结的项目摘要(时效性较差，仅供参考)，主要是帮助我们自己快速读懂项目用的(我们在跑项目的时候似乎根本用不到━━∑(￣□￣*|||━━))
