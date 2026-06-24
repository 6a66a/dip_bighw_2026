import kagglehub
import os
# Download latest version
os.environ["KAGGLEHUB_CACHE"] = r"D:\Pythonfiles\dip\Project\Big_hw\Dataset\dataset_original\Plastic_Bottl_and_Bag"
path = kagglehub.dataset_download("aaronvincent6411/litter-detection-plastic-bottle-and-bag")

print("Path to dataset files:", path)