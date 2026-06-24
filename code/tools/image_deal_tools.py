import numpy as np
import cv2
from matplotlib import pyplot as plt
def gamma_correction(image, gamma):#对彩色图像做gamma校正
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)