# import os
# import torch
# from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
# from tqdm import tqdm
#
# from utils.utils import get_lr
# from utils.utils_metrics import f_score
# import cv2
# import numpy as np
# from PIL import Image
# from torch.utils.data.dataset import Dataset
#
# from utils.utils import cvtColor, preprocess_input
#
# img  = Image.open('H:/code/segmentation/SUIM/TRAIN/masks8/A_001.bmp')
# # img.show()
# img    = np.transpose(preprocess_input(np.array(img, np.float64)), [2,0,1])
# imgs    = torch.from_numpy(img).type(torch.FloatTensor)


import matplotlib.pyplot as plt
import pylab
import cv2
import numpy as np

img = plt.imread("H:/code/segmentation/SUIM/TRAIN/masks8/A_001.bmp")                        #在这里读取图片

# plt.imshow(img)                                     #显示读取的图片
# pylab.show()

fil = np.array([[ 1,-1, 0],                        #这个是设置的滤波，也就是卷积核
                [ -1, 0, 1],
                [  0, 1, -1]])

res = cv2.filter2D(img,-1,fil)                      #使用opencv的卷积函数

plt.imshow(res)                                     #显示卷积后的图片
plt.imsave("res.jpg",res)
pylab.show()