# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

TRAIN_DIR = '../dataset/stage1_train'
TEST_DIR = '../dataset/stage1_test'

count = len(os.listdir(TRAIN_DIR)) + len(os.listdir(TEST_DIR))
base_img = np.zeros((512, 512, 3))

for data_dir in [TRAIN_DIR, TEST_DIR]:
    for item in os.listdir(data_dir):
        temp_img_path = os.path.join(data_dir, item, 'images',
        '{}.png'.format(item))
        temp_img = transform.resize(plt.imread(temp_img_path)[:, :, :3], (512, 512))
        base_img += temp_img

mean_img = base_img / float(count)
mean_rgb = np.mean(np.mean(mean_img, axis=0), axis=0)
print(mean_rgb * 255)

# output:
# [0.17456407 0.15945011 0.19068251] * 255
# [44.51383877 40.65977746 48.62403917]