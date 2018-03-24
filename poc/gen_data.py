# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np

def data_augmentation(input_image, masks,
                      h_flip=True,
                      v_flip=True,
                      rotation=90, # 360
                      zoom=1.2, # 1.5
                      brightness=0, # 0.5
                      crop=False):
    # first is input all other are output
    # Data augmentation
    output_image = input_image.copy()
    output_masks = masks.copy()

    # random flip
    if h_flip and random.randint(0, 1):
        output_image = np.fliplr(output_image)
        output_masks = np.fliplr(output_masks)

    if v_flip and random.randint(0, 1):
        output_image = np.flipud(output_image)
        output_masks = np.flipud(output_masks)

    factor = 1.0 + abs(random.gauss(mu=0.0, sigma=brightness))
    if random.randint(0, 1):
        factor = 1.0 / factor
    table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    output_image = cv2.LUT(output_image, table)
    if rotation:
        rotate_times = random.randint(0, rotation/90)
    else:
        rotate_times = 0.0
    for r in range(0, rotate_times):
        output_image = np.rot90(output_image)
        output_masks = np.rot90(output_masks)

    return output_image, output_masks