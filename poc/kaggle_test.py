# -*- coding: utf-8 -*-
import pathlib
import imageio
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
import pandas as pd

# Glob the training data and load a single image path
training_paths = pathlib.Path('../dataset/stage1_train').glob('*/images/*.png')
training_sorted = sorted([x for x in training_paths])
im_path = training_sorted[45]
im = imageio.imread(str(im_path))

# Print the image dimensions
print('Original image shape: {}'.format(im.shape))
# Coerce the image into grayscale format (if not already)
im_gray = rgb2gray(im)
print('New image shape: {}'.format(im_gray.shape))


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > (prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])

# print('RLE Encoding for the current mask is: {}'.format(rle_encoding(label_mask)))