# -*- coding: utf-8 -*-
"""
Data Augmentation
"""
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to load source nuclei dataset(training)
DATA_DIR = os.path.join(ROOT_DIR, "../dataset/stage1_train")

# print(os.listdir(DATA_DIR))

data_gen_args = dict(rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='reflect')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

TEST_ID = '00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e'
IMG_PATH = os.path.join(DATA_DIR, TEST_ID, 'images')
# test_img = os.path.join(DATA_DIR, test_id, 'images', '{}.png'.format(test_id))
MASK_PATH = os.path.join(DATA_DIR, TEST_ID, 'masks')

# mask_ids = os.listdir(mask_path)
# total_mask = np.zeros(plt.imread(os.path.join(mask_path, mask_ids[0])).shape)
# for mask_id in mask_ids:
#     total_mask += plt.imread(os.path.join(mask_path, mask_id))

"""
y = img_to_array(total_mask)
# print(y.shape)
y = y.reshape((1,) + y.shape)
# print(y.shape)

img = load_img(test_img)  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# print(x.shape)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
# print(x.shape)
"""

seed = 1
# image_datagen.fit(x, augment=True, seed=seed)
# mask_datagen.fit(y, augment=True, seed=seed)

# source
###############
# image_generator = \
# image_datagen.flow_from_directory(IMG_PATH,
#                                 save_to_dir='augmentation',
#                                 class_mode=None,
#                                 save_prefix='image', save_format='png',
#                                 seed=seed)

img = load_img(os.path.join(IMG_PATH, TEST_ID+'.png'))  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
image_generator = \
image_datagen.flow(x, batch_size=1,
                    save_to_dir='augmentation',
                    save_prefix='image', save_format='png',
                    seed=seed)

# mask
###############
generator = [image_generator]
for mask in os.listdir(MASK_PATH):
    temp_mask_path = os.path.join(MASK_PATH, mask)
    temp_mask = plt.imread(os.path.join(MASK_PATH, mask))
    temp_mask = img_to_array(temp_mask)
    temp_mask = temp_mask.reshape((1,) + temp_mask.shape)
    temp_gen = copy.copy(mask_datagen)
    temp_mask_gen = temp_gen.flow(temp_mask, batch_size=1,
                                save_to_dir='augmentation',
                                save_prefix='mask', save_format='png',
                                seed=seed)
    del temp_gen
    generator.append(temp_mask_gen)

# mask_generator = \
# mask_datagen.flow_from_directory(mask_path,
#                                 save_to_dir='augmentation',
#                                 save_prefix='mask', save_format='png',
#                                 seed=seed)

# mask_generator = mask_datagen.flow(y, batch_size=1,
#                                     save_to_dir='augmentation',
#                                     save_prefix='mask', save_format='png',
#                                     seed=seed)

train_generator = zip(generator)

# train_generator = zip(image_generator, mask_generator)
# train_generator = zip(image_generator)
# print(list(train_generator))
i = 0
# for batch_x, batch_y in train_generator:
for _ in zip(*generator):
    i += 1
    if i > 1:
        break  # otherwise the generator would loop indefinitely