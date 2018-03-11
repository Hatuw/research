# -*- coding: utf-8 -*-
"""
Data Augmentation
"""
import os
import sys
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

test_id = '00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e'
test_img = os.path.join(DATA_DIR, test_id, 'images', '{}.png'.format(test_id))
mask_path = os.path.join(DATA_DIR, test_id, 'masks')
mask_ids = os.listdir(mask_path)
total_mask = np.zeros(plt.imread(os.path.join(mask_path, mask_ids[0])).shape)
for mask_id in mask_ids:
    total_mask += plt.imread(os.path.join(mask_path, mask_id))

# print(total_mask.min(), total_mask.max(), total_mask.shape)
# plt.imshow(total_mask, cmap='gray')
# plt.show()
y = img_to_array(total_mask)
# print(y.shape)
y = y.reshape((1,) + y.shape)
# print(y.shape)

img = load_img(test_img)  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# print(x.shape)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
# print(x.shape)

seed = 1
image_datagen.fit(x, augment=True, seed=seed)
mask_datagen.fit(y, augment=True, seed=seed)
mask_batch = len(os.listdir(mask_path))
image_generator = image_datagen.flow_from_directory(os.path.join(DATA_DIR, test_id, 'images'),
                                                    save_to_dir='augmentation',
                                                    save_prefix='image', save_format='png',
                                                    seed=seed)
# image_generator = image_datagen.flow(x, batch_size=1,
#                                     save_to_dir='augmentation',
#                                     save_prefix='image', save_format='png',
#                                     seed=seed)
mask_generator = mask_datagen.flow_from_directory(mask_path,
                                                save_to_dir='augmentation',
                                                save_prefix='mask', save_format='png',
                                                seed=seed)
# mask_generator = mask_datagen.flow(y, batch_size=1,
#                                     save_to_dir='augmentation',
#                                     save_prefix='mask', save_format='png',
#                                     seed=seed)

train_generator = zip(image_generator, mask_generator)

i = 0
for batch_x, batch_y in train_generator:
    i += 1
    if i > 0:
        break  # otherwise the generator would loop indefinitely

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
# i = 0
# for batch_x, batch_y in datagen.flow(x, y, batch_size=1,
#                           save_to_dir='augmentation', save_prefix='nuclei', save_format='png'):
#     i += 1
#     if i > 2:
#         break  # otherwise the generator would loop indefinitely