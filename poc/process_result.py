"""
Kaggle nuclei detect
Detecting nuclei and processing results.
"""
# -*- coding:utf-8 -*-
import os
import sys
sys.path.append("Mask_RCNN")
import random
import numpy as np
import pandas as pd
import skimage
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import label

from config import Config
import utils
import model as modellib
import visualize

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to load source nuclei dataset(training)
DATA_TEST_DIR = os.path.join(ROOT_DIR, "../dataset/stage1_test")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "Mask_RCNN/mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4  # 8

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 3  # background + 3 shapes
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512 # 128

    # Image mean (RGB)
    # MEAN_PIXEL = np.array([44.5, 40.7, 48.6])

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200  # 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100   # 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20   # 5


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # USE_MINI_MASK = False
    # MINI_MASK_SHAPE = (1, 1)

config = InferenceConfig()
# config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
class_names = ['BG', 'nuclei']


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def detect_nuclei(image_id):
    """
    This function is used to detect nuclei by using Mask-RCNN model.
    """

    image = skimage.io.imread(os.path.join(DATA_TEST_DIR, '{0}/images/{0}.png'.format(image_id)))[:, :, :3]

    # print(image.shape)
    # Run detection
    results = model.detect([image], verbose=1)
    # print(results)

    # process result
    # mask = results[0]['masks']
    mask = np.sum(results[0]['masks'], -1)

    # r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], ax=get_ax())
    # exit()

    mask[mask != 0] = 1
    plt.imsave('./output/{}.png'.format(image_id), 1.-mask, cmap='binary')
    return mask

def process_result(mask):
    '''
    This function is used to encode the mask image.
    Stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
        mask: mask image
        return:
            list: encoding list
    '''
    def rle_encoding(x):
        dots = np.where(x.T.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b>prev+1):
                run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths


    def prob_to_rles(x, cutoff=0.5):
        lab_img = label(x > cutoff)
        for i in range(1, lab_img.max() + 1):
                yield rle_encoding(lab_img == i)

    # inner_mask = plt.imread(os.path.join('./output', '{}.png'.format(image_id)))[:, :, 0]
    # plt.imshow(mask, cmap='gray'), plt.axis('off')
    # plt.show()
    encode_result = list(prob_to_rles(mask))
    return encode_result

# split_training and testing set
image_ids = os.listdir(DATA_TEST_DIR)
dtset_size = len(image_ids)

# init
new_test_ids = []
rles = []
# detect and encode result
for index, image_id in enumerate(image_ids):
    mask = detect_nuclei(image_id)
    encode_result = process_result(mask)
    rles.extend(encode_result)
    new_test_ids.extend([image_id] * len(encode_result))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-5.csv', index=False)