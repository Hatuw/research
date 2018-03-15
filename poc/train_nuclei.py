# -*- coding: utf-8 -*-
import os
import sys
sys.path.append("Mask_RCNN")
import random
import math
import re
import time
import numpy as np
import cv2
import skimage
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to load source nuclei dataset(training)
DATA_DIR = os.path.join(ROOT_DIR, "../dataset/stage1_train")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "Mask_RCNN/mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

TRAINING = True

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
    IMAGES_PER_GPU = 2  # 8

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 3  # background + 3 shapes
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512 # 128
    IMAGE_MAX_DIM = 512 # 128

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.5 # 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 320   # 256

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 2000

    # Image mean (RGB)
    MEAN_PIXEL = np.array([44.5, 40.7, 48.6])

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 256

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 400   # 100

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 512  # 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500   # 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 70   # 5


config = ShapesConfig()
# config.display()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    '''
    load_image()
    load_mask()
    image_reference()
    '''

    def load_shapes(self, count, image_ids):
    # def load_shapes(self, count, height, width, image_ids):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "nuclei")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        if not image_ids:
            image_ids = os.listdir(DATA_DIR)
        for index, item in enumerate(image_ids):
            temp_image_path = "{0}/{1}/images/{1}.png".format(DATA_DIR, item)
            # print(temp_image_path)
            # break
            self.add_image("shapes", image_id=index,
                            kaggle_id=item,
                            path=temp_image_path)


    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image = plt.imread(info['path'])[:,:,:3]    # some image maybe 4 channels, need to fix it
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        count = 1
        kaggle_id = info['kaggle_id']
        mask_dir = "{0}/{1}/masks".format(DATA_DIR, kaggle_id)
        masks_list = os.listdir(mask_dir)

        count = len(masks_list)
        temp_img = skimage.io.imread(mask_dir + "/" + masks_list[0])
        mask = np.zeros([temp_img.shape[0], temp_img.shape[1], len(masks_list)])
        for index, item in enumerate(masks_list):
            temp_mask_path = "{}/{}".format(mask_dir, item)
            mask[:, :, index:index+1] = skimage.io.imread(temp_mask_path)[:, : , np.newaxis]
            # if index == 0:
            #     # mask = plt.imread(temp_mask_path)
            #     mask = skimage.io.imread(temp_mask_path)
            # else:
            #     mask += skimage.io.imread(temp_mask_path)
            #     # mask += plt.imread(temp_mask_path)

        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # class_ids = np.array([1])
        class_ids = np.array([1 for _ in range(0, count, 1)])
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return mask, class_ids.astype(np.int32)

# split_training and testing set
image_ids = os.listdir(DATA_DIR)
np.random.shuffle(image_ids)
dtset_size = len(image_ids)
training_ids = image_ids[:int(dtset_size*0.8)]
testing_ids = image_ids[int(dtset_size*0.2):]

# Training dataset
dataset_train = ShapesDataset()
# dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.load_shapes(len(training_ids), image_ids=training_ids)
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
# dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.load_shapes(len(testing_ids), image_ids=testing_ids)
dataset_val.prepare()


# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 2)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids,
#                                 dataset_train.class_names,
#                                 limit=1)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

if TRAINING:
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=4,
                layers='heads')
    # epochs 1


    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    # learning_rate = 0.0001
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=8,
                layers="all")
    # epochs 2


    # Save weights
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
    # model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
    # model.keras_model.save_weights(model_path)


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

# log("original_image", original_image)
# log("image_meta", image_meta)
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)

# visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
#                             dataset_train.class_names, figsize=(8, 8))


results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
"""
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))
"""

# Compute training mAP and validating map
# !!!The code in this need to be combined
def cal_map(dtset_ids, dtset):
    APs = []
    for image_id in dtset_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dtset, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
        return APs

# training mAP
traning_aps = cal_map(dataset_train.image_ids, dataset_train)
print("Training mAP: ", np.mean(traning_aps))
# validating mAP
validating_aps = cal_map(dataset_val.image_ids, dataset_val)
print("Validating mAP: ", np.mean(validating_aps))