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
DATA_DIR = os.path.join(ROOT_DIR, "../dataset/stage1_train")

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
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 256 # 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 200   # 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10   # 5

config = ShapesConfig()


class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """


    def load_shapes(self, count, image_ids):
    # def load_shapes(self, count, height, width, image_ids):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "nuclei")

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

        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # class_ids = np.array([1])
        class_ids = np.array([1 for _ in range(0, count, 1)])
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return mask, class_ids.astype(np.int32)


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()


def detect_nuclei():
    """
    This function is used to detect nuclei by using Mask-RCNN model.
    """

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

    # process_result(gt_mask[:, :, 0])
    print(dataset_val)
    fig = plt.figure()
    add_mask = np.sum(gt_mask, -1)
    # plt.imsave('./{}.png'.format(image_id), add_mask, cmap='gray')
    plt.subplot(1,2,1),plt.imshow(original_image)
    plt.subplot(1,2,2),plt.imshow(add_mask, cmap='gray')
    plt.show()
    # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
    #                         dataset_train.class_names, figsize=(8, 8))


def process_result(img_list):
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

    # print(list(prob_to_rles(img_list)))


# split_training and testing set
image_ids = os.listdir(DATA_DIR)
np.random.shuffle(image_ids)
dtset_size = len(image_ids)
training_ids = image_ids[:int(dtset_size*0.8)]
testing_ids = image_ids[int(dtset_size*0.8):]

# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(len(training_ids), image_ids=training_ids)
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(len(testing_ids), image_ids=testing_ids)
dataset_val.prepare()

detect_nuclei()