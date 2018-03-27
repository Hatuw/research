Notes Details
===================

`有关Kaggle的细胞核识别的详细笔记`

Introduction
>>>>>>>>>>>>>>>>>>>

Related Work
>>>>>>>>>>>>>>>>>>>

- Nature上一篇有关用DL来判断细胞是否是活细胞的工作.
    (https://www.nature.com/articles/d41586-018-02174-z)

    - 该领域对ground truth data要求高,为了规避这一挑战,researchers更加希望通过 **小样本训练** 或者 **Transfer Learning** .(例如将锯齿动物细胞的模型迁移到人类细胞图像识别中.)
    - "Another challenge with deep learning is that the computers are both unintelligent and lazy, notes Michelle Dimon, a research scientist at Google Accelerated Science. They lack the judgement to distinguish biologically relevant differences from normal variation."


Mixed
>>>>>>>>>>>>>>>>>>>

- 貌似生物领域的话是挂到bioRxiv(preprint).


HyperPara
>>>>>>>>>>>>>>>>>>>

- 训练集的Mean RGB为 ``MEAN_PIXEL = np.array([44.5, 40.7, 48.6])``

- 当输入为512*512时, resnet_graph对应的C1-C5大小为:

    .. code::

        C1 = (?, 128, 128, 64)
        C2 = (?, 128, 128, 256)
        C3 = (?, 64, 64, 512)
        C4 = (?, 32, 32, 1024)
        C5 = (?, 16, 16, 2048)

- 两次实验结果相对来说比较好的参数:

    .. code::

        class ShapesConfig(Config):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 2
            NUM_CLASSES = 1 + 1
            IMAGE_MIN_DIM = 512
            IMAGE_MAX_DIM = 512
            USE_MINI_MASK = True
            MINI_MASK_SHAPE = (56, 56)
            RPN_NMS_THRESHOLD = 0.5
            RPN_TRAIN_ANCHORS_PER_IMAGE = 320   # 256
            POST_NMS_ROIS_TRAINING = 2000
            POST_NMS_ROIS_INFERENCE = 2000
            MEAN_PIXEL = np.array([44.5, 40.7, 48.6])
            MAX_GT_INSTANCES = 256
            DETECTION_MAX_INSTANCES = 400
            RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
            TRAIN_ROIS_PER_IMAGE = 512
            STEPS_PER_EPOCH = 300
            VALIDATION_STEPS = 70
        # Training mAP: 0.9714530196558473
        # Validating mAP: 0.960999267232544
        # LB: 0.355
        # `shapes20180314T1748`

    .. code::

        class ShapesConfig(Config):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 2
            NUM_CLASSES = 1 + 1
            IMAGE_MIN_DIM = 512
            IMAGE_MAX_DIM = 512
            USE_MINI_MASK = True
            MINI_MASK_SHAPE = (56, 56)
            RPN_NMS_THRESHOLD = 0.7
            RPN_TRAIN_ANCHORS_PER_IMAGE = 320   # 256
            POST_NMS_ROIS_TRAINING = 2000
            POST_NMS_ROIS_INFERENCE = 2000
            MEAN_PIXEL = np.array([44.5, 40.7, 48.6])
            MAX_GT_INSTANCES = 256
            DETECTION_MAX_INSTANCES = 400
            RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
            TRAIN_ROIS_PER_IMAGE = 512
            STEPS_PER_EPOCH = 1000
            VALIDATION_STEPS = 70
        # 最后增加了所有层的训练, LEARNING_RATE/20=0.5*10^(-5), epoches=2
        # Training mAP: 1.0
        # Validating mAP: 0.9386724366082086
        # LB: 0.381
        # `shapes20180325T2211`

