Competitions Notes
===================

Kaggle 细胞核识别
-------------------

https://www.kaggle.com/competitions

 `Find the nuclei in divergent images to advance medical discovery` (2 months to go)

 目标要创建一个自动化细胞核检测算法，加快医学研究。官方给出了一些测试集和训练集，问题可以转化为在图像中找到特定的目标？(`这个是机器学习/CV在医疗方面的应用，比较感兴趣，但可能需要医学背景，待讨论`)

 `关于题目更多的背景介绍可以看` `这里 <https://www.kaggle.com/c/data-science-bowl-2018#description>`_。下为Poster:

 |speed-cures|

.. |speed-cures| image:: ../assets/speed-cures.jpg
 :width: 400px
 :align: middle

Notes
>>>>>>>>>>>>>>>>>>>

- Windows 下的Mask-RCNN编译
    - Github repo:
        https://github.com/CharlesShang/FastMaskRCNN
    - How-to:
        1) Go to ``./libs/datasets/pycocotools`` and run ``make``
        2) Download COCO dataset, place it into ``./data``, then run ``python download_and_convert_data.py`` to build tf-records. It takes a while.（要先解压）
        3) Download pretrained resnet50 model, ``wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz``, unzip it, place it into ``./data/pretrained_models/``
        4) Go to ``./libs`` and run ``make``
        5) run ``python train/train.py`` for training

    Windows平台下编译 `1.` 时需要先将 ``FastMaskRCNN\\libs\\datasets\\pycocotools`` 下的 ``setpy.py`` 的 ``-Wno-cpp`` 和 ``-Wno-unused-function`` 编译参数去掉（如下）

    >>> ext_modules = [
        Extension(
        '_mask',
            sources=['./common/maskApi.c', '_mask.pyx'],
            include_dirs = [np.get_include(), './common'],
            extra_compile_args=['-std=c99'],
        )
    ]
