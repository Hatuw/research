Competitions Notes
===================

Kaggle 细胞核识别
-------------------

:doc:`details` :一些Introduction, Related work...

Introduction
>>>>>>>>>>>>>>>>>>>

https://www.kaggle.com/competitions

`Find the nuclei in divergent images to advance medical discovery` (2 months to go)

    目标要创建一个自动化细胞核检测算法，加快医学研究。官方给出了一些测试集和训练集，问题可以转化为在图像中找到特定的目标？

    部分images数据如下：

    |nuclei-demo|

    .. |nuclei-demo| image:: ../assets/demo_nuclei.png
        :width: 500px
        :align: middle

    (mask为细胞核的mask二值图)

    `关于题目更多的背景介绍可以看` `这里 <https://www.kaggle.com/c/data-science-bowl-2018#description>`_。

..
    |speed-cures|
    .. |speed-cures| image:: ../assets/speed-cures.jpg
        :width: 400px
        :align: middle

TODO
>>>>>>>>>>>>>>>>>>>

+--------------------------+-------------+--------------------+
|           task           |   完成情况  |    comment         |
+==========================+=============+====================+
|     mask-rcnn环境搭建    |      ✔      |                    |
+--------------------------+-------------+--------------------+
|     mask-rcnn training   |      ✔      |                    |
+--------------------------+-------------+--------------------+
|            调参          |   进行中    |                    |
+--------------------------+-------------+--------------------+
|    Data Augmentation     |   进行中    |                    |
+--------------------------+-------------+--------------------+

.. todo::
    - [✔] 计算图像平均RGB值，并重新训练
    - [✔] Data Augmentation(需要解决images和多张masks不匹配问题)
    - [x] 修改最后两层的初始化方法(xavier)
    - [x] 修改网络结构，如删除最后两层再进行训练

关于Data Augmentation的可以参考这个:

http://keras-cn.readthedocs.io/en/latest/preprocessing/image/


Notes
>>>>>>>>>>>>>>>>>>>

- U-Net

    `U-Net: Convolutional Networks for Biomedical Image Segmentation <https://arxiv.org/pdf/1505.04597.pdf>`_

    |U-Net|
    
    .. |U-Net| image:: ../assets/U-Net.png
        :width: 400px
        :align: middle

    一个做医学图像分割的网络，数据集是International Symposium on Biomedical Imaging (ISBI)的 workshop 比赛。Kaggle上有个Kernel实现这个U-Net：

        https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277

- Windows 下的Mask-RCNN编译
    |mask-rcnn|
        
    .. |mask-rcnn| image:: ../assets/mask-rcnn.png
        :width: 400px
        :align: middle

    - Github repo:
        https://github.com/matterport/Mask_RCNN (亲测可行)

    |demo-mask-rcnn|

    .. |demo-mask-rcnn| image:: ../assets/demo-mask-rcnn.png
        :width: 700px
        :align: middle

    - CSDN上有个这个repo的踩坑记录(http://blog.csdn.net/u011974639/article/details/78483779?locationNum=9&fps=1)

    - Github上有个Mask-RCN的细胞核检测Baseline(https://github.com/killthekitten/kaggle-ds-bowl-2018-baseline)

    - Kaggle上面有个kernel用Mask-RCNN加上其他一些trick的,目前排名第七(https://www.kaggle.com/c/data-science-bowl-2018/discussion/50795)

    ``Mask_RCNN/model.py`` 是Mask-RCNN的 **resnet101** 实现； ``Mask_RCNN/train_shapes.ipynb`` 是用自己数据集训练Mask_RCNN的一个demo，其中 ``ShapesDataset`` 类下的 ``load_image()`` 、 ``load_mask()`` 、``image_reference()`` 方法需要重写以向外提供数据。 ``poc/train_nuclei.py`` 就是将此project应用于检测细胞核的尝试。

- Data Augmentation

    由于数据集太小，在此对Data做Agumentation以扩大Dataset