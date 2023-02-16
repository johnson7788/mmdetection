#!/usr/bin/env python
# coding: utf-8
# 使用FasterRCNN进行目标检测训练
import copy
import os
from tqdm import tqdm
import mmcv
import numpy as np
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, inference_detector, show_result_pyplot

COSMETIC_CLASSES = []
class_files = os.path.join("cosmetic", "train", "classes.txt")
with open(class_files, "r") as f:
    for line in f:
        COSMETIC_CLASSES.append(line.strip())

@DATASETS.register_module()
class CosmeticsDataset(CustomDataset):
    CLASSES = COSMETIC_CLASSES
    def __init__(self, *args, **kwargs):
        """
        生成类别信息
        """
        class2label = {k: i for i, k in enumerate(self.CLASSES)}
        self.class2label = class2label
        super().__init__(*args, **kwargs)
        print(f"共有标签数量: {len(self.CLASSES)} 个")

    def load_annotations(self, ann_file):
        # 读取标注文件
        image_list = os.listdir(self.ann_file)
        # image_list = image_list[:100]
        print(f"要读取: {len(image_list)} 个标注文件, 读取中")
        data_infos = []
        # convert annotations to middle format
        image_prefix = self.ann_file.replace("labels", "images")
        for image_id in tqdm(image_list,desc="读取"):
            image_id = image_id.replace(".txt","")
            filename = f'{image_prefix}/{image_id}.jpg'
            if not os.path.exists(filename):
                print(f"图片文件: {filename} 不存在，跳过")
                continue
            image = mmcv.imread(filename)
            height, width = image.shape[:2]
    
            data_info = dict(filename=f'{image_id}.jpg', width=width, height=height)
    
            # 加载标注文件
            lines = mmcv.list_from_file(os.path.join(self.ann_file, f'{image_id}.txt'))
    
            content = [line.strip().split(' ') for line in lines]
            bbox_labels = [x[0] for x in content]
            bboxes = [[float(info) for info in x[1:5]] for x in content]
    
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
    
            # filter 'DontCare'
            for bbox_label, bbox in zip(bbox_labels, bboxes):
                gt_labels.append(bbox_label)
                gt_bboxes.append(bbox)
            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long),
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                       dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos

# 修改配置

cfg = Config.fromfile('../configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')


# Given a config that trains a Faster R-CNN on COCO dataset, we need to modify some values to use it for training Faster R-CNN on KITTI dataset. We modify the config of datasets, learning rate schedules, and runtime settings

# Modify dataset type and path
cfg.dataset_type = 'CosmeticsDataset'
cfg.data_root = 'cosmetic/'

cfg.data.train.type = 'CosmeticsDataset'
cfg.data.train.data_root = 'cosmetic/'
cfg.data.train.ann_file = 'train/labels'
cfg.data.train.img_prefix = 'train/images'
cfg.runner.max_epochs = 100
cfg.data.test.type = 'CosmeticsDataset'
cfg.data.test.data_root = 'cosmetic/'
cfg.data.test.ann_file = 'dev/labels'
cfg.data.test.img_prefix = 'dev/images'

cfg.data.val.type = 'CosmeticsDataset'
cfg.data.val.data_root = 'cosmetic/'
cfg.data.val.ann_file = 'dev/labels'
cfg.data.val.img_prefix = 'dev/images'

# 修改类别数量
cfg.model.roi_head.bbox_head.num_classes = len(COSMETIC_CLASSES)
# If we need to finetune a model based on a pre-trained detector, we need to
# use load_from to set the path of checkpoints.
cfg.load_from = '../checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './cosmetic_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.device = 'cuda'
cfg.gpu_ids = range(1)

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]


# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


# ### Train a new detector
# 
# Finally, lets initialize the dataset and detector, then train a new detector! We use the high-level API `train_detector` implemented by MMDetection. This is also used in our training scripts. For details of the implementation, please see [here](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/apis/train.py).



# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)


# ### Understand the log
# From the log, we can have a basic understanding on the training process and know how well the detector is trained.
# 
# First, since the dataset we are using is small, we loaded a pre-trained Faster R-CNN model and fine-tune it for detection. 
# The original Faster R-CNN is trained on COCO dataset that contains 80 classes but KITTI Tiny dataset only have 3 classes. Therefore, the last FC layers of the pre-trained Faster R-CNN for classification and regression have different weight shape and are not used.
# 
# Second, after training, the detector is evaluated by the default VOC-style evaluation. The results show that the detector achieves 58.1 mAP on the val dataset, not bad!
# 
# We can also check the tensorboard to see the curves.

# load tensorboard in colab
# get_ipython().run_line_magic('load_ext', 'tensorboard')

# see curves in tensorboard
# get_ipython().run_line_magic('tensorboard', '--logdir ./tutorial_exps')


# From the tensorboard, we can observe that changes of loss and learning rate. We can see the losses of each branch gradually decrease as the training goes by.
# 
# ## Test the Trained Detector
# 
# After finetuning the detector, let's visualize the prediction results!


img = mmcv.imread('cosmetic/dev/images/fffbb778f15b22a652df7f2672cf8061.jpg')
#
model.cfg = cfg
result = inference_detector(model, img)
show_result_pyplot(model, img, result, out_file="fffbb778f15b22a652df7f2672cf8061_res.png")


# ## What to Do Next?
# 
# So far, we have learnt how to test and train a two-stage detector using MMDetection. To further explore MMDetection, you could do several other things as shown below:
# 
# - Try single-stage detectors, e.g., [RetinaNet](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet) and [SSD](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd) in [MMDetection model zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/model_zoo.md). Single-stage detectors are more commonly used than two-stage detectors in industry.
# - Try anchor-free detectors, e.g., [FCOS](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos) and [RepPoints](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints) in [MMDetection model zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/model_zoo.md). Anchor-free detector is a new trend in the object detection community.
# - Try 3D object detection using [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), also one of the OpenMMLab projects. In MMDetection3D, not only can you try all the methods supported in MMDetection but also some 3D object detectors.
# 
