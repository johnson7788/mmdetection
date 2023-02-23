#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv


config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
checkpoint_file = 'cosmetic_exps/latest.pth'
assert os.path.exists(checkpoint_file), f"checkpoint_file文件不存在，请检查: {checkpoint_file}"


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# In[4]:


# test a single image
img = 'cosmetic/train/images/809848abdd9102af586c531cfcc8a69a.jpg'
result = inference_detector(model, img)


# In[5]:


# show the results
show_result_pyplot(model, img, result, score_thr=0.6, out_file="demo_res.png")

