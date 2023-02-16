#!/usr/bin/env python
# coding: utf-8

# # Instance Segmentation
# 
# In this tutorial, you will learn:
# - the basic structure of Mask R-CNN.
# - to perform inference with a MMDetection detector.
# - to train a new instance segmentation model with a new dataset.
# 
# Let's start!
# 

# ## Install MMDetection

# In[1]:


# Check nvcc version
get_ipython().system('nvcc -V')
# Check GCC version
get_ipython().system('gcc --version')


# In[2]:


# install dependencies: (use cu111 because colab has CUDA 11.1)
get_ipython().system('pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html')

# install mmcv-full thus we could use CUDA operators
get_ipython().system('pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html')

# Install mmdetection
get_ipython().system('rm -rf mmdetection')
get_ipython().system('git clone https://github.com/open-mmlab/mmdetection.git')
get_ipython().run_line_magic('cd', 'mmdetection')

get_ipython().system('pip install -e .')


# In[3]:


# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())


# ## Perform Inference with An MMDetection Detector

# ### A two-stage detector
# 
# In this tutorial, we use Mask R-CNN, a simple two-stage detector as an example.
# 
# The high-level architecture of Mask R-CNN is shown in the following picture. More details can be found in the [paper](https://arxiv.org/abs/1703.06870).
# 
# <img src="https://user-images.githubusercontent.com/40661020/143967081-c2552bed-9af2-46c4-ae44-5b3b74e5679f.png" alt="mask rcnn" align="bottom" />
# 
# Mask R-CNN adds a mask branch based on the original Faster R-CNN. It also uses RoIAlign, a more precise version of RoIPooling for RoI feature extraction to improve the performance.
# 
# 
# 
# 

# In[4]:


get_ipython().system('mkdir checkpoints')
get_ipython().system('wget -c https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth       -O checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth')


# In[5]:


import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

# Choose to use a config and initialize the detector
config = 'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
# Setup a checkpoint file to load
checkpoint = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# Set the device to be used for evaluation
device='cuda:0'

# Load the config
config = mmcv.Config.fromfile(config)
# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = None

# Initialize the detector
model = build_detector(config.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = checkpoint['meta']['CLASSES']

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()


# From the printed model, we will find that the model does consist of the components that we described earlier. It uses ResNet as its CNN backbone, and has a RPN head and RoI Head. 
# The RoI Head includes box head and mask head. In addition, the model has a neural network module, named neck, directly after the CNN backbone. It is a [feature pyramid network (FPN)](https://arxiv.org/abs/1612.03144) for enhancing the multi-scale features.
# 
# 
# ### Inference with the detector
# 
# The model is successfully created and loaded, let's see how good it is. We use the high-level API `inference_detector` implemented in the MMDetection. This API is created to ease the inference process. The details of the codes can be found [here](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/apis/inference.py#L15).

# In[6]:


# Use the detector to do inference
img = 'demo/demo.jpg'
result = inference_detector(model, img)


# In[7]:


# Let's plot the result
show_result_pyplot(model, img, result, score_thr=0.3)


# ## Train a Detector on A Customized Dataset
# 
# To train a new detector, there are usually three things to do:
# 1. Support a new dataset
# 2. Modify the config
# 3. Train a new detector
# 
# 

# ### Support a new dataset
# 
# There are three ways to support a new dataset in MMDetection: 
#   1. Reorganize the dataset into a COCO format
#   2. Reorganize the dataset into a middle format
#   3. Implement a new dataset
# 
# We recommend the first two methods, as they are usually easier than the third.
# 
# In this tutorial, we give an example that converts the data into COCO format because MMDetection **only support evaluating mask AP of dataset in COCO format for now**. Other methods and more advanced usages can be found in the [doc](https://mmdetection.readthedocs.io/en/latest/tutorials/customize_dataset.html).
# 
# First, let's download the [the balloon dataset](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon).

# In[8]:


# download and unzip the data
get_ipython().system('wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip')
get_ipython().system('unzip balloon_dataset.zip > /dev/null')


# In[9]:


# Check the directory structure of the tiny data

# Install tree first
get_ipython().system('apt-get -q install tree')
get_ipython().system('tree balloon')


# In[10]:


# Let's take a look at the dataset image
import mmcv
import matplotlib.pyplot as plt

img = mmcv.imread('balloon/train/10464445726_6f1e3bbe6a_k.jpg')
plt.figure(figsize=(15, 10))
plt.imshow(mmcv.bgr2rgb(img))
plt.show()


# After downloading the data, we need to implement a function to convert the annotation format into the COCO format. Then we can use implemented `COCODataset` to load the data and perform training and evaluation.
# Let's take a look at the annotation json file.
# 
# 

# In[11]:


# Check the label of a single image
annotation = mmcv.load('balloon/train/via_region_data.json')


# In[12]:


# The annotation is a dict, and its values looks like the following
annotation['34020010494_e5cb88e1c4_k.jpg1115004']


# According to the above observation, each single image has a corresponding annotation containing keys `filename` and `regions` that are necessary for training.
# We need to read annotations of each image and convert them into COCO format as below:
# 
# ```python
# {
#     "images": [image],
#     "annotations": [annotation], 
#     "categories": [category]
# }
# 
# 
# image = {
#     "id": int,
#     "width": int,
#     "height": int,
#     "file_name": str,
# }
# 
# annotation = {
#     "id": int,
#     "image_id": int,
#     "category_id": int,
#     "segmentation": RLE or [polygon],
#     "area": float,
#     "bbox": [x,y,width,height],
#     "iscrowd": 0 or 1,
# }
# 
# categories = [{
#     "id": int,
#     "name": str,
#     "supercategory": str,
# }]
# ```
# **Note**: We only list the necessary keys for training, as shown above. For a full COCO format, please see [here](https://cocodataset.org/#format-data).

# In[13]:


import os.path as osp

def convert_balloon_to_coco(ann_file, out_file, image_prefix):
    data_infos = mmcv.load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        bboxes = []
        labels = []
        masks = []
        for _, obj in v['regions'].items():
            assert not obj['region_attributes']
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (
                min(px), min(py), max(px), max(py))


            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id':0, 'name': 'balloon'}])
    mmcv.dump(coco_format_json, out_file)


# In[14]:


convert_balloon_to_coco(
    'balloon/train/via_region_data.json',
    'balloon/train/annotation_coco.json',
    'balloon/train/')
convert_balloon_to_coco(
    'balloon/val/via_region_data.json',
    'balloon/val/annotation_coco.json',
    'balloon/val/')


# ### Modify the config
# 
# In the next step, we need to modify the config for the training.
# To accelerate the process, we finetune a detector using a pre-trained detector.

# In[15]:


from mmcv import Config
cfg = Config.fromfile('./configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py')


# Given a config that trains a Mask R-CNN on COCO dataset, we need to modify some values to use it for training on the balloon dataset.

# In[16]:


from mmdet.apis import set_random_seed

# Modify dataset type and path
cfg.dataset_type = 'COCODataset'

cfg.data.test.ann_file = 'balloon/val/annotation_coco.json'
cfg.data.test.img_prefix = 'balloon/val/'
cfg.data.test.classes = ('balloon',)

cfg.data.train.ann_file = 'balloon/train/annotation_coco.json'
cfg.data.train.img_prefix = 'balloon/train/'
cfg.data.train.classes = ('balloon',)


cfg.data.val.ann_file = 'balloon/val/annotation_coco.json'
cfg.data.val.img_prefix = 'balloon/val/'
cfg.data.val.classes = ('balloon',)

# modify num classes of the model in box head and mask head
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

# We can still the pre-trained Mask RCNN model to obtain a higher performance
cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
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
# Finally, lets initialize the dataset and detector, then train a new detector!

# In[17]:


from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model)

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)


# ### Understand the log
# From the log, we can have a basic understanding on the training process and know how well the detector is trained.
# 
# First, since the dataset we are using is small, we loaded a Mask R-CNN model and finetune it for detection. Because the original Mask R-CNN is trained on COCO dataset that contains 80 classes but KITTI Tiny dataset only have 3 classes. Therefore, the last FC layers of the pre-trained Mask R-CNN for classification and regression have different weight shape and are not used. The pre-trained weights of mask prediction layer `mask_head.conv_logits` also does not matches the current model and is not used due to similar reason.
# 
# Third, after training, the detector is evaluated by the default COCO-style evaluation. The results show that the detector achieves 79.6 bbox AP and 81.5 mask AP on the val dataset, not bad!
# 
#  We can also check the tensorboard to see the curves.

# In[18]:


# load tensorboard in colab
get_ipython().run_line_magic('load_ext', 'tensorboard')

# see curves in tensorboard
get_ipython().run_line_magic('tensorboard', '--logdir ./tutorial_exps')


# ## Test the Trained Detector
# 
# After finetuning the detector, let's visualize the prediction results!

# In[19]:


img = mmcv.imread('balloon/train/7178882742_f090f3ce56_k.jpg')

model.cfg = cfg
result = inference_detector(model, img)
show_result_pyplot(model, img, result)


# ## What to Do Next?
# 
# So far, we have learnt how to test and train Mask R-CNN. To further explore the segmentation task, you could do several other things as shown below:
# 
# - Try cascade methods, e.g., [Cascade Mask R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn) and [HTC](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc) in [MMDetection model zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/model_zoo.md). They are powerful detectors that are ranked high in many benchmarks, e.g., COCO dataset.
# - Try single-stage methods, e.g., [K-Net](https://github.com/ZwwWayne/K-Net) and [Dense-RepPoints](https://github.com/justimyhxu/Dense-RepPoints). These two algorithms are based on MMDetection. Box-free instance segmentation is a new trend in the instance segmentation community.
# - Try semantic segmentation. Semantic segmentation is also a popular task with wide applications. You can explore [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/); we also provide a [colab tutorial](https://github.com/open-mmlab/mmsegmentation/blob/master/demo/MMSegmentation_Tutorial.ipynb) for semantic segmentation using MMSegmentation.
# 
