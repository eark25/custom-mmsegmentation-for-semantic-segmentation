import mmcv
from mmcv import Config

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor

import os.path as osp

import copy

cfg = Config.fromfile('../configs/deeplabv3plus/mydeeplabv3plus.py')

# print(cfg.optimizer.lr)
# import sys
# sys.exit(0)

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# for val loss
if len(cfg.workflow) == 2:
    val_dataset = copy.deepcopy(cfg.data.val)
    val_dataset.pipeline = cfg.val_pipeline
    datasets.append(build_dataset(val_dataset))
    # datasets.append(build_dataset(cfg.data.val))

# Build the detector
model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())