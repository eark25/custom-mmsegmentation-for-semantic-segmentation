import argparse
import warnings
import mmcv
from mmcv import Config

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor

import os.path as osp

import copy

import wandb

'''
learning rate - 1e-4 - 1 ?
batch size - 4/8/16/32
backbone - resnet 18/34/50/101
ignore bg - True/False
crop size - 256/512/1024
keep_ratio resize - True/False
lr_scheduler - poly/none
momentum - 0-1
weight_decay 1e-8/1e-4/1e-2
'''

def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks')
    parser.add_argument('--learning_rate', '-lr', dest='learning_rate', metavar='LR', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--batch_size', '-bs', dest='batch_size', metavar='BS', type=int, default=32, help='Batch size')
    parser.add_argument('--backbone', '-bb', dest='backbone', metavar='BB', type=str, default='r50', help='Backbone')
    parser.add_argument('--ignore_bg', '-ib', dest='ignore_bg', metavar='IB', type=bool, default=False, help='Ignore background')
    parser.add_argument('--crop_size', '-cs', dest='crop_size', metavar='CS', type=int, default=512, help='Crop size')
    parser.add_argument('--keep_ratio', '-kr', dest='keep_ratio', metavar='KR', type=bool, default=False, help='Keep ratio')
    parser.add_argument('--scheduler', '-sch', dest='lr_scheduler', metavar='SCH', type=str, default=None, help='Learning rate scheduler')
    parser.add_argument('--momentum', '-mm', dest='momentum', metavar='MM', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', '-wd', dest='weight_decay', metavar='WD', type=float, default=1e-8, help='Weight decay')
    # what is nargs metavar action choices?
    # nargs 
    # - '+'/'*' multiple
    # - '?' use default single
    # metavar displayed name in -h
    # action

    return parser.parse_args()

def main():
    args = get_args()

    cfg = Config.fromfile('../configs/deeplabv3plus/mydeeplabv3plus.py')

    # print(args.ignore_bg)
    # print(args.keep_ratio)

    if args.learning_rate:
        cfg.optimizer.lr = args.learning_rate
        cfg.lr_config.min_lr = args.learning_rate
    if args.batch_size:
        cfg.optimizer_config.cumulative_iters = int(args.batch_size / cfg.data.samples_per_gpu)
    if args.backbone:
        if args.backbone == 'r50':
            cfg.model.pretrained = 'open-mmlab://resnet50_v1c'
            cfg.model.backbone.depth = 50
        if args.backbone == 'r101':
            cfg.model.pretrained = 'open-mmlab://resnet101_v1c'
            cfg.model.backbone.depth = 101
    if args.ignore_bg:
        cfg.model.decode_head.ignore_index = 0
        cfg.model.decode_head.loss_decode[0].avg_non_ignore = True
        cfg.model.decode_head.loss_decode[1].ignore_index = 0
        cfg.model.auxiliary_head.ignore_index = 0
        cfg.model.auxiliary_head.loss_decode[0].avg_non_ignore = True
        cfg.model.auxiliary_head.loss_decode[1].ignore_index = 0
        cfg.data.train.pipeline[3].ignore_index = 0
        cfg.data.train.pipeline[5].seg_pad_val = 0
        cfg.data.train.pipeline[8].seg_pad_val = 0
        cfg.val_pipeline[3].ignore_index = 0
        cfg.val_pipeline[6].seg_pad_val = 0
        cfg.data.val.type='BuildingFacadeBGDataset'
        cfg.data.val.pipeline[2].transforms[0].ignore_index = 0
        cfg.data.val.pipeline[2].transforms[2].seg_pad_val = 0
        cfg.data.test.type='BuildingFacadeBGDataset'
        cfg.data.test.pipeline[2].transforms[0].ignore_index = 0
        cfg.data.test.pipeline[2].transforms[2].seg_pad_val = 0
    # if args.crop_size:
    #     cfg.crop_size = (args.crop_size, args.crop_size)
    if args.keep_ratio:
        cfg.data.train.pipeline[2].keep_ratio = True
        # cfg.val_pipeline[2].keep_ratio = True
    # if args.lr_scheduler:
    if args.momentum:
        cfg.optimizer.momentum = args.momentum
    if args.weight_decay:
        cfg.optimizer.weight_decay = args.weight_decay

    print(cfg.pretty_text)

    cfg.dump('/root/mmsegmentation/configs/deeplabv3plus/mydeeplabv3plus_test.py')

if __name__ == '__main__':
    main()