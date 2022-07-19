norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                loss_weight=1.0,
                avg_non_ignore=True),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        ],
        ignore_index=255),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                loss_weight=0.4,
                avg_non_ignore=True),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.2)
        ],
        ignore_index=255),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'BuildingFacadeDataset'
data_root = '/root/mmsegmentation/data/buildingfacade'
img_norm_cfg = dict(
    mean=[121.89, 115.0305, 105.4935],
    std=[61.9395, 59.976, 59.619],
    to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize',
        img_scale=None,
        multiscale_mode='range',
        ratio_range=(0.5, 2.0),
        keep_ratio=False),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomRotate', prob=0.5, degree=40),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[121.89, 115.0305, 105.4935],
        std=[61.9395, 59.976, 59.619],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='BuildingFacadeDataset',
        data_root='/root/mmsegmentation/data/buildingfacade',
        img_dir='imgs',
        ann_dir='all_new_masks',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize',
                img_scale=None,
                multiscale_mode='range',
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='RandomRotate', prob=0.5, degree=40),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[121.89, 115.0305, 105.4935],
                std=[61.9395, 59.976, 59.619],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        split='new_splits/train.txt'),
    val=dict(
        type='BuildingFacadeDataset',
        data_root='/root/mmsegmentation/data/buildingfacade',
        img_dir='imgs',
        ann_dir='all_new_masks',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                img_ratios=[1.0],
                flip=False,
                transforms=[
                    dict(
                        type='RandomCrop',
                        crop_size=(512, 512),
                        cat_max_ratio=0.75),
                    dict(
                        type='Normalize',
                        mean=[121.89, 115.0305, 105.4935],
                        std=[61.9395, 59.976, 59.619],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(512, 512),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(
                        type='ImageToTensor', keys=['img', 'gt_semantic_seg']),
                    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
                ])
        ],
        split='new_splits/val.txt'),
    test=dict(
        type='BuildingFacadeDataset',
        data_root='/root/mmsegmentation/data/buildingfacade',
        img_dir='imgs',
        ann_dir='all_new_masks',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                img_ratios=[1.0],
                flip=False,
                transforms=[
                    dict(
                        type='RandomCrop',
                        crop_size=(512, 512),
                        cat_max_ratio=0.75),
                    dict(
                        type='Normalize',
                        mean=[121.89, 115.0305, 105.4935],
                        std=[61.9395, 59.976, 59.619],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(512, 512),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(
                        type='ImageToTensor', keys=['img', 'gt_semantic_seg']),
                    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
                ])
        ],
        split='new_splits/test.txt'))
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(
            type='WandbLoggerHook',
            by_epoch=True,
            init_kwargs=dict(
                project='deeplab_last_run_lyric',
                resume='allow',
                anonymous='must'))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=1e-08)
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook', cumulative_iters=8)
lr_config = dict(policy='poly', power=0.9, min_lr=0.001, by_epoch=True)
checkpoint_config = dict(by_epoch=True, interval=-1, save_last=False)
evaluation = dict(
    interval=1,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU',
    max_keep_ckpts=1)
img_dir = 'imgs'
ann_dir = 'all_new_masks'
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize', img_scale=None, ratio_range=(1.0, 1.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[121.89, 115.0305, 105.4935],
        std=[61.9395, 59.976, 59.619],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
try_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(
                type='Normalize',
                mean=[121.89, 115.0305, 105.4935],
                std=[61.9395, 59.976, 59.619],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img', 'gt_semantic_seg']),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ])
]
work_dir = '/root/mmsegmentation/deeplab_last_run_bg'
runner = dict(type='EpochBasedRunner', max_epochs=1000)
seed = 0
gpu_ids = range(2, 3)
