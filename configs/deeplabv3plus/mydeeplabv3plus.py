from mmseg.apis import set_random_seed

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# Since we use only one GPU, BN is used instead of SyncBN
norm_cfg = dict(type='BN', requires_grad=True)

# modify num classes of the model in decode/auxiliary head
model = dict(
    # pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(norm_cfg=norm_cfg), # depth=101),
    decode_head=dict(num_classes=6,
                    norm_cfg=norm_cfg,
                    ignore_index = 255,
                    loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, avg_non_ignore=True),
                                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]
    ),
    auxiliary_head=dict(num_classes=6,
                    norm_cfg=norm_cfg,
                    ignore_index = 255,
                    loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=0.4, avg_non_ignore=True),
                                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.2)]
    )
)
# model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

# Modify dataset type and path
dataset_type = 'BuildingFacadeDataset'
data_root = '/root/mmsegmentation/data/buildingfacade'
img_dir = 'imgs'
ann_dir = 'all_new_masks'

# img_norm_cfg = dict(
#     mean=[255*0.4780, 255*0.4511, 255*0.4137],
#     std=[255*0.2429, 255*0.2352, 255*0.2338],
#     to_rgb=True
# )

img_norm_cfg = dict(
    mean=[255*0.485, 255*0.456, 255*0.406],
    std=[255*0.229, 255*0.224, 255*0.225],
    to_rgb=True
)

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='CLAHE', clip_limit=3.0, tile_grid_size=(8, 8)),
    dict(type='Resize', img_scale=None, multiscale_mode='range', ratio_range=(0.5, 2.0), keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomRotate', prob=0.5, degree=40),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='CLAHE', clip_limit=3.0, tile_grid_size=(8, 8)),
    dict(type='Resize', img_scale=None, ratio_range=(1.0, 1.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(512, 512),
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
try_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='CLAHE', clip_limit=3.0, tile_grid_size=(8, 8)),
            # dict(type='Resize', ratio_range=(1.0, 1.0), keep_ratio=False),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            # dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img', 'gt_semantic_seg']),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=train_pipeline,
        split='new_splits/train.txt'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=try_pipeline,
        split='new_splits/val.txt'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=try_pipeline,
        split='new_splits/test.txt')
    )

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
# load_from = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# Set up working dir to save files and logs.
work_dir = '/root/mmsegmentation/deeplabv3plus_imgnet_CLAHE_run'

runner = dict(type='EpochBasedRunner', max_epochs=1000)
log_config = dict(interval = 1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',  by_epoch=True, init_kwargs=dict(project='deeplab_last_run_firm', resume='allow', anonymous='must'))
    ])
evaluation = dict(interval = 1, pre_eval=True, save_best='mIoU', max_keep_ckpts=1)
checkpoint_config = dict(by_epoch=True, interval = -1, save_last = False)

# optimizer
optimizer = dict(type='SGD', lr=1e-1, momentum=0.9, weight_decay=1e-8)
optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=8)

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-1, by_epoch=True)

# Set seed to facitate reproducing the result
seed = 0
set_random_seed(0, deterministic=False)
gpu_ids = range(1, 2)

workflow = [('train', 1), ('val', 1)]