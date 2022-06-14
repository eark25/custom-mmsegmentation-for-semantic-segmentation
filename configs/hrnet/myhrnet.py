from mmseg.apis import set_random_seed

# # hr18
# _base_ = [
#     '../_base_/models/fcn_hr18.py', '../_base_/datasets/loveda.py',
#     '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
# ]

# # Since we use only one GPU, BN is used instead of SyncBN
# norm_cfg = dict(type='BN', requires_grad=True)

# model = dict(decode_head=dict(num_classes=7))

_base_ = './fcn_hr18_512x512_80k_ade20k.py'

# Since we use only one GPU, BN is used instead of SyncBN
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        num_classes=6,
        in_channels=[48, 96, 192, 384],
        channels=sum([48, 96, 192, 384]),
        norm_cfg=norm_cfg,
        ignore_index=255,
        loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, avg_non_ignore=True),
                    dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]
        )
)

# Modify dataset type and path
dataset_type = 'BuildingFacadeDataset'
data_root = '/root/mmsegmentation/data/buildingfacade'
img_dir = 'imgs'
ann_dir = 'all_new_masks'

img_norm_cfg = dict(
    mean=[255*0.4780, 255*0.4511, 255*0.4137],
    std=[255*0.2429, 255*0.2352, 255*0.2338],
    to_rgb=True
)

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
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
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=train_pipeline,
        split='splits/train.txt'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=try_pipeline,
        split='splits/val.txt'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=try_pipeline,
        split='splits/val.txt')
    )

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
# load_from = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# Set up working dir to save files and logs.
work_dir = '/root/mmsegmentation/tutorial'

runner = dict(type='EpochBasedRunner', max_epochs=1000)
log_config = dict(interval = 1)
evaluation = dict(interval = 1, save_best='mIoU', max_keep_ckpts=1, pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval = -1)

# optimizer
optimizer = dict(type='SGD', lr=1, momentum=0.9, weight_decay=1e-8)
optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=8)

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1, by_epoch=True)

# Set seed to facitate reproducing the result
seed = 0
set_random_seed(0, deterministic=False)
gpu_ids = range(3, 4)

workflow = [('train', 1), ('val', 1)]