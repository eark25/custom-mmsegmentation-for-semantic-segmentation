from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmseg.models import build_segmentor
import mmcv
from mmcv.runner import load_checkpoint

config_file = '/root/mmsegmentation/configs/hrnet/myhrnet_final_test.py'
checkpoint_file = '/root/mmsegmentation/hrnet_final_run/best_mIoU_epoch_635.pth'
classes = ('background', 'wall', 'floor', 'column', 'opening', 'facade/deco')
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], [0, 11, 123], [118, 20, 12]]
device = 'cuda:0'

img_norm_cfg = dict(
    mean=[255*0.4780, 255*0.4511, 255*0.4137],
    std=[255*0.2429, 255*0.2352, 255*0.2338],
    to_rgb=True
)

crop_size = (512, 512)
inference_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# init segmentor
config = mmcv.Config.fromfile(config_file)
config.model.pretrained = None
config.model.train_cfg = None
config.data.test.pipeline = inference_pipeline
# del config.data.test.pipeline[1]
model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
model.CLASSES = checkpoint['meta']['CLASSES']
model.PALETTE = palette
model.cfg = config  # save the config in the model for convenience
model.to(device)
model.eval()

input = '/root/mmsegmentation/data/buildingfacade/imgs/cmp_b0022.jpg'
result = inference_segmentor(model, input)

output = show_result_pyplot(model, input, result, model.PALETTE)
print(output)
print(output.shape)
import cv2
cv2.imwrite('test_script_output.jpg', output)