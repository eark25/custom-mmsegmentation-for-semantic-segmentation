import mmcv
import matplotlib.pyplot as plt

# img = mmcv.imread('/home/eark/thesis/Pytorch-UNet/data/base/imgs/cmp_b0378.jpg')
# plt.figure(figsize=(8, 6))
# plt.imshow(mmcv.bgr2rgb(img))
# plt.show()

# img = mmcv.imread('/home/eark/thesis/Pytorch-UNet/data/base/all_new_masks/cmp_b0378.png')
# plt.figure(figsize=(8, 6))
# plt.imshow(mmcv.bgr2rgb(img))
# plt.show()

import os.path as path
import numpy as np
from PIL import Image

# convert dataset annotation to semantic segmentation map
data_root = '/home/eark/thesis/Pytorch-UNet/data/base'
img_dir = 'imgs'
ann_dir = 'all_new_masks'

# define class and palette for better visualization
classes = ('background', 'wall', 'floor', 'column', 'opening', 'facade/deco')
for_palette = Image.open('/home/eark/thesis/Pytorch-UNet/data/base/all_new_masks/cmp_b0378.png')
# palette = for_palette.getpalette()
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
           [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]
palette = np.array(palette, dtype=np.uint8)#.reshape((256,3))[0:len(classes)]
print('===== Palette data =====\n', palette)
print(np.unique(for_palette))

for_palette.putpalette(palette)

import matplotlib.patches as mpatches

plt.figure()
# plt.imshow(np.array(for_palette.convert('RGB')))
plt.imshow(for_palette)

# create the legend
patches = [mpatches.Patch(color=(np.array(palette[i])/255.), label=classes[i]) for i in range(len(classes))]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')

plt.show()