import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class BuildingFacadeBGDataset(CustomDataset):
    CLASSES = ('background', 'wall', 'floor', 'column', 'opening', 'facade/deco')
    PALETTE = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], [0, 11, 123], [118, 20, 12]]#, [122, 81, 25], [241, 134, 51]]
    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', split=split, ignore_index=0, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None