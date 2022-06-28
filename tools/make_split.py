import mmcv
import os.path as osp

data_root = '/root/mmsegmentation/data/buildingfacade'
img_dir = 'imgs'
ann_dir = 'all_new_masks'

# split train/val set randomly
split_dir = 'new_splits'
mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(data_root, ann_dir), suffix='.png')]
with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
  # select first 4/5 as train set
  train_length = int(len(filename_list) * 4/5)
  val_length = (len(filename_list) - train_length) // 2
  val_range = train_length + val_length
  f.writelines(line + '\n' for line in filename_list[:train_length])
with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
  # select half of 1/5 as val set
  f.writelines(line + '\n' for line in filename_list[train_length:val_range])
with open(osp.join(data_root, split_dir, 'test.txt'), 'w') as f:
  # select another half of 1/5 as test set
  f.writelines(line + '\n' for line in filename_list[val_range:])