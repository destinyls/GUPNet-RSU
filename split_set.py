import os
import random

root = "/home/yanglei/DataSets/DAIR-V2X/Rope3D-KITTI-v2/ImageSets"
train_txt = os.path.join(root, "train.txt")
val_txt =  os.path.join(root, "val.txt")
trainval_txt =  os.path.join(root, "trainval.txt")

trainval_idx_list = [x.strip() for x in open(trainval_txt).readlines()]
random.shuffle(trainval_idx_list)

train_idx_list = trainval_idx_list[:int(0.7 * len(trainval_idx_list))]
val_idx_list = trainval_idx_list[int(0.7 * len(trainval_idx_list)):]

with open(train_txt,'w') as f:
    for idx in train_idx_list:
        f.write(idx)
        f.write("\n")

with open(val_txt,'w') as f:
    for idx in val_idx_list:
        f.write(idx)
        f.write("\n")