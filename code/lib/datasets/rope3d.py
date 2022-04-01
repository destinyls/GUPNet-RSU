import os

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.utils import get_angle_from_box3d,check_range
from lib.datasets.rope3d_utils import get_objects_from_label
from lib.datasets.rope3d_utils import Calibration
from lib.datasets.rope3d_utils import get_affine_transform
from lib.datasets.rope3d_utils import affine_transform
from lib.datasets.rope3d_utils import *

class Rope3D(data.Dataset):
    def __init__(self, root_dir, split, cfg):
        self.num_classes = 8
        self.max_objs = 250
        self.class_name = ['car', 'van', 'truck', 'bus', 'pedestrian', 'cyclist', 'motorcyclist', 'tricyclist']
        self.cls2id = {'car': 0, 'van': 1 , 'truck': 2, 'bus': 3, 'pedestrian': 4, 'cyclist': 5, 'motorcyclist': 6, 'tricyclist': 7}

        self.resolution = np.array([1920, 1056])  # W * H
        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['Van', 'Truck'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare'])

        # h w l
        self.cls_mean_size = np.array([[1.28877281, 1.69392866, 4.25668836],
                                       [1.7199305, 1.7356766, 4.6411424],
                                       [2.6822665, 2.3482678, 6.940249 ],
                                       [2.958849,  2.519919, 10.542151],
                                       [1.59624921, 0.47972058, 0.46641969],
                                       [1.4376054,  0.48926293, 1.5239355 ],
                                       [1.41811709, 0.58211171, 1.67083623],
                                       [1.5357665, 1.0777776, 2.5793583],])       
        # data split loading
        assert split in ['train', 'val']
        self.split = split
        self.root_dir = root_dir

        # path configuration
        self.data_dir = os.path.join(self.root_dir, 'training' if split == 'train' else 'validation')
        self.split_txt = os.path.join(self.root_dir, 'training/train.txt' if split == 'train' else 'validation/val.txt')
        self.calib_dir = os.path.join(self.data_dir, "calib")
        self.label_dir = os.path.join(self.data_dir, "label_2")
        self.extrinsics_dir = os.path.join(self.data_dir, "extrinsics")
        self.image_dir = os.path.join(self.root_dir, 
            "training-image_2a_6978886144233472/training-image_2a" if split == 'train' else 
            'validation-image_2_6978886144233472/validation-image_2')   

        idx_list = [x.strip() for x in open(self.split_txt).readlines()]
        self.idx_list = []
        for index in idx_list:
            img_file = os.path.join(self.image_dir, index + ".jpg")
            if os.path.exists(img_file):
                self.idx_list.append(index)
        self.idx_list = self.idx_list

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # others
        self.downsample = 4

    def __len__(self):
        return len(self.idx_list[:30])

    def get_image(self, index):
        img_file = os.path.join(self.image_dir, index + ".jpg")
        assert os.path.exists(img_file)
        return Image.open(img_file)           # (H, W, 3) RGB mode

    def get_calib(self, index):
        calib_file = os.path.join(self.calib_dir, index + ".txt")
        extrinsic_file = os.path.join(self.extrinsics_dir, index + ".yaml")
        assert os.path.exists(calib_file)
        assert os.path.exists(extrinsic_file)
        return Calibration(calib_file, extrinsic_file)

    def get_label(self, index):
        label_file = os.path.join(self.label_dir, index + ".txt")
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def __getitem__(self, idx):
        #  ============================   get inputs   ===========================
        index = self.idx_list[idx]
        # img_proto = self.get_image(index)
        # img = img_proto.copy()
        img = self.get_image(index)
        img_size = np.array(img.size)
        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False
        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() < self.random_crop:
                random_crop_flag = True
                crop_size = img_size * np.clip(np.random.randn()*self.scale + 1, 1 - self.scale, 1 + self.scale)
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        coord_range = np.array([center-crop_size/2, center+crop_size/2]).astype(np.float32)                   
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        calib = self.get_calib(index)
        features_size = self.resolution // self.downsample  # W * H

        #  ============================   get labels   ==============================
        if self.split == 'train':
            objects = self.get_label(index)
            # data augmentation for labels
            if random_flip_flag:
                calib.flip(img_size)
                for object in objects:
                    [x1, _, x2, _] = object.box2d
                    object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    object.ry = np.pi - object.ry
                    object.pos[0] *= -1
                    if object.ry > np.pi:  object.ry -= 2 * np.pi
                    if object.ry < -np.pi: object.ry += 2 * np.pi

            # labels encoding
            heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32) # C * H * W
            size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            depth = np.zeros((self.max_objs, 1), dtype=np.float32)
            heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
            height2d = np.zeros((self.max_objs, 1), dtype=np.float32)
            cls_ids = np.zeros((self.max_objs), dtype=np.int64)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
            mask_3d = np.zeros((self.max_objs), dtype=np.uint8)
            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
            for i in range(object_num):
                # filter objects by writelist
                if objects[i].cls_type not in self.writelist:
                    continue
                # filter inappropriate samples by difficulty
                if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                    continue
                if np.sum(objects[i].pos) == 0:
                    continue

                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()
                # add affine transformation for 2d boxes.
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                # modify the 2d bbox according to pre-compute downsample ratio
                bbox_2d[:] /= self.downsample
    
                # process 3d bbox & get 3d center
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
                center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                center_3d = center_3d[0]  # shape adjustment
                center_3d = affine_transform(center_3d.reshape(-1), trans)
                center_3d /= self.downsample      
            
                # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
                if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
                if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue
    
                # generate the radius of gaussian heatmap
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))
    
                if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                    draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                    continue
    
                cls_id = self.cls2id[objects[i].cls_type]
                cls_ids[i] = cls_id
                draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)
    
                # encoding 2d/3d offset & 2d size
                indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                offset_2d[i] = center_2d - center_heatmap
                size_2d[i] = 1. * w, 1. * h
    
                # encoding depth
                depth[i] = objects[i].pos[-1]
                # encoding heading angle
                #heading_angle = objects[i].alpha
                heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0]+objects[i].box2d[2])/2)
                if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi: heading_angle += 2 * np.pi
                heading_bin[i], heading_res[i] = angle2class(heading_angle)
    
                # encoding 3d offset & size_3d
                offset_3d[i] = center_3d - center_heatmap
                src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                size_3d[i] = src_size_3d[i] - mean_size
                if objects[i].trucation <=0.5 and objects[i].occlusion<=2:    
                    mask_2d[i] = 1
           
            targets = {'depth': depth,
                       'size_2d': size_2d,
                       'heatmap': heatmap,
                       'offset_2d': offset_2d,
                       'indices': indices,
                       'size_3d': size_3d,
                       'offset_3d': offset_3d,
                       'heading_bin': heading_bin,
                       'heading_res': heading_res,
                       'cls_ids': cls_ids,
                       'mask_2d': mask_2d} 
        else:
            targets = {}
        inputs = img
        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}   
        return inputs, calib.P2, coord_range, targets, info   # for training
        # return img_proto, calib, coord_range, objects, info     # for debug

if __name__ == '__main__':
    cfg = {'random_flip':0.0, 'random_crop':0.0, 'scale':0.0, 'shift':0.0, 'use_dontcare': False,
           'class_merging': False, 'writelist':['car','van' ,'truck','bus','pedestrian','cyclist','motorcyclist', 'Tricyclist'], 'use_3d_center':True}
    root_dir = "/home/yanglei/DataSets/DAIR-V2X/Rope3D"
    dataset = Rope3D(root_dir, 'train', cfg)
    for i in range(20):
        break
        img_proto, calib, coord_range, objs, info = dataset[i]
        demo_img = cv2.cvtColor(np.asarray(img_proto), cv2.COLOR_RGB2BGR)
        for obj in objs:
            if np.sum(obj.pos) == 0:
                continue
            corners3d = obj.generate_corners3d()
            pts_img, pts_depth = calib.rect_to_img(corners3d)
            obj_c = np.mean(pts_img, axis=0)
            box2d = obj.box2d
            demo_img = draw_box_3d(demo_img, pts_img)
            cv2.circle(demo_img, (int(obj_c[0]), int(obj_c[1])), 6, (0, 255, 0), thickness=-1)
            cv2.rectangle(demo_img, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), (0, 255, 0))
        cv2.imwrite(os.path.join("debug", str(i) + ".jpg"), demo_img)


'''
import numpy as np
import re
import os
import sys
import getopt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.datasets.rope3d_utils import *

class Rope3DDataset:
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.data_dir = os.path.join(self.root_dir, 'training' if split == 'train' else 'validation')
        self.split_txt = os.path.join(self.root_dir, 'training/train.txt' if split == 'train' else 'validation/val.txt')
        self.calib_path = os.path.join(self.data_dir, "calib")
        self.label_path = os.path.join(self.data_dir, "label_2")
        self.extrinsics_path = os.path.join(self.data_dir, "extrinsics")
        self.image_path = os.path.join(self.root_dir, 
            "training-image_2a_6978886144233472/training-image_2a" if split == 'train' else 
            'validation-image_2_6978886144233472/validation-image_2')   
        
        idx_list = [x.strip() for x in open(self.split_txt).readlines()]
        self.idx_list = []
        for index in idx_list:
            img_file = os.path.join(self.image_path, index + ".jpg")
            if os.path.exists(img_file):
                self.idx_list.append(index)

    def __len__(self):
        return len(self.idx_list)

    def get_image(self, idx):
        img_file = os.path.join(self.image_path, self.idx_list[idx] + ".jpg")
        if not os.path.exists(img_file):
            print(img_file)

        return Image.open(img_file)    # (H, W, 3) RGB mode

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_path, self.idx_list[idx] + ".txt")
        extrinsic_file = os.path.join(self.extrinsics_path, self.idx_list[idx] + ".yaml")
        assert os.path.exists(calib_file)
        assert os.path.exists(extrinsic_file)
        return Calibration(calib_file, extrinsic_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_path, self.idx_list[idx] + ".txt")
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def __getitem__(self, idx):
        image = self.get_image(idx)
        calib = self.get_calib(idx)
        objs = self.get_label(idx)
        demo_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        
        for obj in objs:
            if np.sum(obj.pos) == 0:
                continue
        
            corners3d = obj.generate_corners3d()
            pts_img, pts_depth = calib.rect_to_img(corners3d)
            obj_c = np.mean(pts_img, axis=0)
            demo_img = draw_box_3d(demo_img, pts_img)
            cv2.circle(demo_img, (int(obj_c[0]), int(obj_c[1])), 6, (0, 255, 0), thickness=-1)
            box2d = obj.box2d
            cv2.rectangle(demo_img, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), (0, 255, 0))
        
        return objs, demo_img

if __name__ == "__main__":
    root_dir = "/home/yanglei/DataSets/DAIR-V2X/Rope3D"
    dataset = Rope3DDataset(root_dir=root_dir, split="train")
    dim_cls = {'car': 0,'van': None ,'truck': None,'bus': None,'pedestrian': None,'cyclist': None,'motorcyclist': None,'tricyclist': None}
    dim_num = {'car': 0,'van': 0 ,'truck': 0,'bus': 0,'pedestrian': 0,'cyclist': 0,'motorcyclist': 0,'tricyclist': 0}

    for i in range(len(dataset)):
        objs, demo_img = dataset[i]
        cv2.imwrite(os.path.join("debug", str(i) + ".jpg"), demo_img)

        for obj in objs:
            if np.sum(obj.pos) == 0 or obj.cls_type not in dim_cls.keys():
                continue
            if dim_cls[obj.cls_type] is None:
                dim_cls[obj.cls_type] = obj.dim
                dim_num[obj.cls_type] = 1
            else:
                dim_cls[obj.cls_type] += obj.dim
                dim_num[obj.cls_type] += 1

    for cat, dim_sum in dim_cls.items():
        print(cat, dim_sum, dim_num[cat])
        print(cat, dim_sum / dim_num[cat])
'''