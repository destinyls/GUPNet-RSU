from ast import Break
import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.utils import get_angle_from_box3d, check_range
from lib.datasets.kitti_utils import get_objects_from_label
from lib.datasets.kitti_utils import Calibration
from lib.datasets.kitti_utils import get_affine_transform
from lib.datasets.kitti_utils import affine_transform
from lib.datasets.kitti_utils import compute_box_3d, draw_box_3d
import pdb

class KITTI(data.Dataset):
    def __init__(self, root_dir, split, cfg):
        # basic configuration
        self.num_classes = 3
        self.max_objs = 250
        self.class_name = ['pedestrian', 'car', 'cyclist']
        self.cls2id = {'pedestrian': 0, 'car': 1, 'cyclist': 2}
        
        self.resolution = np.array([1920, 1056])  # W * H
        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['van', 'truck'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare'])

        ##l,w,h
        self.cls_mean_size = np.array([[1.59624921, 0.47972058, 0.46641969],
                                       [1.28877281, 1.69392866, 4.25668836],
                                       [1.4376054,  0.48926293, 1.5239355 ]])                                            
        # data split loading
        assert split in ['train', 'val']
        self.split = split
        self.root_dir = root_dir
        self.data_dir = os.path.join(self.root_dir, 'training' if split == 'train' else 'validation')
        self.split_txt = os.path.join(self.root_dir, 'training/train.txt' if split == 'train' else 'validation/val.txt')
        self.calib_dir = os.path.join(self.data_dir, "calib")
        self.label_dir = os.path.join(self.data_dir, "label_2")
        self.depth_dir = os.path.join(self.data_dir, 'depth')
        self.denorm_dir = os.path.join(self.data_dir, "denorm")
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

        # data augmentation configuration
        self.data_augmentation = True if split in ['train'] else False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4
        
    def get_image(self, index):
        img_file = os.path.join(self.image_dir, index + ".jpg")
        assert os.path.exists(img_file)
        return Image.open(img_file)           # (H, W, 3) RGB mode

    def get_calib(self, index):
        calib_file = os.path.join(self.calib_dir, index + ".txt")
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def get_label(self, index):
        label_file = os.path.join(self.label_dir, index + ".txt")
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_denorm(self, index):
        denorm_file = os.path.join(self.denorm_dir, index + ".txt")
        assert os.path.exists(denorm_file)
        with open(denorm_file, 'r') as f:
            lines = f.readlines()
        denorm = np.array([float(item) for item in lines[0].split(' ')])
        return denorm

    def __len__(self):
        return len(self.idx_list[:1000])

    def __getitem__(self, item):
        index = self.idx_list[item]
        denorm = self.get_denorm(index)
        # img_proto = self.get_image(index)
        # img = img_proto.copy()
        img = self.get_image(index)
        img_size = np.array(img.size)
        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
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
        coord_range = np.array([center-crop_size/2,center+crop_size/2]).astype(np.float32)                   
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        calib = self.get_calib(index)
        features_size = self.resolution // self.downsample# W * H
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
    
                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()
                # add affine transformation for 2d boxes.
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                # modify the 2d bbox according to pre-compute downsample ratio
                bbox_2d[:] /= self.downsample
    
                # process 3d bbox & get 3d center
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
                center_3d, corners3d = objects[i].generate_corners3d_denorm(denorm)     # real 3D center in 3D space
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
    
                if objects[i].cls_type in ['van', 'truck']:
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

                #objects[i].trucation <=0.5 and objects[i].occlusion<=2 and (objects[i].box2d[3]-objects[i].box2d[1])>=25:
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
        # collect return data
        inputs = img
        info = {'img_id': index,
                'img_size': img_size,
                'denorm': denorm,
                'bbox_downsample_ratio': img_size/features_size}   
        return inputs, calib.P2, coord_range, targets, info   #calib.P2
        # return img_proto, calib, coord_range, objects, info     # for debug

if __name__ == '__main__':
    cfg = {'random_flip':0.0, 'random_crop':0.0, 'scale':0.0, 'shift':0.0, 'use_dontcare': False,
           'class_merging': False, 'writelist':['car', 'pedestrian','cyclist'], 'use_3d_center':True}
    root_dir = "/home/yanglei/DataSets/DAIR-V2X/Rope3D"
    dataset = KITTI(root_dir, 'train', cfg)
    for i in range(1):
        break
        img_proto, calib, coord_range, objs, info = dataset[i]
        demo_img = cv2.cvtColor(np.asarray(img_proto), cv2.COLOR_RGB2BGR)
        for obj in objs:
            if np.sum(obj.pos) == 0:
                continue
            center3d, corners3d = obj.generate_corners3d_denorm(info['denorm'])
            pts_img, pts_depth = calib.rect_to_img(corners3d)
            obj_c = np.mean(pts_img, axis=0)
            box2d = obj.box2d
            demo_img = draw_box_3d(demo_img, pts_img)
            cv2.circle(demo_img, (int(obj_c[0]), int(obj_c[1])), 6, (0, 255, 0), thickness=-1)
            cv2.rectangle(demo_img, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), (0, 255, 0))
        cv2.imwrite(os.path.join("debug", str(i) + ".jpg"), demo_img)

