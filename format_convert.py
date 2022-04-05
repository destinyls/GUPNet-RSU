import os
import sys
import argparse
import cv2

import numpy as np
from shutil import copyfile
from tqdm import tqdm

category_map = {'car': 'Car', 'van': 'Van', 'truck': 'Truck', 'bus': 'Bus', 'pedestrian': 'Pedestrian', 'cyclist': 'Cyclist', 'motorcyclist': 'Motorcyclist', 'tricyclist': 'Tricyclist', 'trafficcone': 'Trafficcone', 'unknown_unmovable': 'Unknown_unmovable', 'barrow': 'Barrow', 'unknowns_movable': 'Unknowns_movable', 'triangle plate': 'Triangle plate'}

def parse_option():
    parser = argparse.ArgumentParser('Convert rope3D dataset to standard kitti format', add_help=False)
    parser.add_argument('--src_root', type=str, required=False, metavar="", help='root path to rope3d dataset')
    parser.add_argument('--dest_root', type=str, required=False, metavar="", help='root path to rope3d dataset in kitti format')
    parser.add_argument('--split', type=str, default='train', help='split: train or val',)
    args = parser.parse_args()
    return args
    

def convert_calib(src_calib_file, dest_calib_file):
    with open(src_calib_file) as f:
        lines = f.readlines()
    obj = lines[0].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    kitti_calib = dict()
    kitti_calib["P0"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["P1"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["P2"] = P2  # Left camera transform.
    kitti_calib["P3"] = np.zeros((3, 4))  # Dummy values.
    # Cameras are already rectified.
    kitti_calib["R0_rect"] = np.identity(3)
    kitti_calib["Tr_velo_to_cam"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["Tr_imu_to_velo"] = np.zeros((3, 4))  # Dummy values.
    
    with open(dest_calib_file, "w") as calib_file:
        for (key, val) in kitti_calib.items():
            val = val.flatten()
            val_str = "%.12e" % val[0]
            for v in val[1:]:
                val_str += " %.12e" % v
            calib_file.write("%s: %s\n" % (key, val_str))

def ry2alpha(ry, pos):
    alpha = ry - np.arctan2(pos[0], pos[2])
    if alpha > np.pi:
        alpha -= 2 * np.pi
    if alpha < -np.pi:
        alpha += 2 * np.pi
    return alpha

def alpha2roty(alpha, pos):
    ry = alpha + np.arctan2(pos[0], pos[2])
    if ry > np.pi:
        ry -= 2 * np.pi
    if ry < -np.pi:
        ry += 2 * np.pi
    return ry
 
def convert_label(src_label_file, dest_label_file):
    with open(src_label_file, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        label = line.strip().split(' ')
        cls_type = label[0]
        label[0] = category_map[cls_type]
        if cls_type not in ['car', 'pedestrian', 'cyclist']:
            continue
        
        truncated = int(label[1])
        if truncated > 0:
            truncated = 0.3
        label[1] = str(truncated)
        
        alpha = float(label[3])
        pos = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        ry = float(label[14])
        if alpha > np.pi:
            alpha -= 2 * np.pi
            ry = alpha2roty(alpha, pos)
        label[3] = str(alpha) 
        label[14] = str(ry)
        new_lines.append(' '.join(label))
        
    with open(dest_label_file,'w') as f:
        for line in new_lines:
            f.write(line)
            f.write("\n")
            
def main(src_root, dest_root, split='train'):
    if split == 'train':
        src_dir = os.path.join(src_root, "training")
        dest_dir = os.path.join(dest_root, "training")
        img_path = "training-image_2a_6978886144233472/training-image_2a"
    else:
        src_dir = os.path.join(src_root, "validation")
        dest_dir = os.path.join(dest_root, "testing")
        img_path = "validation-image_2_6978886144233472/validation-image_2"
        
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "image_2"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "label_2"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "calib"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "../", "ImageSets"), exist_ok=True)
    imageset_txt = os.path.join(dest_dir, "../", "ImageSets", "train.txt" if split=='train' else 'test.txt')
    
    src_img_path = os.path.join(src_dir, "../", img_path)
    src_label_path = os.path.join(src_dir, "label_2")
    src_calib_path = os.path.join(src_dir, "calib")
    
    split_txt = os.path.join(src_dir, "train.txt" if split=='train' else 'val.txt')
    idx_list = [x.strip() for x in open(split_txt).readlines()]
    idx_list_valid = []
    for index in idx_list:
        img_file = os.path.join(src_img_path, index + ".jpg")
        if os.path.exists(img_file):
            idx_list_valid.append(index)
            
    img_id = 0
    img_id_list = []
    for index in tqdm(idx_list_valid):
        src_img_file = os.path.join(src_img_path, index + ".jpg")
        src_label_file = os.path.join(src_label_path, index + ".txt")
        src_calib_file = os.path.join(src_calib_path, index + ".txt")
    
        dest_img_file = os.path.join(dest_dir, "image_2", '{:06d}.png'.format(img_id))
        dest_label_file = os.path.join(dest_dir, "label_2", '{:06d}.txt'.format(img_id))
        dest_calib_file = os.path.join(dest_dir, "calib", '{:06d}.txt'.format(img_id))
        
        img_id_list.append(img_id)
        img_id = img_id + 1
        
        # image_2
        img = cv2.imread(src_img_file)
        cv2.imwrite(dest_img_file, img)
        # calib
        convert_calib(src_calib_file, dest_calib_file)
        # label
        convert_label(src_label_file, dest_label_file)

    with open(imageset_txt,'w') as f:
        for idx in img_id_list:
            frame_name = "{:06d}".format(idx)
            f.write(frame_name)
            f.write("\n")

if __name__ == "__main__":
    args = parse_option()
    src_root, dest_root, split = args.src_root, args.dest_root, args.split
    main(src_root, dest_root, split)
        