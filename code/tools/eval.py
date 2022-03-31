import os
import sys
import yaml
import logging
import shutil
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import numpy as np

from lib.evaluation import evaluate_python
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.tester_helper import Tester

from train_val import create_logger

parser = argparse.ArgumentParser(description='implementation of GUPNet eval')
parser.add_argument('--config', type=str, default = 'experiments/config.yaml')
args = parser.parse_args()

if __name__ == "__main__":
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg_train = cfg['trainer']
    logger = create_logger(os.path.join(cfg_train['log_dir'],'train.log'))    

    gt_label_path = "../datasets/KITTI/training/label_2/"
    imageset_txt = "../datasets/KITTI/ImageSets/val.txt"
    pred_label_path = os.path.join('./outputs', 'data')
    os.makedirs(pred_label_path, exist_ok=True)

    evaluation_path = os.path.join(cfg_train['output_dir'], 'eval_metric')
    os.makedirs(evaluation_path, exist_ok=True)

    checkpoint_path = os.path.join(cfg_train['output_dir'], 'checkpoints')
    train_loader, val_loader, test_loader = build_dataloader(cfg['dataset'])
    model = build_model(cfg['model'],train_loader.dataset.cls_mean_size)

    best_mAP_3d_moderate = 0
    for pth_name in os.listdir(checkpoint_path):
        if "pth" not in pth_name or "best" in pth_name:
            continue
        epoch = int(pth_name[17:-4])
        model_state = torch.load(os.path.join(checkpoint_path, pth_name))["model_state"]
        model.load_state_dict(model_state)
        tester = Tester(cfg['tester'], model, val_loader, logger)
        tester.test()

        result, ret_dict = evaluate_python(label_path=gt_label_path, 
                                            result_path=pred_label_path,
                                            label_split_file=imageset_txt,
                                            current_class=["Car", "Pedestrian", "Cyclist"],
                                            metric='R40')
        mAP_3d_moderate = ret_dict['Car_3d_0.70/moderate']
        if mAP_3d_moderate > best_mAP_3d_moderate:
            shutil.copyfile(os.path.join(checkpoint_path, pth_name), os.path.join(checkpoint_path, 'checkpoint_best.pth'))
            best_mAP_3d_moderate = mAP_3d_moderate
        os.remove(os.path.join(checkpoint_path, pth_name))

        print(result)
        with open(os.path.join(evaluation_path, 'epoch_result_{:07d}_{}.txt'.format(epoch, round(mAP_3d_moderate, 2))), "w") as f:
            f.write(result)
