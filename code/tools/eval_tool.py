import os
import argparse
from lib.evaluation import evaluate_python
from lib.evaluation.kitti_object_eval_python import kitti_common as kitti
# from mmdet3d.core.evaluation.kitti_utils.eval import kitti_eval

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def parse_option():
    parser = argparse.ArgumentParser('Evaluate tools', add_help=False)
    parser.add_argument('--dt_root', type=str, required=False, metavar="", help='root path to rope3d dataset')
    parser.add_argument('--pred_path', type=str, required=False, metavar="", help='root path to preds')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_option()
    dt_root, pred_path = args.dt_root, args.pred_path
    gt_label_path = os.path.join(dt_root, "training", "label_2")
    imageset_txt = os.path.join(dt_root, "ImageSets", "val.txt")
    '''
    dt_annos = kitti.get_label_annos(pred_path)
    score_thresh = 0.15
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(imageset_txt)
    gt_annos = kitti.get_label_annos(gt_label_path, val_image_ids)
    print(len(dt_annos), len(gt_annos))
    
    ret = kitti_eval(gt_annos, dt_annos, ["Car"], ['3d'])
    print(ret)
    '''
    result, ret_dict = evaluate_python(label_path=gt_label_path, 
                                        result_path=pred_path,
                                        label_split_file=imageset_txt,
                                        current_class=["Car"],
                                        metric='R40')         
    print(result)
    
    