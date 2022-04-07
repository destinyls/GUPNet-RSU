import pdb

from .kitti_object_eval_python.evaluate import evaluate as _evaluate_python
from .kitti_object_eval_python.evaluate import generate_kitti_3d_detection, check_last_line_break

def evaluate_python(label_path, result_path, label_split_file, current_class, metric):
    
    print(label_path, result_path)
    result, ret_dict = _evaluate_python(label_path, result_path, label_split_file, current_class, metric=metric)

    return result, ret_dict

if __name__ == '__main__':
    pass