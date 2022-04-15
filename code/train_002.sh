CUDA_VISIBLE_DEVICES=6,7 python tools/train_val.py --config experiments/config_003.yaml
CUDA_VISIBLE_DEVICES=6 python tools/eval.py --config experiments/config_003.yaml