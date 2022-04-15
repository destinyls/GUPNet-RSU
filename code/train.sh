CUDA_VISIBLE_DEVICES=4,5 python tools/train_val.py --config experiments/config.yaml
CUDA_VISIBLE_DEVICES=4 python tools/eval.py --config experiments/config.yaml

CUDA_VISIBLE_DEVICES=4,5 python tools/train_val.py --config experiments/config_002.yaml
CUDA_VISIBLE_DEVICES=4 python tools/eval.py --config experiments/config_002.yaml
