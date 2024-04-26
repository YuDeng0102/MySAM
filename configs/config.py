from box import Box
from configs.base_config import base_config


config = {
    "gpu_ids": "0",
    "batch_size": 1,
    "val_batchsize": 1,
    "num_workers": 2,
    "num_epochs": 10,
    "max_nums": 50,
    "num_points": 5,
    "resume": False,
    "start_fold":0,
    "dataset": "COCO",
    "visual": False,
    "load_type": "soft",
    "prompt": "box",
    "out_dir": "output/MyCoco5/v0",
    "name": "base",
    "corrupt": None,
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay":0.0001,
        "momentum":0.9
    }
}

cfg = Box(base_config)
cfg.merge_update(config)