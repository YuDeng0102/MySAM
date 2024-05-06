import os
import torch
import lightning as L
from box import Box
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader

from configs.config import cfg
from datasets.COCO import COCODataset
from datasets.tools import ResizeAndPad, collate_fn
from model import Model
from utils.eval_coco_mAp import evaluate_coco_map
from utils.tools import copy_model, create_csv, check_grad, momentum_update, reduce_instances
import argparse



def configure_opt(cfg: Box, model: Model):
    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor ** 2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def corrupt_main(cfg):
    for corrupt in cfg.corruptions:
        cfg.corrupt = corrupt
        cfg.name = corrupt
        torch.cuda.empty_cache()
        main(cfg)


def main(cfg: Box) -> None:
    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)

    fabric = L.Fabric(accelerator="cuda",
                      devices=1,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir)])
    fabric.launch()
    fabric.seed_everything(1737 + fabric.global_rank)

    with fabric.device:
        model = Model(cfg)
        model.setup()
    transform = ResizeAndPad(1024)
    test_dataset = COCODataset(
        cfg,
        root_dir=cfg.datasets.coco.root_dir,
        transform=transform,
        dataset_type='test',
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn
    )
    test_loader = fabric._setup_dataloader(test_loader)

    full_checkpoint = fabric.load(cfg.resume_dir)
    model.load_state_dict(full_checkpoint["model"])

    evaluate_coco_map(fabric, cfg, model, test_loader, name=cfg.name, epoch=cfg.num_epochs)




if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

    print("avaiable:", torch.cuda.device_count())

    import torch
    parser = argparse.ArgumentParser(
    description=__doc__)

    parser.add_argument('--resume_dir', default='checkpoints/last-ckpt.pth', type=str, help='resume from checkpoint')
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size')

    args = parser.parse_args()
    if args.batch_size!=0:
        cfg.batch_size = args.batch_size
    if args.resume_dir!='':
        cfg.resume_dir =args.resume_dir


    main(cfg)
    torch.cuda.empty_cache()