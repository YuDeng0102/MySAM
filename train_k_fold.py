import os
import time
import torch
import lightning as L
import torch.nn.functional as F
# import segmentation_models_pytorch as smp
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader

from configs.config import cfg
from datasets.COCO import COCODataset
from datasets.tools import ResizeAndPad, collate_fn, collate_fn_soft
from losses import DiceLoss, FocalLoss, ContraLoss
from datasets import call_load_dataset

from model import Model
from utils.eval_utils import AverageMeter, calc_iou, get_prompts, validate
from utils.tools import copy_model, create_csv, check_grad, momentum_update, reduce_instances
from utils.eval_coco_mAp import evaluate_coco_map
from sklearn.model_selection import KFold

def train_sam(
        cfg: Box,
        fabric: L.Fabric,
        model: Model,
        anchor_model: Model,
        optimizer: _FabricOptimizer,
        scheduler: _FabricOptimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    contra_loss = ContraLoss()
    max_iou = 0.
    max_map = 0.





    for epoch in range(1, cfg.num_epochs + 1):



        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        anchor_losses = AverageMeter()
        contra_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()
        num_iter = len(train_dataloader)

        for iter, data in enumerate(train_dataloader):

            data_time.update(time.time() - end)
            images_weak, images_strong, bboxes, gt_masks = data
            batch_size = images_weak.size(0)
            num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
            if num_insts > cfg.max_nums:
                print(num_insts)
                bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            with torch.no_grad():
                 anchor_image_embeds, anchor_masks, anchor_iou_predictions, anchor_res_masks = anchor_model(images_weak,
                                                                                                            prompts)

            soft_image_embeds, soft_masks, soft_iou_predictions, soft_res_masks = model(images_weak, prompts)  # teacher
            pred_image_embeds, pred_masks, iou_predictions, pred_res_masks = model(images_strong, prompts)  # student

            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            loss_anchor = torch.tensor(0., device=fabric.device)
            loss_contra = torch.tensor(0., device=fabric.device)

            for i, (pred_mask, soft_mask, anchor_mask, iou_prediction) in enumerate(
                    zip(pred_masks, soft_masks, anchor_masks, iou_predictions)):
                anchor_mask = (anchor_mask > 0.).float()
                loss_contra += contra_loss(soft_image_embeds[i], anchor_image_embeds[i],
                                           soft_res_masks[i].clone().detach(), anchor_res_masks[i].clone().detach())

                loss_anchor += (0.5 * dice_loss(pred_mask, anchor_mask) + 0.5 * dice_loss(soft_mask, anchor_mask))

                soft_mask = (soft_mask > 0.).float()
                loss_focal += focal_loss(pred_mask, soft_mask, num_masks)
                loss_dice += dice_loss(pred_mask, soft_mask, num_masks)
                batch_iou = calc_iou(pred_mask, soft_mask)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            loss_total = 20. * loss_focal + loss_dice + loss_iou + loss_anchor + loss_contra
            fabric.backward(loss_total)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            batch_time.update(time.time() - end)
            end = time.time()

            # momentum_update(model, anchor_model, momentum=cfg.ema_rate)

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            anchor_losses.update(loss_anchor.item(), batch_size)
            contra_losses.update(loss_contra.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            fabric.print(f'Epoch: [{epoch}][{iter + 1}/{len(train_dataloader)}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                         f' | Anchor Loss [{anchor_losses.val:.4f} ({anchor_losses.avg:.4f})]'
                         f' | Contrast Loss [{contra_losses.val:.4f} ({contra_losses.avg:.4f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

            loss_logger = {"Focal Loss": focal_losses.avg, "Dice Loss": dice_losses.avg,
                           "IoU Loss": iou_losses.avg, "Anchor Loss": anchor_losses.avg,
                           "Contrast Loss": contra_losses.avg, "Total Loss": total_losses.avg}
            fabric.log_dict(loss_logger, num_iter * (epoch - 1) + iter)
            torch.cuda.empty_cache()

        if epoch % cfg.eval_interval == 0:
            # iou, f1_score = validate(fabric, cfg, model, val_dataloader, cfg.name, epoch)


            map, map_50, map_75 = evaluate_coco_map(fabric, cfg, model, val_dataloader, cfg.name, epoch)
            if map > max_map:
                state = {"model": model, "optimizer": optimizer}
                fabric.save(os.path.join(cfg.out_dir, "save", "last-ckpt.pth"), state)
                max_map = map


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

    optimizer = torch.optim.SGD(model.model.parameters(), lr=cfg.opt.learning_rate,momentum=cfg.opt.momentum, weight_decay=cfg.opt.weight_decay)
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





    data_root=cfg.datasets.coco.root_dir
    transform = ResizeAndPad(1024)
    out_root=cfg.out_dir





    best_map=0
    best_fold=cfg.start_fold
    for fold in range(cfg.start_fold,5):
        print(f"Training fold {fold + 1}/5")
        cfg.out_dir=os.path.join(out_root,f'fold_{fold}')
        if fabric.global_rank == 0:
            os.makedirs(os.path.join(cfg.out_dir, "save"), exist_ok=True)
            create_csv(os.path.join(cfg.out_dir, "metrics.csv"), csv_head=cfg.csv_keys)
        print(f'out_dir={cfg.out_dir}')


        train_dataset=COCODataset(
            cfg,
            root_dir=os.path.join(data_root,f'fold_{fold}'),
            transform=transform,
            dataset_type='train',
            if_self_training=True,
        )
        val_dataset= COCODataset(
            cfg,
            root_dir=os.path.join(data_root,f'fold_{fold}'),
            transform=transform,
            dataset_type='val',
        )

        train_loader =DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn_soft,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn
        )

        train_loader = fabric._setup_dataloader(train_loader)
        val_loader = fabric._setup_dataloader(val_loader)


        with fabric.device:
            model = Model(cfg)
            model.setup()
        optimizer, scheduler = configure_opt(cfg, model)
        if cfg.resume and cfg.resume_dir is not None:
            full_checkpoint = fabric.load(cfg.resume_dir)
            model.load_state_dict(full_checkpoint["model"])
            optimizer.load_state_dict(full_checkpoint["optimizer"])

        model, optimizer = fabric.setup(model, optimizer)
        anchor_model = copy_model(model)
        train_sam(cfg, fabric, model, anchor_model, optimizer, scheduler, train_loader, val_loader)
        map,_,_=evaluate_coco_map(fabric, cfg, model, val_loader, name=cfg.name, epoch=cfg.num_epochs)
        if map>best_map:
            best_map = map
            best_fold=fold
        del model, anchor_model, train_loader, val_loader




    with fabric.device:
        best_model = Model(cfg)
        best_model.setup()

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


    full_checkpoint = fabric.load(os.path.join(out_root,f'fold_{best_fold}','save','last-ckpt.pth'))
    best_model.load_state_dict(full_checkpoint["model"])

    fabric.print(f'the best model:{best_fold}')

    evaluate_coco_map(fabric, cfg, best_model, test_loader, name=cfg.name, epoch=cfg.num_epochs)
    # validate(fabric, cfg, anchor_model, val_data, name=cfg.name, epoch=0)



if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

    print("avaiable:", torch.cuda.device_count())

    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--resume_dir', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--batch_size', default=0, type=int, help='batch_size')
    parser.add_argument('--adapted_img_encoder', default=False, type=bool, help='adapted_img_encoder')

    args = parser.parse_args()

    if args.batch_size != 0:
        cfg.batch_size = args.batch_size
    if args.resume_dir != '':
        cfg.resume_dir = args.resume_dir
    cfg.adapted_img_encoder = args.adapted_img_encoder

    import torch

    main(cfg)
    #print(torch.cuda.is_available())
    torch.cuda.empty_cache()