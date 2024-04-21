
import lightning as L
import torch
from box import Box
from torch.utils.data import DataLoader
from model import Model
from utils.eval_utils import get_prompts,AverageMeter
from torchmetrics.detection import MeanAveragePrecision
from utils.tools import write_csv
import os
def evaluate_coco_map(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, epoch: int = 0):
    model.eval()

    metric = MeanAveragePrecision(iou_type="segm")
    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks,_= data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            _, pred_masks, ious, _ = model(images, prompts)

            preds=[]
            targets=[]
            for i in range(num_images):
                    masks=torch.sigmoid(pred_masks[i])
                    masks=masks>0.5

                    pred={
                        'masks':masks,
                        'scores':ious[i].reshape(-1),
                        'labels':torch.IntTensor([1 for i in range(masks.size()[0])])
                    }
                    target={
                        'masks':torch.Tensor(gt_masks[i].byte()),
                        'labels': torch.IntTensor([1 for i in range(masks.size()[0])])
                    }
                    preds.append(pred)
                    targets.append(target)
            metric.update(preds, targets)
            result = metric.compute()
            map,map_50,map_75=result['map'],result['map_50'],result['map_75']
            fabric.print(
                f'Val: [{epoch}] - [{iter+1}/{len(val_dataloader)}]: map: [{map:.3f}] -- map_50: [{map_50:.3f}] -- map_75: [{map_75:}]'
            )

            torch.cuda.empty_cache()
    result = metric.compute()
    map, map_50, map_75 = result['map'], result['map_50'], result['map_75']
    fabric.print(
        f'Validation: [{epoch}] - [{iter+1}/{len(val_dataloader)}]: map: [{map:.3f}] -- map_50: [{map_50:.3f}] -- map_75: [{map_75:3f}'
    )

    csv_dict = {"Name": name, "Prompt": cfg.prompt, "map": f"{map:.4f}", "map_50": f"{map_50:.4f}", "map_75":f"{map_75}","epoch":epoch}

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, "metrics.csv"), csv_dict, csv_head=cfg.csv_keys)
    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, "metrics.csv"), csv_dict, csv_head=cfg.csv_keys)
    model.train()
    return  map, map_50, map_75