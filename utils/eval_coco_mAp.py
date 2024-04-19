import os
import torch
import lightning as L
from box import Box
from torch.utils.data import DataLoader
from model import Model
from utils.sample_utils import get_point_prompts
from utils.eval_utils import get_prompts
from utils.tools import decode_mask
from tqdm import tqdm

from pycocotools import mask
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_predictions = {
    "annotations": [],
    "categories": [{"id": 1, "name": "tree"}]
}

def encode_mask(tensor_mask):
    # 确保 tensor 在 CPU 上，并转换为 numpy 数组
    if tensor_mask.is_cuda:
        tensor_mask = tensor_mask.cpu()
    numpy_mask = tensor_mask.numpy()

    # 如果掩码是概率或其他格式，这里可以应用阈值来确保它是二进制的
    # 例如，将所有大于 0.5 的值设为 1，其他设为 0
    numpy_mask = (numpy_mask > 0.5).astype(np.uint8)

    # 将 NumPy 数组转换为 Fortran 风格
    fortran_mask = np.asfortranarray(numpy_mask)
    
    # 使用 RLE 编码
    encoded = mask.encode(fortran_mask)
    # 不需要解码 'counts' 字段
    return encoded

def eval_coco_mAp(cfg: Box, model: Model, val_dataloader: DataLoader):
    model.eval()
    annotation_id=1;
    
    with torch.no_grad():
        for data in tqdm((val_dataloader),total=len(val_dataloader)):
           
            images, bboxes, gt_masks ,img_ids= data
            num_images = images.size(0)
            
            prompts = get_prompts(cfg, bboxes, gt_masks)
            _, pred_masks, ious, _ = model(images, prompts)
            for pred_mask,iou,img_id in zip(pred_masks,ious,img_ids):
             
                masks=encode_mask(pred_mask)
                for mask in masks:
                    rle_encoded_mask = encode_mask(mask)
                    bbox = mask.toBbox(rle_encoded_mask).tolist()
                    area = mask.area(rle_encoded_mask)
                    coco_predictions["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": rle_encoded_mask,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "score": iou
                    })
                    annotation_id += 1
            
            torch.cuda.empty_cache()
            with open('sam_predictions_coco_format.json', 'w') as f:
                    json.dump(coco_predictions, f)
            gt_annotations_file=os.path.join(cfg.datasets.coco.root_dir,'annotations','instance_val.json')
            coco_gt = COCO(gt_annotations_file)  # 真实标注的文件路径
            coco_dt = coco_gt.loadRes('sam_predictions_coco_format.json')

            # 创建评估器
            coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            print('AP:', coco_eval.stats[0])  
            
            

