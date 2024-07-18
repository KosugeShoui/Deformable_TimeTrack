# Modified by Peize Sun, Rufeng Zhang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T


def multiply_loss_giou_values(weight_dict, factor):
    for key in weight_dict:
        if 'loss_giou' in key:
            weight_dict[key] = factor
    return weight_dict

# Sigmoid Base Scheduler : proposed method
def sigmoid(x):
    return  1 / (1 + np.exp(-x))

def sigmoid_base_sche(initial_weight,final_weight,num_epochs):
    
    x = np.linspace(0, num_epochs, num_epochs)  
    scaled_x = 12 * (x / num_epochs) - 6
    y = sigmoid(scaled_x)
    scaled_y = (initial_weight-final_weight) * (1 - y) + final_weight 
    return scaled_y

def normalize_to_rgb(array):
        array = np.asarray(array)
        array = np.clip(array, 0, 1)
        rgb_array = (array * 255).astype(np.uint8)
        
    
        return rgb_array

def nest2tensor(samples,tensor_type):
        samples.tensors = samples.tensors.type(tensor_type)
        return samples.tensors

def save_img(samples,tensor_type,name):
        samples.tensors = samples.tensors.type(tensor_type)
        samples_sub = samples.tensors
        unnormalize = T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    
        
        for i,img in enumerate(samples_sub):
            img = unnormalize(img)
            img = img.to('cpu').detach().numpy().copy()
            img = normalize_to_rgb(img.transpose(1,2,0))
            #print('img_shape = ',img.shape)
            plt.imshow(img)
            plt.savefig(name + '_{}.png'.format(i))  
    


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, scaler: torch.cuda.amp.GradScaler,
                    epoch: int, new_weight_dict : dict, time_weight : float , max_norm: float = 0,fp16=False):
    fp16 = False
    tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor
    
    model.train()
    criterion.train()
    tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = '\n --------- Epoch : [{}] ----------\n'.format(epoch + 1)
    print_freq = 300

    
    
    #data_iter = iter(data_loader)
    #first_batch = next(data_iter)
    #i,j,k = first_batch
    #save_img(i,tensor_type,'w_input_frame/sample_train')
    
    
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets, pre_samples = prefetcher.next()
    #print(pre_samples)
    

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        samples.tensors = samples.tensors.type(tensor_type)
        samples.mask = samples.mask.type(tensor_type)
        # データ確認
        #save_img(samples,tensor_type,'w_input_frame/sample')
        #save_img(pre_samples,tensor_type,'w_input_frame/pre_sample')
        #print(targets[0]['frame_id'])

        with torch.cuda.amp.autocast(enabled=fp16):
            # input frames
            outputs, pre_outputs, pre_targets = model([samples, targets, pre_samples],time_weight)
            loss_dict = criterion(outputs, targets, pre_outputs, pre_targets)
            
            #weight_dict = criterion.weight_dict
            #schedule weight
            weight_dict = new_weight_dict
            #print('giou weight = ',weight_dict['loss_giou'])
            #print('\n')
            #print(new_weight_dict)
            
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets, presamples = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('\n')
    print("-------------- Averaged stats ------------ \n ", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def nest2tensor(samples,tensor_type):
        samples.tensors = samples.tensors.type(tensor_type)
        return samples.tensors

def save_combined_image(tensor1, tensor2, filename='combined_image.png'):
    fp16 = False
    tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor
        # training

    # GPUからCPUに移す
    tensor1 = nest2tensor(tensor1,tensor_type)
    tensor2 = nest2tensor(tensor2,tensor_type)
    tensor1_cpu = tensor1.cpu()
    tensor2_cpu = tensor2.cpu()

    # テンソルをnumpy配列に変換
    np_tensor1 = tensor1_cpu.squeeze(0).permute(1, 2, 0).numpy()  # (高さ, 幅, チャンネル)の形に変換
    np_tensor2 = tensor2_cpu.squeeze(0).permute(1, 2, 0).numpy()  # (高さ, 幅, チャンネル)の形に変換

    # 画像データを正規化
    np_tensor1 = (np_tensor1 - np_tensor1.min()) / (np_tensor1.max() - np_tensor1.min())
    np_tensor2 = (np_tensor2 - np_tensor2.min()) / (np_tensor2.max() - np_tensor2.min())

    # matplotlibを使って2つの画像を横に並べて表示
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np_tensor1)
    ax[1].imshow(np_tensor2)

    # 画像を保存
    plt.savefig(filename)


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, tracker=None, 
             phase='train', det_val=False, fp16=False):
    tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor
    model.eval()
#     criterion.eval()
       
    metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    res_tracks = dict()
    pre_embed = None
    #count = 1
    for samples, targets , past_samples in metric_logger.log_every(data_loader, 10, header):
        # pre process for track.
        if tracker is not None:
            if phase != 'train':
                assert samples.tensors.shape[0] == 1, "Now only support inference of batchsize 1." 
            frame_id = targets[0].get("frame_id", None)
            assert frame_id is not None
            frame_id = frame_id.item()
            if frame_id == 1:
                tracker.reset_all()
                pre_embed = None
                
        samples = samples.to(device)
        samples.tensors = samples.tensors.type(tensor_type)
        samples.mask = samples.mask.type(tensor_type)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.cuda.amp.autocast(enabled=fp16):
            if det_val:
                outputs = model(samples)
            else:
                # Time Frame Input 
                outputs, pre_embed = model(samples, past_samples, pre_embed)
                    #count += 1
                    #pre_samples = samples
                    #print(torch.equal(nest2tensor(pre_samples,tensor_type), nest2tensor(samples,tensor_type)))
                
                    #print(torch.equal(nest2tensor(pre_samples,tensor_type), nest2tensor(samples,tensor_type)))
                    #save_combined_image(pre_samples,samples,'combined_{}.png'.format(count))
                    #outputs, pre_embed = model(samples, pre_embed)
                    #count += 1
                    #pre_samples = samples
                    
                    
            
#             loss_dict = criterion(outputs, targets)
            
#         weight_dict = criterion.weight_dict

#         reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                       for k, v in loss_dict_reduced.items()}
#         metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
#                              **loss_dict_reduced_scaled,
#                              **loss_dict_reduced_unscaled)
#         metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        # post process for track.
        if tracker is not None:
            if frame_id == 1:
                res_track = tracker.init_track(results[0])
            else:
                res_track = tracker.step(results[0])
            res_tracks[targets[0]['image_id'].item()] = res_track

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator, res_tracks
