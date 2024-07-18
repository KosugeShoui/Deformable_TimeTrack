# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
fp16 = False
tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor

def normalize_to_rgb(array):
        array = np.asarray(array)
        array = np.clip(array, 0, 1)
        rgb_array = (array * 255).astype(np.uint8)
        return rgb_array

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
            print('img_shape = ',img.shape)
            plt.imshow(img)
            plt.savefig(name + '_{}.png'.format(i))  
    

def to_cuda(samples, targets, pre_samples, device):
    #print(type(samples),type(pre_samples))
    #samples.tensors = samples.tensors.type(tensor_type)
    #print(samples.tensors.shape)
    #save_img(samples,tensor_type,'w_input_prefetch')
    try:
        #print('try ok')
        pre_samples_sub = torch.stack(pre_samples, dim=0)
        pre_samples = pre_samples_sub
        pre_samples = pre_samples.to(device, non_blocking=True)
    except:
        #print('presample None')
        pre_samples_sub = None
    
    #print(pre_samples_sub.shape)
    
    samples = samples.to(device, non_blocking=True)
    
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets, pre_samples

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        fp16 = False
        tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor
        try:
            self.next_samples, self.next_targets, self.next_presamples = next(self.loader)
            #self.next_samples.tensors = self.next_samples.tensors.type(tensor_type)
            #print(self.next_samples.tensors.shape) -->[4,3,833,1066]
            #print(type(self.next_presamples))
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            self.next_presamples = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            #self.next_presamples = torch.Tensor(self.next_presamples)
            #print(self.next_presamples)
            #print(self.next_presamples.tensors.shape)
            self.next_samples, self.next_targets,self.next_presamples = to_cuda(self.next_samples, self.next_targets,self.next_presamples, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            pre_samples = self.next_presamples
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets, pre_samples = next(self.loader)
                samples, targets, pre_samples = to_cuda(samples, targets,pre_samples, self.device)
            except StopIteration:
                samples = None
                targets = None
                pre_samples = None
        return samples, targets, pre_samples
