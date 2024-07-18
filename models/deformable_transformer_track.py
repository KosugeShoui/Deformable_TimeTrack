# Modified by Peize Sun, Rufeng Zhang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
import numpy as np 
import matplotlib.pyplot as plt
from timesformer.models.vit import TimeSformer
import os
import cv2
import torchvision.transforms as transforms
import tqdm


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 checkpoint_enc_ffn=False, checkpoint_dec_ffn=False,time_attn = False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points,
                                                          checkpoint_enc_ffn)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points,
                                                          checkpoint_dec_ffn)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        self.decoder_track = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        
        #Time Attention layer
        self.time_attn = time_attn
        self.out_extent = nn.Linear(768,256)
        #self.conv2d = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1)

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def normalize_tensor(self,tensor):
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor
    
    def normalize_tensor_rev(self,tensor):
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        normalized_tensor = 1 - normalized_tensor
        return normalized_tensor
    
    def normalize_14x14(self, tensor):
        # 各チャネルごとに正規化を行い、新しいテンソルを格納するリストを作成
        normalized_channels = []
        
        for c in range(tensor.shape[-1]):
            # それぞれのチャネルに対して正規化を行う
            min_val = tensor[:, :, :, c].min()
            max_val = tensor[:, :, :, c].max()
            normalized_channel = (tensor[:, :, :, c] - min_val) / (max_val - min_val)
            normalized_channels.append(normalized_channel)
        
        # 正規化されたチャネルを結合して新しいテンソルを作成
        normalized_tensor = torch.stack(normalized_channels, dim=-1)
        
        return normalized_tensor
    
    def scale_tensor(self,tensor, min_val, max_val):
        # テンソルの最小値と最大値を取得
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * (max_val - min_val) + min_val
        
        return scaled_tensor

    def threshold_array(self,arr, threshold,min):
        arr[arr <= threshold] = min
        return arr
    
    #nestから通常のtensorにする関数
    def nest2tensor(self,samples,tensor_type):
        samples.tensors = samples.tensors.type(tensor_type)
        return samples.tensors
    #tensorを結合する関数 for timesformer
    def stack_tensor(self,ten1,ten2,tensor_type):
        t1 = self.nest2tensor(ten1,tensor_type)
        t2 = self.nest2tensor(ten2,tensor_type)
        resize = transforms.Resize((224, 224))
        t1_resized = torch.stack([resize(img) for img in t1])
        t2_resized = torch.stack([resize(img) for img in t2])
        combine_ten = torch.stack((t1_resized, t2_resized), dim=2)
        return combine_ten

    #def forward(self, srcs, masks, pos_embeds, query_embed=None, pre_reference=None, pre_tgt=None, memory=None):
    def forward(self, srcs, time_frame,time_weight, masks, pos_embeds, query_embed=None, pre_reference=None, pre_tgt=None, memory=None):
        assert self.two_stage or query_embed is not None
        fp16 = False
        tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor
        
        #print(self.time_attn)
        if self.time_attn != None:
            #print('time')
            time_flag = True
        else:
            time_flag = False
        
            
        # prepare input for encoder
        src_flatten = []
        time_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        src_shape_list = []
        
        #Time-Module
        if time_flag :
            #[batch,3,2,height,width]
            """
            time_frame_sub = time_frame[3,:,:,:].to('cpu').detach().numpy().copy()

            # 画像をnumpy配列に変換
            image1 = time_frame_sub[:, 0, :, :].transpose(1, 2, 0)
            image2 = time_frame_sub[:, 1, :, :].transpose(1, 2, 0)

            # 画像をプロットして横並びに表示
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(image1)
            axes[0].axis('off')
            axes[0].set_title('Image 1')
            axes[1].imshow(image2)
            axes[1].axis('off')
            axes[1].set_title('Image 2')
            plt.tight_layout()
            plt.savefig('horizontal_images.png')
            """
            """
            map1 =map1[:,:3,:,:]
            map3 = map1[3,:,:].to('cpu').detach().numpy().copy()
            map3 =np.transpose(map3,(1,2,0))
            print(map3.shape)
            plt.imshow(map3)
            plt.savefig('map3.png')
            """
            
            time_memory = self.time_attn(time_frame)
            time_memory = time_memory[:,::2,:]
            #time_memory = time_memory.view(1, 196, 256, 3).mean(dim=-1)
            time_memory = self.out_extent(time_memory)
            #[batch , 196, 256]
     
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            
            bs, c, h, w = src.shape
            src_shape_list.append([h,w])
            #print(src_shape_list)
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            #print(src_shape_list)
            """
            if lvl == 0:
                src_sub2 = src[bs-1,:,:,:].to('cpu').detach().numpy().copy()
                src_sub2 = np.mean(src_sub2[:,:,:],axis=0)
                src_sub2 = self.normalize_tensor_rev(src_sub2)
                plt.imshow(src_sub2,cmap='jet')
                plt.savefig('w_weightedmap.png')
            """
            #print(lvl)
            # Feature Map + Resized Attention Weight or F * Attention Weight
            if time_flag:
                time_memory_map = time_memory.view(bs,14,14,256)
                time_memory_map = self.normalize_14x14(time_memory_map)
                # Attention Weight Threshold [0.3.0.5.0.7]
                time_memory_map = F.interpolate(time_memory_map.permute(0, 3, 1, 2), size=(h, w), mode='bilinear', align_corners=False)
                # patch selection 
                #time_memory_map = self.threshold_array(time_memory_map,0.7)
                time_memory_map = time_memory_map.permute(0, 1, 2, 3)
                #print(time_memory_map.shape)
                # Scaling for src matching time_memory_map
                #time_memory_map = self.scale_tensor(time_memory_map, 0 , 3)
                
                """
                if lvl == 0:
                #for i in range(256):
                #visual data prepare
                #time_memory_map_sub = time_memory_map[bs-1,:,:,:].to('cpu').detach().numpy().copy()
                    time_memory_map_sub = time_memory_map[bs-1,:,:,:].to('cpu').detach().numpy().copy()
                    #print('Time Mem Scale = ',np.max(time_memory_map_sub),np.min(time_memory_map_sub))
                    #time_memory_map_sub = time_memory_map_sub
                    #time_memory_map_sub = 1 - time_memory_map_sub
                    time_memory_map_sub = np.mean(time_memory_map_sub,axis=0)
                    save_path = 'w_eval_time_train_11'
                    os.makedirs(save_path,exist_ok=True)
                    list_num = len(os.listdir(save_path))
                    #1 time memory
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    plt.imshow(time_memory_map_sub,cmap='jet')
                    plt.axis('tight')
                    plt.axis('off')
                    plt.savefig(save_path + '/time_weight_{}.png'.format(list_num+1),bbox_inches='tight',pad_inches=0)
                """
                
                
                
                
                # memory flatten 
                # 最も大きい特徴マップにのみAttention Weightを加える
                if lvl == 0:
                    time_memory_map = time_memory_map.flatten(2).transpose(1, 2)
                else:
                    time_memory_map = time_memory_map.flatten(2).transpose(1, 2)
                    time_memory_map = time_memory_map * 0
                    #print(time_memory_map)
                    
                time_flatten.append(time_memory_map)
               
            # Normal Phase
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        
        # Final flatten phase
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        if memory is None:
            memory = self.encoder(src_flatten, spatial_shapes, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
            #time track
            if time_flag:
                time_flatten = torch.cat(time_flatten, 1)
                #print(time_flatten.shape)
                #min_vals = memory.min(dim=1, keepdim=True).values
                #max_vals = memory.max(dim=1, keepdim=True).values
                #time_flatten2  = time_flatten * (max_vals - min_vals) + min_vals
                #max_values = torch.max(time_flatten, dim=1).values
                #min_values = torch.min(time_flatten, dim=1).values
                #print("Max values along feature dimension:\n", max_values,max_vals)
                #print("Min values along feature dimension:\n", min_values,min_vals)
                #memory2 = memory
                memory = memory + 0.01 * time_weight * time_flatten
                #Encode後のAttentionWeigntの可視化
                """
                for i in tqdm.tqdm(range(256)):
                
                    memory_h,memory_w = src_shape_list[0][0],src_shape_list[0][1]
                    memory_sub = memory[:,:memory_h*memory_w,:]
                    src_sub = memory_sub[bs-1,:,:].to('cpu').detach().numpy().copy()
                    src_sub = src_sub.reshape(memory_h,memory_w,256)
                    #src_sub = src_sub[:,:,]
                    
                    src_sub = src_sub[:,:,i]
                    #src_sub = self.threshold_array(src_sub,0.5,np.min(src_sub))
                    #print('Mem Scale = ',np.max(src_sub),np.min(src_sub))
                    src_sub = self.normalize_tensor(src_sub)
                    save_path = 'w_eval_timemem_train_13'
                    os.makedirs(save_path,exist_ok=True)
                    list_num = len(os.listdir(save_path))
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    plt.imshow(src_sub,cmap='jet')
                    plt.axis('tight')
                    plt.axis('off')
                    plt.savefig(save_path +'/w_eval_encode_{}.png'.format(list_num+1),bbox_inches='tight',pad_inches=0)
                """
                
                
                
            
        # prepare input for decoder
        bs, _, c = memory.shape
        if pre_reference is not None:
            if self.two_stage and self.training:
                output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

                # hack implementation for two-stage Deformable DETR
                enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
                enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
                
            tgt = pre_tgt
            reference_points = pre_reference
            init_reference_out = reference_points

            query_embed = None
            # decoder
            hs, inter_references = self.decoder_track(tgt, reference_points, memory,
                                                      spatial_shapes, valid_ratios, query_embed, mask_flatten)
            inter_references_out = inter_references
        else:
            if self.two_stage:
                output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

                # hack implementation for two-stage Deformable DETR
                enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
                enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

                topk = self.two_stage_num_proposals
                topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
                topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
                topk_coords_unact = topk_coords_unact.detach()
                reference_points = topk_coords_unact.sigmoid()
                init_reference_out = reference_points
                pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            else:
                query_embed, tgt = torch.split(query_embed, c, dim=1)
                query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
                tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
                reference_points = self.reference_points(query_embed).sigmoid()
                init_reference_out = reference_points
            query_embed = None
            # decoder
            hs, inter_references = self.decoder(tgt, reference_points, memory,
                                                spatial_shapes, valid_ratios, query_embed, mask_flatten)
            inter_references_out = inter_references
        
        if self.two_stage and self.training:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, memory
        
        return hs, init_reference_out, inter_references_out, None, None, memory


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 checkpoint_ffn=False):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # use torch.utils.checkpoint.checkpoint to save memory
        self.checkpoint_ffn = checkpoint_ffn

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        if self.checkpoint_ffn:
            src = torch.utils.checkpoint.checkpoint(self.forward_ffn, src)
        else:
            src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 checkpoint_ffn=False):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # use torch.utils.checkpoint.checkpoint to save memory
        self.checkpoint_ffn = checkpoint_ffn

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        if self.checkpoint_ffn:
            tgt = torch.utils.checkpoint.checkpoint(self.forward_ffn, tgt)
        else:
            tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args,time_attn):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        checkpoint_enc_ffn=args.checkpoint_enc_ffn,
        checkpoint_dec_ffn=args.checkpoint_dec_ffn,
        time_attn=time_attn
    )