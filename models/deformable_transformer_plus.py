# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
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

from models.structures import Boxes, matched_boxlist_iou, pairwise_iou

from util.misc import inverse_sigmoid
from util.box_ops import box_cxcywh_to_xyxy
from models.ops.modules import MSDeformAttn
import matplotlib.pyplot as plt

#from attn_vis import visualize
import numpy as np

from timesformer.models.vit import TimeSformer
import os

"""
class TimeSformer_getattn(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = TimeSformer(img_size=224, num_classes=0, num_frames=2, 
                                    attention_type='divided_space_time',  pretrained_model='')
        self.backbone_output_dim = 768
    
    def forward(self, x):
        # xの形状は [batch_size, num_frames, channels, height, width]
        
        batch_size, channels, num_frames, height, width = x.shape
        #print('channel , height , width = ',channels, height, width)
        assert channels == 3 and height == 224 and width == 224, \
            "Input shape must be [batch_size, 3 , num_frames , 224, 224]"

        # TimeSformerに渡すためにテンソルの形状を変更
        #x = x.view(batch_size, num_frames, height, width, channels)  # [batch_size, num_frames, height, width, channels]
        #x = x.permute(0, 4, 1, 2, 3)  # [batch_size, channels, num_frames, height, width]
        #print(x.shape)

        # TimeSformerモデルに入力
        cls_token, features = self.backbone(x)
        # 特徴量を [1, 3, 256] に変形
        #print('time attn = ',features.shape)
        #[1,392,768]
        #features = features.view(batch_size, 3, 256)  # [batch_size, 3, 256]

        return features
"""

class Normalizer:
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

    

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 decoder_self_cross=True, sigmoid_attn=False,
                 extra_track_attn=False,timesformer = None):
        super().__init__()

        self.new_frame_adaptor = None
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points,
                                                          sigmoid_attn=sigmoid_attn)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, decoder_self_cross,
                                                          sigmoid_attn=sigmoid_attn, extra_track_attn=extra_track_attn)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        
        #Time Attention layer
        self.time_attn = timesformer
        #print(self.time_attn)
        self.tensor_norm = Normalizer()
        #model see

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
    
    def normalize_tensor_rev(self,tensor):
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        normalized_tensor_1 = (tensor - min_val) / (max_val - min_val)
        
        normalized_tensor = 1 - normalized_tensor_1
        
        return normalized_tensor
    
    def normalize_tensor(self,tensor):
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        
        
        return normalized_tensor
    

    # 14x14の次元を正規化する関数
    def normalize_14x14(self,tensor):
        # チャネルの次元を分離
        for c in range(tensor.shape[-1]):
            # それぞれのチャネルに対して正規化を行う
            min_val = tensor[:, :, :, c].min()
            max_val = tensor[:, :, :, c].max()
            tensor[:, :, :, c] = (tensor[:, :, :, c] - min_val) / (max_val - min_val)
        return tensor

    
    #---> from self.transformer
    def forward(self, srcs, time_frame, masks, pos_embeds, query_embed=None, ref_pts=None):
        assert self.two_stage or query_embed is not None
        
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        src_shape_list = []
        
        # Timeformer class --> ここにtimesformerからのAttentionの機構を導入したい
        #print('time frames = ',time_frame.shape)
        #[1,3,2,224,224]
        #print(time_frame.shape)
        time_memory = self.time_attn(time_frame)
        time_memory = time_memory[:,::2,:]
        #チャネル方向の平均を取る
        time_memory = time_memory.view(1, 196, 256, 3).mean(dim=-1)
        
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            src_shape_list.append([h , w])
            #multi-scale feature map
            #torch.Size([1, 256, 112, 132])
            """
            if 10 <= h <= 20 :
                f_map = src[0,0:3,:]
                f_map = f_map.to('cpu').detach().numpy().copy()
                f_map = f_map.transpose(1,2,0)
                print(f_map.shape)
                f_map = self.normalize_tensor(f_map)
                plt.clf()
                plt.imshow(f_map)
                plt.savefig('f_map_h={}.png'.format(h))
            """
            #torch.Size([1, 256, 56, 66])
            #torch.Size([1, 256, 28, 33])
            #torch.Size([1, 256, 14, 17])
            
            
            
            #attentionweightを変形
            time_memory_map = time_memory.view(1,14,14,256)
            time_memory_map = self.tensor_norm.normalize_14x14(time_memory_map)
            #特徴マップのサイズにリサイズ
            #time_memory_map.requires_grad_(False)
            time_memory_map = F.interpolate(time_memory_map.permute(0, 3, 1, 2), size=(h, w), mode='bilinear', align_corners=False)
            time_memory_map = time_memory_map.permute(0, 1, 2, 3)
            #print('time memory = ', time_memory_map.shape)
            #print('src shape = ',src.shape)
            
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            
            #src_sub = src[0,:,:,:].to('cpu').detach().numpy().copy()
            #src_sub = np.mean(src_sub,axis= 0)
            #src_sub = self.normalize_tensor_rev(src_sub)
            
            # Feature Map + Resized Attention Weight or F * Attention Weight
            src = src + time_memory_map
            #print('src shape = ',src.shape)
            
            
            """
            #visualization
            #最も解像度が大きい特徴マップにのみ適応
            if lvl == 0:
                #visual data prepare
                time_memory_map_sub = time_memory_map[0,:,:,:].to('cpu').detach().numpy().copy()
                time_memory_map_sub = np.mean(time_memory_map_sub,axis=0)
                #time_memory_map_sub = 1 - time_memory_map_sub
                
                #src_sub2 = src[0,:,:,:].to('cpu').detach().numpy().copy()
                #src_sub2 = np.mean(src_sub2, axis= 0)
                #rc_sub2 = self.normalize_tensor(src_sub2)
                #print(src_sub2)
                
                save_path = 'w_eval_attention_add_visem2'
                os.makedirs(save_path,exist_ok=True)
                
                list_num = len(os.listdir(save_path))
                save_num = list_num // 3
                
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                #1 time memory
                plt.imshow(time_memory_map_sub,cmap='viridis')
                plt.axis('tight')
                plt.axis('off')
                plt.savefig(save_path + '/time_weight_{}.png'.format(save_num),bbox_inches='tight',pad_inches=0)
                
                #2 normal src
                #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.imshow(src_sub,cmap='viridis')
                plt.axis('tight')
                plt.axis('off')
                plt.savefig(save_path + '/src_notime{}.png'.format(save_num),bbox_inches='tight',pad_inches=0)
                #3 timed src
                #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                #print('src_notime = ',src_sub)
                #print('min max = ',np.min(src_sub),np.max(src_sub))
                
                src_sub2 = src_sub * time_memory_map_sub
                src_sub2 = self.normalize_tensor(src_sub2)
                
                plt.imshow(src_sub2,cmap='viridis')
                plt.axis('tight')
                plt.axis('off')
                plt.savefig(save_path + '/src_time{}.png'.format(save_num),bbox_inches='tight',pad_inches=0)
                #print('src_time = ',src_sub2)
                #print('min max = ',np.min(src_sub2),np.max(src_sub2))
            
            """
            # end 
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        
        
        
            
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        
        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        #print(memory.shape)
        
        # prepare input for decoder
        bs, _, c = memory.shape
        
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            #print(output_memory.shape)
            #[1,flatten,256]

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
            
            if ref_pts is None:
                reference_points = self.reference_points(query_embed).sigmoid()
            else:
                reference_points = ref_pts.unsqueeze(0).repeat(bs, 1, 1).sigmoid()
            init_reference_out = reference_points
        
        #print(tgt.shape)
        #[1,flatten,256][1,300,256]
        #print(query_embed.shape)
            
        #print(tgt.shape)
        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        #debug
        #print(type(hs))
        #print(hs.shape)
        #print(hs[0,0,:,0].shape) # --. (303)
        #print(inter_references.shape)
        """
        frame_num = 0
        if hs[0,0,:,0].size(0) >= 301: #検出物体が存在する場合は300以上
            max_num = hs[0,0,:,0].size(0)
            for num,i in enumerate(hs[0,0,:]):
                ##print(i.shape)
                #print(i)
                if num >= 300 :
                    visualize(num+1,i,frame_num)
        """
                

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, self.time_attn , None, None



class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        #print(type(self.self_attn))
        
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        #print(src.shape) --> [1,23935,256]
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        #print(reference_points.shape)
        
        # Multi-Scalr Deformable Attention --> self.attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        #print(src2.shape)
        #[1,flatten,256]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        
        #print(type(src),src.shape)
        #[1,23935,256]
        #特徴マップをflattenに + channel = [1,flatten,256]
        
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

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        #print(output.shape) --> [1,23674,256]
        #output2 = output[0,0,:].cpu().detach().numpy()
        #output2 = output2.reshape(16,16)
        #plt.imshow(output2)
        #plt.savefig('encoder_last.png')
        
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        
        #print(type(reference_points))
        #print(reference_points.shape)  --> [1,23935,4,2]
        
        #layerの数 --> Attention Blockの数
        #ここでAttentionによるEncode 演算をやっている
        for _, layer in enumerate(self.layers):
            #print('output shape = ',output.shape)
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        
        #print(output.shape) --> [1,23674,256]
        
        #print(output[0,0,:].shape) --> 256
        

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, self_cross=True, sigmoid_attn=False, extra_track_attn=False):
        super().__init__()

        self.self_cross = self_cross
        self.num_head = n_heads

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

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

        # update track query_embed
        self.extra_track_attn = extra_track_attn
        if self.extra_track_attn:
            print('Training with Extra Self Attention in Every Decoder.', flush=True)
            self.update_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)

        if self_cross:
            print('Training with Self-Cross Attention.')
        else:
            print('Training with Cross-Self Attention.')

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _forward_self_attn(self, tgt, query_pos, attn_mask=None):
        if self.extra_track_attn:
            tgt = self._forward_track_attn(tgt, query_pos)

        q = k = self.with_pos_embed(tgt, query_pos)
        if attn_mask is not None:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),
                                  attn_mask=attn_mask)[0].transpose(0, 1)
        else:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        return self.norm2(tgt)

    def _forward_track_attn(self, tgt, query_pos):
        q = k = self.with_pos_embed(tgt, query_pos)
        if q.shape[1] > 300:
            tgt2 = self.update_attn(q[:, 300:].transpose(0, 1),
                                    k[:, 300:].transpose(0, 1),
                                    tgt[:, 300:].transpose(0, 1))[0].transpose(0, 1)
            tgt = torch.cat([tgt[:, :300],self.norm4(tgt[:, 300:]+self.dropout5(tgt2))], dim=1)
        return tgt

    def _forward_self_cross(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                            src_padding_mask=None, attn_mask=None):

        # self attention
        tgt = self._forward_self_attn(tgt, query_pos, attn_mask)
        # cross attention
        #print(src_spatial_shapes.shape) --> [4,2]
        #print('padding mask = ',src_padding_mask)
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

    def _forward_cross_self(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                            src_padding_mask=None, attn_mask=None):
        # cross attention
        #print(src_spatial_shapes.shape)
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # self attention
        tgt = self._forward_self_attn(tgt, query_pos, attn_mask)
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        attn_mask = None
        if self.self_cross:
            return self._forward_self_cross(tgt, query_pos, reference_points, src, src_spatial_shapes,
                                            level_start_index, src_padding_mask, attn_mask)
        return self._forward_cross_self(tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                                        src_padding_mask, attn_mask)


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
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
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

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
        return nn.ReLU(True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



def build_deforamble_transformer(args,timesformer):
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
        decoder_self_cross=not args.decoder_cross_self,
        sigmoid_attn=args.sigmoid_attn,
        extra_track_attn=args.extra_track_attn,
        timesformer = timesformer
    )


