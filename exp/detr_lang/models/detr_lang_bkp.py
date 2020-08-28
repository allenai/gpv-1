# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

import utils.box_ops as box_ops
from utils.detr_misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .transformer import build_transformer
from utils.matcher import build_matcher


class DETRLang(nn.Module):
    """ This is the DETR module that performs object detection with language input"""
    def __init__(self, backbone, transformer, fusion_transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            fusion_transformer: transformer for fusing language with visual features
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.add_module('fusion_transformer',fusion_transformer)
        #self.fusion_transformer = fusion_transformer
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        #self.class_embed = MLP(hidden_dim, hidden_dim, num_classes+1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.vision_token = nn.Parameter(torch.randn([hidden_dim]),requires_grad=True)
        self.lang_token = nn.Parameter(torch.randn([hidden_dim]),requires_grad=True)

    def forward(self, samples: NestedTensor, query_encodings: torch.Tensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
            
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        pretext_hs = hs

        L = hs.size(0)
        B,T,D = query_encodings.size()
        query_encodings = query_encodings.view(1,B,T,D).repeat(L,1,1,1)
        vision_token = self.vision_token.view(1,1,1,D)
        lang_token = self.lang_token.view(1,1,1,D)
        # hs = torch.cat((hs+vision_token,query_encodings+lang_token),2).view(L*B,-1,D)
        # hs = self.fusion_transformer(hs[:,:self.num_queries],hs[:,self.num_queries:])
        hs = torch.cat((hs+vision_token,query_encodings+lang_token),2).view(L*B,-1,D).permute(1,0,2)
        # hs = self.fusion_transformer(hs[:self.num_queries],hs[self.num_queries:])
        hs = self.fusion_transformer(hs)
        hs = hs.permute(1,0,2).view(L,B,-1,D)
        hs = hs[:,:,:self.num_queries]
        hs = hs #+pretext_hs

        #hs = torch.cat((hs[-1],query_encodings),1).view(1,B,-1,D)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



def build_fusion_transformer(cfg):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=cfg.hidden_dim,
        dropout=cfg.dropout,
        nhead=cfg.nheads)
    
    return nn.TransformerEncoder(encoder_layer,cfg.num_fusion_layers)

    # decoder_layer = nn.TransformerDecoderLayer(
    #     d_model=cfg.hidden_dim,
    #     dropout=cfg.dropout,
    #     nhead=cfg.nheads)
    
    # return nn.TransformerDecoder(decoder_layer,cfg.num_fusion_layers)


def create_detr_lang(cfg):
    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)
    fusion_transformer = build_fusion_transformer(cfg)
    
    model = DETRLang(
        backbone,
        transformer,
        fusion_transformer,
        num_classes=cfg.num_classes,
        num_queries=cfg.num_queries,
        aux_loss=cfg.aux_loss)
    
    return model