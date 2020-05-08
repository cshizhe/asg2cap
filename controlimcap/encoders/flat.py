import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import caption.encoders.vanilla

'''
EncoderConfig is the same as encoder.vanilla.EncoderConfig
'''

def gen_order_embeds(max_len, dim_ft):
  order_embeds = np.zeros((max_len, dim_ft))
  position = np.expand_dims(np.arange(0, max_len - 1).astype(np.float32), 1)
  div_term = np.exp(np.arange(0, dim_ft, 2) * -(math.log(10000.0) / dim_ft))
  order_embeds[1:, 0::2] = np.sin(position * div_term)
  order_embeds[1:, 1::2] = np.cos(position * div_term)
  return order_embeds

class EncoderConfig(caption.encoders.vanilla.EncoderConfig):
  def __init__(self):
    super().__init__()
    self.dim_fts = [2048]
    self.dim_embed = 512
    self.is_embed = True
    self.dropout = 0
    self.norm = False
    self.nonlinear = False
    self.num_node_types = 3
  
class Encoder(caption.encoders.vanilla.Encoder):
  def __init__(self, config):
    super().__init__(config)

    dim_fts = sum(self.config.dim_fts)
    self.node_embedding = nn.Embedding(self.config.num_node_types, dim_fts)

    self.register_buffer('attr_order_embeds',
      torch.FloatTensor(gen_order_embeds(20, dim_fts)))
    
  def forward(self, fts, node_types, attr_order_idxs):
    '''
    Args:
      fts: size=(batch, seq_len, dim_ft)
      node_types: size=(batch, seq_len)
      attr_order_idxs: size=(batch, seq_len)
    Returns:
      embeds: size=(batch, seq_len, dim_embed)
    '''
    node_embeds = self.node_embedding(node_types)
    node_embeds = node_embeds + self.attr_order_embeds[attr_order_idxs]

    inputs = fts * node_embeds
    embeds = super().forward(inputs)

    return embeds
