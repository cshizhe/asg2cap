import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.configbase
import framework.ops


'''
Vanilla Encoder: embed nd array (batch_size, ..., dim_ft)
  - EncoderConfig
  - Encoder

Multilayer Perceptrons: feed forward networks + softmax
  - MLPConfig
  - MLP
'''

class EncoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.dim_fts = [2048]
    self.dim_embed = 512
    self.is_embed = True
    self.dropout = 0
    self.norm = False
    self.nonlinear = False

  def _assert(self):
    if not self.is_embed:
      assert self.dim_embed == sum(self.dim_fts)

class Encoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    if self.config.is_embed:
      self.ft_embed = nn.Linear(sum(self.config.dim_fts), self.config.dim_embed)
    self.dropout = nn.Dropout(self.config.dropout)

  def forward(self, fts):
    '''
    Args:
      fts: size=(batch, ..., sum(dim_fts))
    Returns:
      embeds: size=(batch, dim_embed)
    '''
    embeds = fts
    if self.config.is_embed:
      embeds = self.ft_embed(embeds)
    if self.config.nonlinear:
      embeds = F.relu(embeds)
    if self.config.norm:
      embeds = framework.ops.l2norm(embeds) 
    embeds = self.dropout(embeds)
    return embeds

