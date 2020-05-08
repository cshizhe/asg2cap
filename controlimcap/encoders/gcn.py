import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.configbase
import framework.ops

from controlimcap.encoders.flat import gen_order_embeds


class RGCNLayer(nn.Module):
  def __init__(self, in_feat, out_feat, num_rels, 
    bias=None, activation=None, dropout=0.0):
    super().__init__()
    self.in_feat = in_feat
    self.out_feat = out_feat
    self.num_rels = num_rels
    self.bias = bias
    self.activation = activation

    self.loop_weight = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
    nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

    self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat, self.out_feat))
    nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
    
    if self.bias:
      self.bias = nn.Parameter(torch.Tensor(self.out_feat))
      nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

    self.dropout = nn.Dropout(dropout)

  def forward(self, attn_fts, rel_edges):
    '''Args:
      attn_fts: (batch_size, max_src_nodes, in_feat)
      rel_edges: (batch_size, num_rels, max_tgt_nodes, max_srt_nodes)
    Retunrs:
      node_repr: (batch_size, max_tgt_nodes, out_feat)
    '''
    loop_message = torch.einsum('bsi,ij->bsj', attn_fts, self.loop_weight)
    loop_message = self.dropout(loop_message)

    neighbor_message = torch.einsum('brts,bsi,rij->btj', rel_edges, attn_fts, self.weight)
    
    node_repr = loop_message + neighbor_message
    if self.bias:
      node_repr = node_repr + self.bias
    if self.activation:
      node_repr = self.activation(node_repr)

    return node_repr


class RGCNEncoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.dim_input = 2048
    self.dim_hidden = 512
    self.num_rels = 6
    self.num_hidden_layers = 1
    self.max_attn_len = 10
    self.self_loop = True
    self.dropout = 0.
    self.num_node_types = 3
    self.embed_first = False

class RGCNEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    if self.config.embed_first:
      self.first_embedding = nn.Sequential(
        nn.Linear(self.config.dim_input, self.config.dim_hidden),
        nn.ReLU())

    self.layers = nn.ModuleList()
    dim_input = self.config.dim_hidden if self.config.embed_first else self.config.dim_input
    for _ in range(self.config.num_hidden_layers):
      h2h = RGCNLayer(dim_input, self.config.dim_hidden, self.config.num_rels,
        activation=F.relu, dropout=self.config.dropout)
      dim_input = self.config.dim_hidden
      self.layers.append(h2h)

  def forward(self, attn_fts, rel_edges):
    if self.config.embed_first:
      attn_fts = self.first_embedding(attn_fts)

    for layer in self.layers:
      attn_fts = layer(attn_fts, rel_edges)

    return attn_fts
    

class RoleRGCNEncoder(RGCNEncoder):
  def __init__(self, config):
    super().__init__(config)

    self.node_embedding = nn.Embedding(self.config.num_node_types, 
      self.config.dim_input)

    self.register_buffer('attr_order_embeds', 
      torch.FloatTensor(gen_order_embeds(20, self.config.dim_input)))

  def forward(self, attn_fts, node_types, attr_order_idxs, rel_edges):
    '''Args:
      (num_src_nodes = num_tgt_nodes)
      - attn_fts: (batch_size, num_src_nodes, in_feat)
      - rel_edges: (num_rels, num_tgt_nodes, num_src_nodes) 
      - node_types: (batch_size, num_src_nodes)
      - attr_order_idxs: (batch_size, num_src_nodes)
    '''
    node_embeds = self.node_embedding(node_types)
    node_embeds = node_embeds + self.attr_order_embeds[attr_order_idxs]

    input_fts = attn_fts * node_embeds

    return super().forward(input_fts, rel_edges)



