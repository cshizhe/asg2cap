import numpy as np
import torch
import torch.nn.functional as F

import framework.configbase
import caption.encoders.vanilla
import caption.decoders.attention
import caption.models.attention

import controlimcap.encoders.gcn
import controlimcap.decoders.cfattention

MPENCODER = 'mp_encoder'
ATTNENCODER = 'attn_encoder'
DECODER = 'decoder'

class GraphModelConfig(framework.configbase.ModelConfig):
  def __init__(self):
    super().__init__()
    self.subcfgs[MPENCODER] = caption.encoders.vanilla.EncoderConfig()
    self.subcfgs[ATTNENCODER] = controlimcap.encoders.gcn.RGCNEncoderConfig()
    self.subcfgs[DECODER] = caption.decoders.attention.AttnDecoderConfig()

  def _assert(self):
    assert self.subcfgs[MPENCODER].dim_embed == self.subcfgs[DECODER].hidden_size
    assert self.subcfgs[ATTNENCODER].dim_hidden == self.subcfgs[DECODER].attn_input_size


class GraphBUTDAttnModel(caption.models.attention.BUTDAttnModel):
  def build_submods(self):
    submods = {}
    submods[MPENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[MPENCODER])
    submods[ATTNENCODER] = controlimcap.encoders.gcn.RGCNEncoder(self.config.subcfgs[ATTNENCODER])
    submods[DECODER] = caption.decoders.attention.BUTDAttnDecoder(self.config.subcfgs[DECODER])
    return submods

  def prepare_input_batch(self, batch_data, is_train=False):
    outs = {}
    outs['mp_fts'] = torch.FloatTensor(batch_data['mp_fts']).to(self.device)
    outs['attn_fts'] = torch.FloatTensor(batch_data['attn_fts']).to(self.device)
    outs['attn_masks'] = torch.FloatTensor(batch_data['attn_masks'].astype(np.float32)).to(self.device)
    # build rel_edges tensor
    batch_size, max_nodes, _ = outs['attn_fts'].size()
    num_rels = len(batch_data['edge_sparse_matrices'][0])
    rel_edges = np.zeros((batch_size, num_rels, max_nodes, max_nodes), dtype=np.float32)
    for i, edge_sparse_matrices in enumerate(batch_data['edge_sparse_matrices']):
      for j, edge_sparse_matrix in enumerate(edge_sparse_matrices):
        rel_edges[i, j] = edge_sparse_matrix.todense()
    outs['rel_edges'] = torch.FloatTensor(rel_edges).to(self.device)
    if is_train:
      outs['caption_ids'] = torch.LongTensor(batch_data['caption_ids']).to(self.device)
      outs['caption_masks'] = torch.FloatTensor(batch_data['caption_masks'].astype(np.float32)).to(self.device)
      if 'gt_attns' in batch_data:
        outs['gt_attns'] = torch.FloatTensor(batch_data['gt_attns'].astype(np.float32)).to(self.device)
    return outs

  def forward_encoder(self, input_batch):
    attn_embeds = self.submods[ATTNENCODER](input_batch['attn_fts'], input_batch['rel_edges'])
    graph_embeds = torch.sum(attn_embeds * input_batch['attn_masks'].unsqueeze(2), 1) 
    graph_embeds = graph_embeds / torch.sum(input_batch['attn_masks'], 1, keepdim=True)
    enc_states = self.submods[MPENCODER](
      torch.cat([input_batch['mp_fts'], graph_embeds], 1))
    return {'init_states': enc_states, 'attn_fts': attn_embeds}

    loss = torch.sum(losses * masks) / torch.sum(masks)
    return loss

  def forward_loss(self, batch_data, step=None):
    input_batch = self.prepare_input_batch(batch_data, is_train=True)

    enc_outs = self.forward_encoder(input_batch)
    # logits.shape=(batch*seq_len, num_words)
    logits = self.submods[DECODER](input_batch['caption_ids'][:, :-1], 
      enc_outs['init_states'], enc_outs['attn_fts'], input_batch['attn_masks'])
    cap_loss = self.criterion(logits, input_batch['caption_ids'], 
      input_batch['caption_masks'])

    return cap_loss


class RoleGraphBUTDAttnModel(GraphBUTDAttnModel):
  def build_submods(self):
    submods = {}
    submods[MPENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[MPENCODER])
    submods[ATTNENCODER] = controlimcap.encoders.gcn.RoleRGCNEncoder(self.config.subcfgs[ATTNENCODER])
    submods[DECODER] = caption.decoders.attention.BUTDAttnDecoder(self.config.subcfgs[DECODER])
    return submods

  def prepare_input_batch(self, batch_data, is_train=False):
    outs = super().prepare_input_batch(batch_data, is_train=is_train)
    outs['node_types'] = torch.LongTensor(batch_data['node_types']).to(self.device)
    outs['attr_order_idxs'] = torch.LongTensor(batch_data['attr_order_idxs']).to(self.device)
    return outs

  def forward_encoder(self, input_batch):
    attn_embeds = self.submods[ATTNENCODER](input_batch['attn_fts'],
      input_batch['node_types'], input_batch['attr_order_idxs'], input_batch['rel_edges'])
    graph_embeds = torch.sum(attn_embeds * input_batch['attn_masks'].unsqueeze(2), 1) 
    graph_embeds = graph_embeds / torch.sum(input_batch['attn_masks'], 1, keepdim=True)
    enc_states = self.submods[MPENCODER](
      torch.cat([input_batch['mp_fts'], graph_embeds], 1))
    return {'init_states': enc_states, 'attn_fts': attn_embeds}

