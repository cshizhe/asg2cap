import torch
import torch.nn as nn

import framework.configbase
import caption.encoders.vanilla
import caption.decoders.attention
import caption.models.attention

import controlimcap.encoders.flat

from caption.models.attention import MPENCODER, ATTNENCODER, DECODER


class NodeBUTDAttnModel(caption.models.attention.BUTDAttnModel):
  def forward_encoder(self, input_batch):
    attn_embeds = self.submods[ATTNENCODER](input_batch['attn_fts'])
    graph_embeds = torch.sum(attn_embeds * input_batch['attn_masks'].unsqueeze(2), 1)
    graph_embeds = graph_embeds / torch.sum(input_batch['attn_masks'], 1, keepdim=True)
    enc_states = self.submods[MPENCODER](
      torch.cat([input_batch['mp_fts'], graph_embeds], 1))
    return {'init_states': enc_states, 'attn_fts': attn_embeds}


class NodeRoleBUTDAttnModelConfig(caption.models.attention.AttnModelConfig):
  def __init__(self):
    super().__init__()
    self.subcfgs[ATTNENCODER] = controlimcap.encoders.flat.EncoderConfig()

class NodeRoleBUTDAttnModel(caption.models.attention.BUTDAttnModel):
  def build_submods(self):
    submods = {}
    submods[MPENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[MPENCODER])
    submods[ATTNENCODER] = controlimcap.encoders.flat.Encoder(self.config.subcfgs[ATTNENCODER])
    submods[DECODER] = caption.decoders.attention.BUTDAttnDecoder(self.config.subcfgs[DECODER])
    return submods

  def prepare_input_batch(self, batch_data, is_train=False):
    outs = super().prepare_input_batch(batch_data, is_train=is_train)
    outs['node_types'] = torch.LongTensor(batch_data['node_types']).to(self.device)
    outs['attr_order_idxs'] = torch.LongTensor(batch_data['attr_order_idxs']).to(self.device)
    return outs

  def forward_encoder(self, input_batch):
    attn_embeds = self.submods[ATTNENCODER](input_batch['attn_fts'],
      input_batch['node_types'], input_batch['attr_order_idxs'])
    graph_embeds = torch.sum(attn_embeds * input_batch['attn_masks'].unsqueeze(2), 1)
    graph_embeds = graph_embeds / torch.sum(input_batch['attn_masks'], 1, keepdim=True)
    enc_states = self.submods[MPENCODER](
      torch.cat([input_batch['mp_fts'], graph_embeds], 1))
    return {'init_states': enc_states, 'attn_fts': attn_embeds}

