import numpy as np
import torch

import caption.encoders.vanilla
import controlimcap.encoders.gcn
import controlimcap.decoders.cfattention
import controlimcap.models.graphattn

MPENCODER = 'mp_encoder'
ATTNENCODER = 'attn_encoder'
DECODER = 'decoder'

class GraphBUTDCFlowAttnModel(controlimcap.models.graphattn.GraphBUTDAttnModel):
  def build_submods(self):
    submods = {}
    submods[MPENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[MPENCODER])
    submods[ATTNENCODER] = controlimcap.encoders.gcn.RGCNEncoder(self.config.subcfgs[ATTNENCODER])
    submods[DECODER] = controlimcap.decoders.cfattention.ContentFlowAttentionDecoder(
      self.config.subcfgs[DECODER])
    return submods

  def prepare_input_batch(self, batch_data, is_train=False):
    outs = super().prepare_input_batch(batch_data, is_train=is_train)
    flow_edges = [x.toarray() for x in batch_data['flow_sparse_matrix']]
    flow_edges = np.stack(flow_edges, 0)
    outs['flow_edges'] = torch.FloatTensor(flow_edges).to(self.device)
    return outs

  def forward_loss(self, batch_data, step=None):
    input_batch = self.prepare_input_batch(batch_data, is_train=True)

    enc_outs = self.forward_encoder(input_batch)
    # logits.shape=(batch*seq_len, num_words)
    logits = self.submods[DECODER](input_batch['caption_ids'][:, :-1], 
      enc_outs['init_states'], enc_outs['attn_fts'], input_batch['attn_masks'],
      input_batch['flow_edges'])  
    cap_loss = self.criterion(logits, input_batch['caption_ids'], 
      input_batch['caption_masks'])

    return cap_loss

  def validate_batch(self, batch_data, addition_outs=None):
    input_batch = self.prepare_input_batch(batch_data, is_train=False)
    enc_outs = self.forward_encoder(input_batch)

    batch_size = input_batch['attn_masks'].size(0)
    init_words = torch.zeros(batch_size, dtype=torch.int64).to(self.device)

    pred_sent, _ = self.submods[DECODER].sample_decode(init_words, 
      enc_outs['init_states'], enc_outs['attn_fts'], input_batch['attn_masks'], 
      input_batch['flow_edges'], greedy=True)
      
    return pred_sent

  def test_batch(self, batch_data, greedy_or_beam):
    input_batch = self.prepare_input_batch(batch_data, is_train=False)
    enc_outs = self.forward_encoder(input_batch)

    batch_size = input_batch['attn_masks'].size(0)
    init_words = torch.zeros(batch_size, dtype=torch.int64).to(self.device)

    if greedy_or_beam:
      sent_pool = self.submods[DECODER].beam_search_decode(
        init_words, enc_outs['init_states'], enc_outs['attn_fts'], 
        input_batch['attn_masks'], input_batch['flow_edges'])
      pred_sent = [pool[0][1] for pool in sent_pool]
    else:
      pred_sent, word_logprobs = self.submods[DECODER].sample_decode(
        init_words, enc_outs['init_states'], enc_outs['attn_fts'], 
        input_batch['attn_masks'], input_batch['flow_edges'], greedy=True)
      sent_pool = []
      for sent, word_logprob in zip(pred_sent, word_logprobs):
        sent_pool.append([(word_logprob.sum().item(), sent, word_logprob)])

    return pred_sent, sent_pool


class RoleGraphBUTDCFlowAttnModel(GraphBUTDCFlowAttnModel):
  def build_submods(self):
    submods = {}
    submods[MPENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[MPENCODER])
    submods[ATTNENCODER] = controlimcap.encoders.gcn.RoleRGCNEncoder(self.config.subcfgs[ATTNENCODER])
    submods[DECODER] = controlimcap.decoders.cfattention.ContentFlowAttentionDecoder(
      self.config.subcfgs[DECODER])
    return submods

  def prepare_input_batch(self, batch_data, is_train=False):
    outs = super().prepare_input_batch(batch_data, is_train=is_train)
    outs['node_types'] = torch.LongTensor(batch_data['node_types']).to(self.device)
    outs['attr_order_idxs'] = torch.LongTensor(batch_data['attr_order_idxs']).to(self.device)
    return outs

  def forward_encoder(self, input_batch):
    attn_embeds = self.submods[ATTNENCODER](input_batch['attn_fts'],
      input_batch['node_types'], input_batch['attr_order_idxs'], 
      input_batch['rel_edges'])
    graph_embeds = torch.sum(attn_embeds * input_batch['attn_masks'].unsqueeze(2), 1) 
    graph_embeds = graph_embeds / torch.sum(input_batch['attn_masks'], 1, keepdim=True)
    enc_states = self.submods[MPENCODER](
      torch.cat([input_batch['mp_fts'], graph_embeds], 1))
    return {'init_states': enc_states, 'attn_fts': attn_embeds}
