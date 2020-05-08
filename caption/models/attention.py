import os
import json
import numpy as np
import collections

import torch
import torch.nn as nn

import framework.configbase
import caption.encoders.vanilla
import caption.decoders.attention
import caption.models.captionbase

MPENCODER = 'mp_encoder'
ATTNENCODER = 'attn_encoder'
DECODER = 'decoder'

class AttnModelConfig(framework.configbase.ModelConfig):
  def __init__(self):
    super().__init__()
    self.subcfgs[MPENCODER] = caption.encoders.vanilla.EncoderConfig()
    self.subcfgs[ATTNENCODER] = caption.encoders.vanilla.EncoderConfig()
    self.subcfgs[DECODER] = caption.decoders.attention.AttnDecoderConfig()

  def _assert(self):
    assert self.subcfgs[MPENCODER].dim_embed == self.subcfgs[DECODER].hidden_size
    assert self.subcfgs[ATTNENCODER].dim_embed == self.subcfgs[DECODER].attn_input_size


class AttnModel(caption.models.captionbase.CaptionModelBase):
  def build_submods(self):
    submods = {}
    submods[MPENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[MPENCODER])
    submods[ATTNENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[ATTNENCODER])
    submods[DECODER] = caption.decoders.attention.AttnDecoder(self.config.subcfgs[DECODER])
    return submods

  def prepare_input_batch(self, batch_data, is_train=False):
    outs = {}
    outs['mp_fts'] = torch.FloatTensor(batch_data['mp_fts']).to(self.device)
    outs['attn_fts'] = torch.FloatTensor(batch_data['attn_fts']).to(self.device)
    outs['attn_masks'] = torch.FloatTensor(batch_data['attn_masks'].astype(np.float32)).to(self.device)

    if is_train:
      outs['caption_ids'] = torch.LongTensor(batch_data['caption_ids']).to(self.device)
      outs['caption_masks'] = torch.FloatTensor(batch_data['caption_masks'].astype(np.float32)).to(self.device)
    return outs

  def forward_encoder(self, input_batch):
    encoder_state = self.submods[MPENCODER](input_batch['mp_fts'])
    encoder_outputs = self.submods[ATTNENCODER](input_batch['attn_fts'])
    return {'init_states': encoder_state, 'attn_fts': encoder_outputs}

  def forward_loss(self, batch_data, step=None):
    input_batch = self.prepare_input_batch(batch_data, is_train=True)

    enc_outs = self.forward_encoder(input_batch)
    # logits.shape=(batch*seq_len, num_words)
    logits = self.submods[DECODER](input_batch['caption_ids'][:, :-1], 
      enc_outs['init_states'], enc_outs['attn_fts'], input_batch['attn_masks'])  
    loss = self.criterion(logits, input_batch['caption_ids'], 
      input_batch['caption_masks'])

    return loss

  def validate_batch(self, batch_data, addition_outs=None):
    input_batch = self.prepare_input_batch(batch_data, is_train=False)
    enc_outs = self.forward_encoder(input_batch)
    init_words = torch.zeros(input_batch['attn_masks'].size(0), dtype=torch.int64).to(self.device)

    pred_sent, _ = self.submods[DECODER].sample_decode(init_words, 
      enc_outs['init_states'], enc_outs['attn_fts'], input_batch['attn_masks'], greedy=True)     
    return pred_sent

  def test_batch(self, batch_data, greedy_or_beam):
    input_batch = self.prepare_input_batch(batch_data, is_train=False)
    enc_outs = self.forward_encoder(input_batch)
    init_words = torch.zeros(input_batch['attn_masks'].size(0), dtype=torch.int64).to(self.device)

    if greedy_or_beam:
      sent_pool = self.submods[DECODER].beam_search_decode(
        init_words, enc_outs['init_states'], enc_outs['attn_fts'], 
        input_batch['attn_masks'])
      pred_sent = [pool[0][1] for pool in sent_pool]
    else:
      pred_sent, word_logprobs = self.submods[DECODER].sample_decode(
        init_words, enc_outs['init_states'], enc_outs['attn_fts'], 
        input_batch['attn_masks'], greedy=True)
      sent_pool = []
      for sent, word_logprob in zip(pred_sent, word_logprobs):
        sent_pool.append([(word_logprob.sum().item(), sent, word_logprob)])

    return pred_sent, sent_pool

class BUTDAttnModel(AttnModel):
  def build_submods(self):
    submods = {}
    submods[MPENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[MPENCODER])
    submods[ATTNENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[ATTNENCODER])
    submods[DECODER] = caption.decoders.attention.BUTDAttnDecoder(self.config.subcfgs[DECODER])
    return submods



