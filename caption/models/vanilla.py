import os
import json
import numpy as np
import collections
import time

import torch
import torch.nn as nn

import framework.configbase
import framework.ops

import caption.encoders.vanilla
import caption.decoders.vanilla
import caption.models.captionbase

ENCODER = 'encoder'
DECODER = 'decoder'

class ModelConfig(framework.configbase.ModelConfig):
  def __init__(self):
    super().__init__()
    self.subcfgs[ENCODER] = caption.encoders.vanilla.EncoderConfig()
    self.subcfgs[DECODER] = caption.decoders.vanilla.DecoderConfig()

  def _assert(self):
    assert self.subcfgs[ENCODER].dim_embed == self.subcfgs[DECODER].hidden_size


class VanillaModel(caption.models.captionbase.CaptionModelBase):
  def build_submods(self):
    submods = {}
    submods[ENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[ENCODER])
    submods[DECODER] = caption.decoders.vanilla.Decoder(self.config.subcfgs[DECODER])
    return submods

  def prepare_input_batch(self, batch_data, is_train=False):
    outs = {
      'mp_fts': torch.FloatTensor(batch_data['mp_fts']).to(self.device),
    }
    if is_train:
      outs['caption_ids'] = torch.LongTensor(batch_data['caption_ids']).to(self.device)
      outs['caption_masks'] = torch.FloatTensor(batch_data['caption_masks'].astype(np.float32)).to(self.device)
    return outs

  def forward_encoder(self, input_batch):
    ft_embeds = self.submods[ENCODER](input_batch['mp_fts'])
    return {'init_states': ft_embeds}

  def forward_loss(self, batch_data, step=None):
    input_batch = self.prepare_input_batch(batch_data, is_train=True)
    enc_outs = self.forward_encoder(input_batch)
    # logits.shape=(batch*(seq_len-1), num_words)
    logits = self.submods[DECODER](input_batch['caption_ids'][:, :-1], enc_outs['init_states'])  
    loss = self.criterion(logits, input_batch['caption_ids'], input_batch['caption_masks'])
    return loss

  def validate_batch(self, batch_data):
    input_batch = self.prepare_input_batch(batch_data, is_train=False)
    enc_outs = self.forward_encoder(input_batch)

    batch_size = len(batch_data['mp_fts'])
    init_words = torch.zeros(batch_size, dtype=torch.int64).to(self.device)
    pred_sent, _ = self.submods[DECODER].sample_decode(
      init_words, enc_outs['init_states'], greedy=True)
    return pred_sent

  def test_batch(self, batch_data, greedy_or_beam):
    input_batch = self.prepare_input_batch(batch_data, is_train=False)
    enc_outs = self.forward_encoder(input_batch)

    batch_size = len(batch_data['mp_fts'])
    init_words = torch.zeros(batch_size, dtype=torch.int64).to(self.device)
    if greedy_or_beam:
      sent_pool = self.submods[DECODER].beam_search_decode(
        init_words, enc_outs['init_states'])
      pred_sent = [pool[0][1] for pool in sent_pool]
    else:
      pred_sent, word_logprobs = self.submods[DECODER].sample_decode(
        init_words, enc_outs['init_states'], greedy=True)
      sent_pool = []
      for sent, word_logprob in zip(pred_sent, word_logprobs):
        sent_pool.append([(word_logprob.sum().item(), sent, word_logprob)])
    return pred_sent, sent_pool





