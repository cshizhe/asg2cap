import torch
import torch.nn as nn
import torch.nn.functional as F

import caption.utils.inference

import caption.decoders.vanilla
from framework.modules.embeddings import Embedding
from framework.modules.global_attention import GlobalAttention


class ContentFlowAttentionDecoder(caption.decoders.attention.BUTDAttnDecoder):
  def __init__(self, config):
    super().__init__(config)

    memory_size = self.config.attn_size if self.config.memory_same_key_value else self.config.attn_input_size
    self.address_layer = nn.Sequential(
      nn.Linear(self.config.hidden_size + memory_size, memory_size),
      nn.ReLU(),
      nn.Linear(memory_size, 1 + 3))

  def forward(self, inputs, enc_globals, enc_memories, enc_masks, flow_edges, return_attn=False):
    '''
    Args:
      inputs: (batch, dec_seq_len)
      enc_globals: (batch, hidden_size)
      enc_memories: (batch, enc_seq_len, attn_input_size)
      enc_masks: (batch, enc_seq_len)
    Returns:
      logits: (batch*seq_len, num_words)
    '''
    batch_size, max_attn_len = enc_masks.size()
    device = inputs.device

    states = self.init_dec_state(batch_size) # zero init state

    # initialize content attention
    memory_keys, memory_values = self.gen_memory_key_value(enc_memories)
    
    # initialize location attention score: (batch, max_attn_len)
    prev_attn_score = torch.zeros((batch_size, max_attn_len)).to(device) 
    prev_attn_score[:, 0] = 1

    step_outs, step_attns = [], []
    for t in range(inputs.size(1)):
      wordids = inputs[:, t]
      if t > 0 and self.config.schedule_sampling:
        sample_rate = torch.rand(wordids.size(0)).to(wordids.device)
        sample_mask = sample_rate < self.config.ss_rate
        prob = self.softmax(step_outs[-1]).detach()  # detach grad
        sampled_wordids = torch.multinomial(prob, 1).view(-1)
        wordids.masked_scatter_(sample_mask, sampled_wordids)
      embed = self.embedding(wordids)

      h_attn_lstm, c_attn_lstm = self.attn_lstm(
        torch.cat([states[0][1], enc_globals, embed], dim=1),
        (states[0][0], states[1][0]))

      prev_memory = torch.sum(prev_attn_score.unsqueeze(2) * memory_values, 1)
      address_params = self.address_layer(torch.cat([h_attn_lstm, prev_memory], 1))
      interpolate_gate = torch.sigmoid(address_params[:, :1])
      flow_gate = torch.softmax(address_params[:, 1:], dim=1)

      # content_attn_score: (batch, max_attn_len)
      content_attn_score, content_attn_memory = self.attn(h_attn_lstm, 
        memory_keys, memory_values, enc_masks)

      # location attention flow: (batch, max_attn_len)
      flow_attn_score_1 = torch.einsum('bts,bs->bt', flow_edges, prev_attn_score)
      flow_attn_score_2 = torch.einsum('bts,bs->bt',flow_edges, flow_attn_score_1)
      # (batch, max_attn_len, 3)
      flow_attn_score = torch.stack([x.view(batch_size, max_attn_len) \
        for x in [prev_attn_score, flow_attn_score_1, flow_attn_score_2]], 2)
      flow_attn_score = torch.sum(flow_gate.unsqueeze(1) * flow_attn_score, 2)

      # content + location interpolation
      attn_score = interpolate_gate * content_attn_score + (1 - interpolate_gate) * flow_attn_score

      # final attention
      step_attns.append(attn_score)
      prev_attn_score = attn_score
      attn_memory = torch.sum(attn_score.unsqueeze(2) * memory_values, 1)

      # next layer with attended context
      h_lang_lstm, c_lang_lstm = self.lang_lstm(
        torch.cat([h_attn_lstm, attn_memory], dim=1),
        (states[0][1], states[1][1]))

      outs = h_lang_lstm
      logit = self.calc_logits_with_rnn_outs(outs)
      step_outs.append(logit)
      states = (torch.stack([h_attn_lstm, h_lang_lstm], dim=0),
                torch.stack([c_attn_lstm, c_lang_lstm], dim=0))
    
    logits = torch.stack(step_outs, 1)
    logits = logits.view(-1, self.config.num_words)

    if return_attn:
      return logits, step_attns
    return logits

  def step_fn(self, words, step, **kwargs):
    states = kwargs['states']
    enc_globals = kwargs['enc_globals']
    memory_keys = kwargs['memory_keys']
    memory_values = kwargs['memory_values']
    memory_masks = kwargs['memory_masks']
    prev_attn_score = kwargs['prev_attn_score']
    flow_edges = kwargs['flow_edges']

    batch_size, max_attn_len = memory_masks.size()

    embed = self.embedding(words.squeeze(1))

    h_attn_lstm, c_attn_lstm = self.attn_lstm(
      torch.cat([states[0][1], enc_globals, embed], dim=1),
      (states[0][0], states[1][0]))

    prev_memory = torch.sum(prev_attn_score.unsqueeze(2) * memory_values, 1)
    address_params = self.address_layer(torch.cat([h_attn_lstm, prev_memory], 1))
    interpolate_gate = torch.sigmoid(address_params[:, :1])
    flow_gate = torch.softmax(address_params[:, 1:], dim=1)

    # content_attn_score: (batch, max_attn_len)
    content_attn_score, content_attn_memory = self.attn(h_attn_lstm, 
      memory_keys, memory_values, memory_masks)

    # location attention flow: (batch, max_attn_len)
    flow_attn_score_1 = torch.einsum('bts,bs->bt', flow_edges, prev_attn_score)
    flow_attn_score_2 = torch.einsum('bts,bs->bt', flow_edges, flow_attn_score_1)
    flow_attn_score = torch.stack([x.view(batch_size, max_attn_len) \
      for x in [prev_attn_score, flow_attn_score_1, flow_attn_score_2]], 2)
    flow_attn_score = torch.sum(flow_gate.unsqueeze(1) * flow_attn_score, 2)

    # content + location interpolation
    attn_score = interpolate_gate * content_attn_score + (1 - interpolate_gate) * flow_attn_score

    # final attention
    attn_memory = torch.sum(attn_score.unsqueeze(2) * memory_values, 1)

    h_lang_lstm, c_lang_lstm = self.lang_lstm(
      torch.cat([h_attn_lstm, attn_memory], dim=1),
      (states[0][1], states[1][1]))

    logits = self.calc_logits_with_rnn_outs(h_lang_lstm)
    logprobs = self.log_softmax(logits)
    states = (torch.stack([h_attn_lstm, h_lang_lstm], dim=0),
              torch.stack([c_attn_lstm, c_lang_lstm], dim=0))

    kwargs['prev_attn_score'] = attn_score
    kwargs['states'] = states
    return logprobs, kwargs

  def sample_decode(self, words, enc_globals, enc_memories, enc_masks, flow_edges, greedy=True):
    '''Args:
      words: (batch, )
      enc_globals: (batch, hidden_size)
      enc_memories: (batch, enc_seq_len, attn_input_size)
      enc_masks: (batch, enc_seq_len)
      flow_edges: sparse matrix, (batch*max_attn_len, batch*max_attn_len)
    '''
    batch_size, max_attn_len = enc_masks.size()
    device = enc_masks.device

    states = self.init_dec_state(batch_size)
    memory_keys, memory_values = self.gen_memory_key_value(enc_memories)
    prev_attn_score = torch.zeros((batch_size, max_attn_len)).to(device) 
    prev_attn_score[:, 0] = 1

    seq_words, seq_word_logprobs = caption.utils.inference.sample_decode(
      words, self.step_fn, self.config.max_words_in_sent, 
      greedy=greedy, states=states, enc_globals=enc_globals, 
      memory_keys=memory_keys, memory_values=memory_values, memory_masks=enc_masks,
      prev_attn_score=prev_attn_score, flow_edges=flow_edges)

    return seq_words, seq_word_logprobs

  def beam_search_decode(self, words, enc_globals, enc_memories, enc_masks, flow_edges):
    batch_size, max_attn_len = enc_masks.size()
    device = enc_masks.device
    
    states = self.init_dec_state(batch_size)
    memory_keys, memory_values = self.gen_memory_key_value(enc_memories)
    prev_attn_score = torch.zeros((batch_size, max_attn_len)).to(device) 
    prev_attn_score[:, 0] = 1

    sent_pool = caption.utils.inference.beam_search_decode(words, self.step_fn, 
      self.config.max_words_in_sent, beam_width=self.config.beam_width, 
      sent_pool_size=self.config.sent_pool_size, 
      expand_fn=self.expand_fn, select_fn=self.select_fn,
      memory_keys=memory_keys, memory_values=memory_values, memory_masks=enc_masks,
      states=states, enc_globals=enc_globals,
      prev_attn_score=prev_attn_score, flow_edges=flow_edges)

    return sent_pool
