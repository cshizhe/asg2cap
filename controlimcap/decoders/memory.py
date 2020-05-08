import torch
import torch.nn as nn
import torch.nn.functional as F

import caption.decoders.attention
from framework.modules.embeddings import Embedding
from framework.modules.global_attention import GlobalAttention

from controlimcap.decoders.cfattention import ContentFlowAttentionDecoder


class MemoryDecoder(caption.decoders.attention.BUTDAttnDecoder):
  def __init__(self, config):
    super().__init__(config)

    memory_size = self.config.attn_size if self.config.memory_same_key_value else self.config.attn_input_size
    self.memory_update_layer = nn.Sequential(
      nn.Linear(self.config.hidden_size + memory_size, memory_size),
      nn.ReLU(),
      nn.Linear(memory_size, memory_size * 2))
    self.sentinal_layer = nn.Sequential(
      nn.Linear(self.config.hidden_size, self.config.hidden_size),
      nn.ReLU(),
      nn.Linear(self.config.hidden_size, 1))

  def forward(self, inputs, enc_globals, enc_memories, enc_masks, return_attn=False):
    '''
    Args:
      inputs: (batch, dec_seq_len)
      enc_globals: (batch, hidden_size)
      enc_memories: (batch, enc_seq_len, attn_input_size)
      enc_masks: (batch, enc_seq_len)
    Returns:
      logits: (batch*seq_len, num_words)
    '''
    enc_seq_len = enc_memories.size(1)
    memory_keys, memory_values = self.gen_memory_key_value(enc_memories)
    states = self.init_dec_state(inputs.size(0)) # zero init state

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

      # attn_score: (batch, max_attn_len)
      attn_score, attn_memory = self.attn(h_attn_lstm, 
        memory_keys, memory_values, enc_masks)
      step_attns.append(attn_score)

      h_lang_lstm, c_lang_lstm = self.lang_lstm(
        torch.cat([h_attn_lstm, attn_memory], dim=1),
        (states[0][1], states[1][1]))

      # write: update memory keys and values
      # (batch, enc_seq_len, hidden_size + attn_input_size)
      individual_vectors = torch.cat(
        [h_lang_lstm.unsqueeze(1).expand(-1, enc_seq_len, -1), enc_memories], 2)
      update_vectors = self.memory_update_layer(individual_vectors)
      memory_size = update_vectors.size(-1) // 2
      erase_gates = torch.sigmoid(update_vectors[:, :, :memory_size])
      add_vectors = update_vectors[:, :, memory_size:]

      # some words do not need to attend on visual nodes
      sentinal_gates = torch.sigmoid(self.sentinal_layer(h_lang_lstm))
      memory_attn_score = attn_score * sentinal_gates

      enc_memories = enc_memories * (1 - memory_attn_score.unsqueeze(2) * erase_gates) \
                    + memory_attn_score.unsqueeze(2) * add_vectors
      memory_keys, memory_values = self.gen_memory_key_value(enc_memories)

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
    enc_memories = kwargs['enc_memories']
    memory_masks = kwargs['memory_masks']
    enc_seq_len = enc_memories.size(1)

    embed = self.embedding(words.squeeze(1))

    h_attn_lstm, c_attn_lstm = self.attn_lstm(
      torch.cat([states[0][1], enc_globals, embed], dim=1),
      (states[0][0], states[1][0]))

    memory_keys, memory_values = self.gen_memory_key_value(enc_memories)
    attn_score, attn_memory = self.attn(h_attn_lstm, 
      memory_keys, memory_values, memory_masks)

    h_lang_lstm, c_lang_lstm = self.lang_lstm(
      torch.cat([h_attn_lstm, attn_memory], dim=1),
      (states[0][1], states[1][1]))
    
    logits = self.calc_logits_with_rnn_outs(h_lang_lstm)
    logprobs = self.log_softmax(logits)
    states = (torch.stack([h_attn_lstm, h_lang_lstm], dim=0),
              torch.stack([c_attn_lstm, c_lang_lstm], dim=0))

    # write: update memory keys and values
    individual_vectors = torch.cat([h_lang_lstm.unsqueeze(1).expand(-1, enc_seq_len, -1), enc_memories], 2)
    update_vectors = self.memory_update_layer(individual_vectors)
    memory_size = update_vectors.size(-1) // 2
    erase_gates = torch.sigmoid(update_vectors[:, :, :memory_size])
    add_vectors = update_vectors[:, :, memory_size:]

    sentinal_gates = torch.sigmoid(self.sentinal_layer(h_lang_lstm))
    memory_attn_score = attn_score * sentinal_gates
    enc_memories = enc_memories * (1 - memory_attn_score.unsqueeze(2) * erase_gates) \
                  + memory_attn_score.unsqueeze(2) * add_vectors
    kwargs['enc_memories'] = enc_memories
    kwargs['states'] = states
    return logprobs, kwargs

  def sample_decode(self, words, enc_globals, enc_memories, enc_masks, greedy=True, early_stop=True):
    '''
    Args
      words: (batch, )
      enc_states: (batch, hidden_size)
      enc_memories: (batch, enc_seq_len, attn_input_size)
      enc_masks: (batch, enc_seq_len)
    '''
    states = self.init_dec_state(words.size(0))

    seq_words, seq_word_logprobs = caption.utils.inference.sample_decode(
      words, self.step_fn, self.config.max_words_in_sent, 
      greedy=greedy, early_stop=early_stop, states=states, 
      enc_globals=enc_globals, enc_memories=enc_memories, memory_masks=enc_masks)

    return seq_words, seq_word_logprobs

  def beam_search_decode(self, words, enc_globals, enc_memories, enc_masks):
    states = self.init_dec_state(words.size(0))

    sent_pool = caption.utils.inference.beam_search_decode(words, self.step_fn,
      self.config.max_words_in_sent, beam_width=self.config.beam_width, 
      sent_pool_size=self.config.sent_pool_size,
      expand_fn=self.expand_fn, select_fn=self.select_fn, 
      states=states, enc_globals=enc_globals, 
      enc_memories=enc_memories, memory_masks=enc_masks)
    return sent_pool


class MemoryFlowDecoder(ContentFlowAttentionDecoder):
  def __init__(self, config):
    super().__init__(config)

    memory_size = self.config.attn_size if self.config.memory_same_key_value else self.config.attn_input_size
    self.memory_update_layer = nn.Sequential(
      nn.Linear(self.config.hidden_size + memory_size, memory_size),
      nn.ReLU(),
      nn.Linear(memory_size, memory_size * 2))
    self.sentinal_layer = nn.Sequential(
      nn.Linear(self.config.hidden_size, self.config.hidden_size),
      nn.ReLU(),
      nn.Linear(self.config.hidden_size, 1))

  def forward(self, inputs, enc_globals, enc_memories, enc_masks, flow_edges, return_attn=False):
    '''
    Args:
      inputs: (batch, dec_seq_len)
      enc_globals: (batch, hidden_size)
      enc_memories: (batch, enc_seq_len, attn_input_size)
      enc_masks: (batch, enc_seq_len)
      flow_edges: sparse matrix (num_nodes, num_nodes), num_nodes=batch*enc_seq_len
    Returns:
      logits: (batch*seq_len, num_words)
    '''
    batch_size, max_attn_len = enc_masks.size()
    device = inputs.device

    # initialize states
    states = self.init_dec_state(batch_size) # zero init state

    # location attention: (batch, max_attn_len)
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

      memory_keys, memory_values = self.gen_memory_key_value(enc_memories)

      prev_attn_memory = torch.sum(prev_attn_score.unsqueeze(2) * memory_values, 1)
      address_params = self.address_layer(torch.cat([h_attn_lstm, prev_attn_memory], 1))
      interpolate_gate = torch.sigmoid(address_params[:, :1])
      flow_gate = torch.softmax(address_params[:, 1:], dim=1)

      # content_attn_score: (batch, max_attn_len)
      content_attn_score, content_attn_memory = self.attn(h_attn_lstm, 
        memory_keys, memory_values, enc_masks)

      # location attention flow: (batch, max_attn_len)
      flow_attn_score_1 = torch.einsum('bts,bs->bt', flow_edges, prev_attn_score)
      flow_attn_score_2 = torch.einsum('bts,bs->bt', flow_edges, flow_attn_score_1)
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

      # write: update memory keys and values
      individual_vectors = torch.cat([h_lang_lstm.unsqueeze(1).expand(-1, max_attn_len, -1), enc_memories], 2)
      update_vectors = self.memory_update_layer(individual_vectors)
      memory_size = update_vectors.size(-1) // 2
      erase_gates = torch.sigmoid(update_vectors[:, :, :memory_size])
      add_vectors = update_vectors[:, :, memory_size:]

      # some words do not need to attend on visual nodes
      sentinal_gates = torch.sigmoid(self.sentinal_layer(h_lang_lstm))
      memory_attn_score = attn_score * sentinal_gates

      enc_memories = enc_memories * (1 - memory_attn_score.unsqueeze(2) * erase_gates) \
                    + memory_attn_score.unsqueeze(2) * add_vectors

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
    enc_memories = kwargs['enc_memories']
    memory_masks = kwargs['memory_masks']
    prev_attn_score = kwargs['prev_attn_score']
    flow_edges = kwargs['flow_edges']

    batch_size, max_attn_len = memory_masks.size()
    memory_keys, memory_values = self.gen_memory_key_value(enc_memories)
    embed = self.embedding(words.squeeze(1))

    h_attn_lstm, c_attn_lstm = self.attn_lstm(
      torch.cat([states[0][1], enc_globals, embed], dim=1),
      (states[0][0], states[1][0]))

    prev_attn_memory = torch.sum(prev_attn_score.unsqueeze(2) * memory_values, 1)
    address_params = self.address_layer(torch.cat([h_attn_lstm, prev_attn_memory], 1))
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

    # write: update memory keys and values
    individual_vectors = torch.cat([h_lang_lstm.unsqueeze(1).expand(-1, max_attn_len, -1), enc_memories], 2)
    update_vectors = self.memory_update_layer(individual_vectors)
    memory_size = update_vectors.size(-1) // 2
    erase_gates = torch.sigmoid(update_vectors[:, :, :memory_size])
    add_vectors = update_vectors[:, :, memory_size:]

    sentinal_gates = torch.sigmoid(self.sentinal_layer(h_lang_lstm))
    memory_attn_score = attn_score * sentinal_gates
    enc_memories = enc_memories * (1 - memory_attn_score.unsqueeze(2) * erase_gates) \
                  + memory_attn_score.unsqueeze(2) * add_vectors

    kwargs['states'] = states
    kwargs['enc_memories'] = enc_memories
    kwargs['prev_attn_score'] = attn_score
    return logprobs, kwargs

  def sample_decode(self, words, enc_globals, enc_memories, enc_masks, flow_edges, greedy=True):
    batch_size, max_attn_len = enc_masks.size()
    device = enc_masks.device

    states = self.init_dec_state(batch_size)
    prev_attn_score = torch.zeros((batch_size, max_attn_len)).to(device) 
    prev_attn_score[:, 0] = 1

    seq_words, seq_word_logprobs = caption.utils.inference.sample_decode(
      words, self.step_fn, self.config.max_words_in_sent, 
      greedy=greedy, states=states, enc_globals=enc_globals, 
      enc_memories=enc_memories, memory_masks=enc_masks,
      prev_attn_score=prev_attn_score, flow_edges=flow_edges)

    return seq_words, seq_word_logprobs

  def beam_search_decode(self, words, enc_globals, enc_memories, enc_masks, flow_edges):
    batch_size, max_attn_len = enc_masks.size()
    device = enc_masks.device
    
    states = self.init_dec_state(batch_size)
    prev_attn_score = torch.zeros((batch_size, max_attn_len)).to(device) 
    prev_attn_score[:, 0] = 1

    sent_pool = caption.utils.inference.beam_search_decode(words, self.step_fn, 
      self.config.max_words_in_sent, beam_width=self.config.beam_width, 
      sent_pool_size=self.config.sent_pool_size, 
      expand_fn=self.expand_fn, select_fn=self.select_fn,
      enc_memories=enc_memories, memory_masks=enc_masks,
      states=states, enc_globals=enc_globals,
      prev_attn_score=prev_attn_score, flow_edges=flow_edges)

    return sent_pool
