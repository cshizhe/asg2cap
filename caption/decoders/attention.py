import torch
import torch.nn as nn
import torch.nn.functional as F

import caption.utils.inference

import caption.decoders.vanilla
from framework.modules.embeddings import Embedding
from framework.modules.global_attention import GlobalAttention
from framework.modules.global_attention import AdaptiveAttention

class AttnDecoderConfig(caption.decoders.vanilla.DecoderConfig):
  def __init__(self):
    super().__init__()
    self.memory_same_key_value = True
    self.attn_input_size = 512
    self.attn_size = 512
    self.attn_type = 'mlp' # mlp, dot, general

  def _assert(self):
    assert self.attn_type in ['dot', 'general', 'mlp'], ('Please select a valid attention type.')

class AttnDecoder(caption.decoders.vanilla.Decoder):
  def __init__(self, config):
    super().__init__(config)

    self.attn = GlobalAttention(self.config.hidden_size, self.config.attn_size, self.config.attn_type)
    if self.config.attn_type == 'mlp':
      self.attn_linear_context = nn.Linear(self.config.attn_input_size, 
        self.config.attn_size, bias=False)

    if not self.config.memory_same_key_value:
      self.memory_value_layer = nn.Linear(self.config.attn_input_size, 
        self.config.attn_size, bias=True)

  @property
  def rnn_input_size(self):
    if self.config.memory_same_key_value:
      return self.config.dim_word + self.config.attn_input_size
    else:
      return self.config.dim_word + self.config.attn_size

  def gen_memory_key_value(self, enc_memories):
    if self.config.memory_same_key_value:
      memory_values = enc_memories
    else:
      memory_values = F.relu(self.memory_value_layer(enc_memories))

    if self.config.attn_type == 'mlp':
      memory_keys = self.attn_linear_context(enc_memories)
    else:
      memory_keys = enc_memories

    return memory_keys, memory_values

  def forward(self, inputs, enc_states, enc_memories, enc_masks, return_attn=False):
    '''
    Args:
      inputs: (batch, dec_seq_len)
      enc_states: (batch, dim_embed)
      enc_memoris: (batch, enc_seq_len, dim_embed)
      enc_masks: (batch, enc_seq_len)
    Returns:
      logits: (batch*seq_len, num_words)
    '''
    memory_keys, memory_values = self.gen_memory_key_value(enc_memories)
    states = self.init_dec_state(enc_states)
    outs = states[0][-1] if isinstance(states, tuple) else states[-1]

    step_outs, step_attns = [], []
    for t in range(inputs.size(1)):
      wordids = inputs[:, t]
      if t > 0 and self.config.schedule_sampling:
        sample_rate = torch.rand(wordids.size(0)).to(wordids.device)
        sample_mask = sample_rate < self.config.ss_rate
        prob = self.softmax(step_outs[-1]).detach()
        sampled_wordids = torch.multinomial(prob, 1).view(-1)
        wordids.masked_scatter_(sample_mask, sampled_wordids)
      embed = self.embedding(wordids)
      attn_score, attn_memory = self.attn(outs, 
        memory_keys, memory_values, enc_masks)
      step_attns.append(attn_score)
      rnn_input = torch.cat([embed, attn_memory], 1).unsqueeze(1)
      rnn_input = self.dropout(rnn_input)
      outs, states = self.rnn(rnn_input, states)
      outs = outs[:, 0]
      logit = self.calc_logits_with_rnn_outs(outs)
      step_outs.append(logit)
    
    logits = torch.stack(step_outs, 1)
    logits = logits.view(-1, self.config.num_words)

    if return_attn:
      return logits, step_attns
    return logits

  def step_fn(self, words, step, **kwargs):
    '''
    Args:
      words: (batch, 1)
      kwargs:
        - states: decoder init states (num_layers, batch, hidden_size)
        - outs: last decoder layer hidden as attn query (batch, hidden_size)
        - memory_keys: (batch, enc_seq_len, key_size)
        - memory_values: (batch, enc_seq_len, value_size)
        - memory_masks: (batch, enc_seq_len)
    '''
    states = kwargs['states']
    outs = kwargs['outs']
    memory_keys = kwargs['memory_keys']
    memory_values = kwargs['memory_values']
    memory_masks = kwargs['memory_masks']

    embeds = self.embedding(words)

    attn_score, attn_memory = self.attn(
      outs, memory_keys, memory_values, memory_masks)
    
    attn_memory = attn_memory.unsqueeze(1)
    rnn_inputs = torch.cat([embeds, attn_memory], 2)
    outs, states = self.rnn(rnn_inputs, states)
    outs = outs[:, 0]
    logits = self.calc_logits_with_rnn_outs(outs)
    logprobs = self.log_softmax(logits)

    kwargs['states'] = states
    kwargs['outs'] = outs
    return logprobs, kwargs

  def expand_fn(self, beam_width, **kwargs):
    '''
    Args:
      kwargs: 
        - states
        - outs: (batch, hidden_size)
        - memory_keys, memory_values, memory_masks: (batch, ...)
    '''
    kwargs = super().expand_fn(beam_width, **kwargs)
    for key, value in kwargs.items():
      if key != 'states':
        value_size = list(value.size())
        expand_size = [value_size[0], beam_width] + value_size[1:]
        final_size = [value_size[0] * beam_width] + value_size[1:]
        kwargs[key] = value.unsqueeze(1).expand(*expand_size).contiguous() \
                         .view(*final_size)
    return kwargs

    def select_fn(self, idxs, **kwargs):
      '''Select examples according to idxs
       kwargs:
        - states: lstm tuple (num_layer, batch_size*beam_width, hidden_size)
        - outs, memory_keys, memory_values, memory_masks: (batch, ...)
      '''
    kwargs = super().select_fn(idxs, **kwargs)
    for key, value in kwargs.items():
      if key != 'states':
        kwargs[key] = torch.index_select(value, 0, idxs)
    return kwargs

  def sample_decode(self, words, enc_states, enc_memories, enc_masks, greedy=True, early_stop=True):
    '''
    Args
      words: (batch, )
      enc_states: (batch, hidden_size)
      enc_memories: (batch, enc_seq_len, attn_input_size)
      enc_masks: (batch, enc_seq_len)
    '''
    states = self.init_dec_state(enc_states)
    outs = states[0][-1] if isinstance(states, tuple) else states[-1]
    memory_keys, memory_values = self.gen_memory_key_value(enc_memories)

    seq_words, seq_word_logprobs = caption.utils.inference.sample_decode(
      words, self.step_fn, self.config.max_words_in_sent, 
      greedy=greedy, early_stop=early_stop, states=states, outs=outs,
      memory_keys=memory_keys, memory_values=memory_values, memory_masks=enc_masks)

    return seq_words, seq_word_logprobs

  def beam_search_decode(self, words, enc_states, enc_memories, enc_masks):
    states = self.init_dec_state(enc_states)
    outs = states[0][-1] if isinstance(states, tuple) else states[-1]
    memory_keys, memory_values = self.gen_memory_key_value(enc_memories)

    sent_pool = caption.utils.inference.beam_search_decode(words, self.step_fn,
      self.config.max_words_in_sent, beam_width=self.config.beam_width, 
      sent_pool_size=self.config.sent_pool_size,
      expand_fn=self.expand_fn, select_fn=self.select_fn, 
      states=states, outs=outs, memory_keys=memory_keys,
      memory_values=memory_values, memory_masks=enc_masks)
    return sent_pool


class BUTDAttnDecoder(AttnDecoder):
  '''
  Requires: dim input visual feature == lstm hidden size
  '''
  def __init__(self, config):
    nn.Module.__init__(self) # need to rewrite RNN
    self.config = config
    # word embedding
    self.embedding = Embedding(self.config.num_words,
      self.config.dim_word, fix_word_embed=self.config.fix_word_embed)
    # rnn params (attn_lstm and lang_lstm)
    self.attn_lstm = nn.LSTMCell(
      self.config.hidden_size + self.config.attn_input_size + self.config.dim_word, # (h_lang, v_g, w)
      self.config.hidden_size, bias=True)
    memory_size = self.config.attn_input_size if self.config.memory_same_key_value else self.config.attn_size
    self.lang_lstm = nn.LSTMCell(
      self.config.hidden_size + memory_size, # (h_attn, v_a)
      self.config.hidden_size, bias=True)
    # attentions
    self.attn = GlobalAttention(self.config.hidden_size, self.config.attn_size, self.config.attn_type)
    if self.config.attn_type == 'mlp':
      self.attn_linear_context = nn.Linear(self.config.attn_input_size, self.config.attn_size, bias=False)
    if not self.config.memory_same_key_value:
      self.memory_value_layer = nn.Linear(self.config.attn_input_size, self.config.attn_size, bias=True)
    # outputs
    if self.config.hidden2word:
      self.hidden2word = nn.Linear(self.config.hidden_size, self.config.dim_word)
      output_size = self.config.dim_word
    else:
      output_size = self.config.hidden_size
    if not self.config.tie_embed:
      self.fc = nn.Linear(output_size, self.config.num_words)
    self.log_softmax = nn.LogSoftmax(dim=1)
    self.softmax = nn.Softmax(dim=1)

    self.dropout = nn.Dropout(self.config.dropout)
    self.init_rnn_weights(self.attn_lstm, 'lstm', num_layers=1)
    self.init_rnn_weights(self.lang_lstm, 'lstm', num_layers=1)

  def init_dec_state(self, batch_size):
    param = next(self.parameters())
    states = []
    for i in range(2): # (hidden, cell)
      states.append(torch.zeros((2, batch_size, self.config.hidden_size), 
        dtype=torch.float32).to(param.device))
    return states

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

      attn_score, attn_memory = self.attn(h_attn_lstm, 
        memory_keys, memory_values, enc_masks)
      step_attns.append(attn_score)

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

    embed = self.embedding(words.squeeze(1))

    h_attn_lstm, c_attn_lstm = self.attn_lstm(
      torch.cat([states[0][1], enc_globals, embed], dim=1),
      (states[0][0], states[1][0]))

    attn_score, attn_memory = self.attn(h_attn_lstm, 
      memory_keys, memory_values, memory_masks)

    h_lang_lstm, c_lang_lstm = self.lang_lstm(
      torch.cat([h_attn_lstm, attn_memory], dim=1),
      (states[0][1], states[1][1]))
      
    logits = self.calc_logits_with_rnn_outs(h_lang_lstm)
    logprobs = self.log_softmax(logits)
    states = (torch.stack([h_attn_lstm, h_lang_lstm], dim=0),
              torch.stack([c_attn_lstm, c_lang_lstm], dim=0))

    kwargs['states'] = states
    return logprobs, kwargs

  def sample_decode(self, words, enc_globals, enc_memories, enc_masks, greedy=True):
    states = self.init_dec_state(words.size(0))
    memory_keys, memory_values = self.gen_memory_key_value(enc_memories)

    seq_words, seq_word_logprobs = caption.utils.inference.sample_decode(
      words, self.step_fn, self.config.max_words_in_sent, 
      greedy=greedy, states=states, enc_globals=enc_globals, memory_keys=memory_keys, 
      memory_values=memory_values, memory_masks=enc_masks)

    return seq_words, seq_word_logprobs

  def beam_search_decode(self, words, enc_globals, enc_memories, enc_masks):
    states = self.init_dec_state(words.size(0))
    memory_keys, memory_values = self.gen_memory_key_value(enc_memories)

    sent_pool = caption.utils.inference.beam_search_decode(words, self.step_fn, 
      self.config.max_words_in_sent, beam_width=self.config.beam_width, 
      sent_pool_size=self.config.sent_pool_size, expand_fn=self.expand_fn,
      select_fn=self.select_fn, states=states, enc_globals=enc_globals,
      memory_keys=memory_keys, memory_values=memory_values, memory_masks=enc_masks)

    return sent_pool
