import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.configbase
from framework.modules.embeddings import Embedding
import framework.ops
import caption.utils.inference

class DecoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.rnn_type = 'lstm'
    self.num_words = 0
    self.dim_word = 512
    self.hidden_size = 512
    self.num_layers = 1
    self.hidden2word = False
    self.tie_embed = False
    self.fix_word_embed = False
    self.max_words_in_sent = 20
    self.dropout = 0.5
    self.schedule_sampling = False
    self.ss_rate = 0.05
    self.ss_max_rate = 0.25
    self.ss_increase_rate = 0.05
    self.ss_increase_epoch = 5

    self.greedy_or_beam = False  # test method
    self.beam_width = 1
    self.sent_pool_size = 1

  def _assert(self):
    if self.tie_embed and not self.hidden2word:
      assert self.dim_word == self.hidden_size

class Decoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.embedding = Embedding(self.config.num_words, 
      self.config.dim_word, fix_word_embed=self.config.fix_word_embed)

    kwargs = {}
    self.rnn = framework.ops.rnn_factory(self.config.rnn_type,
      input_size=self.rnn_input_size, hidden_size=self.config.hidden_size, 
      num_layers=self.config.num_layers, dropout=self.config.dropout,
      bias=True, batch_first=True, **kwargs)

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
    
    self.init_rnn_weights(self.rnn, self.config.rnn_type)

  @property
  def rnn_input_size(self):
    return self.config.dim_word

  def init_rnn_weights(self, rnn, rnn_type, num_layers=None):
    if rnn_type == 'lstm':
      # the ordering of weights a biases is ingate, forgetgate, cellgate, outgate
      # init forgetgate as 1 to make rnn remember the past in the beginning
      if num_layers is None:
        num_layers = rnn.num_layers
      for layer in range(num_layers):
        for name in ['i', 'h']:
          try:
            weight = getattr(rnn, 'weight_%sh_l%d'%(name, layer))
          except:
            weight = getattr(rnn, 'weight_%sh'%name)
          nn.init.orthogonal_(weight.data)
          try:
            bias = getattr(rnn, 'bias_%sh_l%d'%(name, layer))
          except:
            bias = getattr(rnn, 'bias_%sh'%name) # BUTD: LSTM Cell
          nn.init.constant_(bias, 0)
          if name == 'i':
            bias.data.index_fill_(0, torch.arange(
              rnn.hidden_size, rnn.hidden_size*2).long(), 1)
            # bias.requires_grad = False

  def init_dec_state(self, encoder_state):
    '''
      The encoder hidden is (batch, dim_embed)
      We need to convert it to (layers, batch, hidden_size)
      assert dim_embed == hidden_size
    '''
    decoder_state = encoder_state.repeat(self.config.num_layers, 1, 1)
    if self.config.rnn_type == 'lstm' or self.config.rnn_type == 'ONLSTM':
      decoder_state = tuple([decoder_state, decoder_state])
    return decoder_state

  def calc_logits_with_rnn_outs(self, outs):
    '''
    Args: 
      outs: (batch, hidden_size)
    Returns:
      logits: (batch, num_words)
    '''
    if self.config.hidden2word:
      outs = torch.tanh(self.hidden2word(outs))
    outs = self.dropout(outs)
    if self.config.tie_embed:
      logits = torch.mm(outs, self.embedding.we.weight.t())
    else:
      logits = self.fc(outs)
    return logits
    
  def forward(self, inputs, encoder_state):
    '''
    Args:
      inputs: (batch, seq_len)
      encoder_state: (batch, dim_embed)
    Returns:
      logits: (batch*seq_len, num_words)
    '''   
    states = self.init_dec_state(encoder_state)

    if self.config.schedule_sampling:
      step_outs = []
      for t in range(inputs.size(1)):
        wordids = inputs[:, t]
        if t > 0:
          sample_rate = torch.rand(wordids.size(0)).to(wordids.device)
          sample_mask = sample_rate < self.config.ss_rate
          prob = self.softmax(step_outs[-1]).detach()
          sampled_wordids = torch.multinomial(prob, 1).squeeze(1)
          wordids.masked_scatter_(sample_mask, sampled_wordids)
        embed = self.embedding(wordids)
        embed = self.dropout(embed)
        outs, states = self.rnn(embed.unsqueeze(1), states)
        outs = outs[:, 0]
        logit = self.calc_logits_with_rnn_outs(outs)
        step_outs.append(logit)
      logits = torch.stack(step_outs, 1)
      logits = logits.view(-1, self.config.num_words)
    # pytorch rnn utilzes cudnn to speed up
    else:
      embeds = self.embedding(inputs)
      embeds = self.dropout(embeds)
      # outs.size(batch, seq_len, hidden_size)
      outs, states = self.rnn(embeds, states) 
      outs = outs.contiguous().view(-1, self.config.hidden_size)
      logits = self.calc_logits_with_rnn_outs(outs)
    return logits

  def step_fn(self, words, step, **kwargs):
    '''
    Args:
      words: (batch_size, 1)
      step: int (start from 0)
      kwargs:
        states: decoder rnn states (num_layers, batch, hidden_size)
    Returns:
      logprobs: (batch, num_words)
      kwargs: dict, {'states'}
    '''
    embeds = self.embedding(words)
    outs, states = self.rnn(embeds, kwargs['states'])
    outs = outs[:, 0]
    logits = self.calc_logits_with_rnn_outs(outs)
    logprobs = self.log_softmax(logits)
    kwargs['states'] = states
    return logprobs, kwargs

  def expand_fn(self, beam_width, **kwargs):
    '''After the first step of beam search, expand the examples to beam_width times
       e.g. (1, 2, 3) -> (1, 1, 2, 2, 3, 3)
    beam_width: int
    kwargs:
      - states: lstm tuple (num_layer, batch_size, hidden_size)
    '''
    states = kwargs['states']
    is_tuple = isinstance(states, tuple)
    if not is_tuple:
      states = (states, )
    
    expanded_states = []
    for h in states:
      num_layer, batch_size, hidden_size = h.size()
      eh = h.unsqueeze(2).expand(-1, -1, beam_width, -1).contiguous() \
            .view(num_layer, batch_size * beam_width, hidden_size)
      expanded_states.append(eh)
    
    if is_tuple:
      states = tuple(expanded_states)
    else:
      states = expanded_states[0]
    
    kwargs['states'] = states
    return kwargs

  def select_fn(self, idxs, **kwargs):
    '''Select examples according to idxs
    kwargs:
      states: lstm tuple (num_layer, batch_size*beam_width, hidden_size)
    '''
    states = kwargs['states']
    if isinstance(states, tuple):
      states = tuple([torch.index_select(h, 1, idxs) for h in states])
    else:
      states = torch.index_select(h, 1, idxs)
    kwargs['states'] = states
    return kwargs

  def sample_decode(self, words, enc_states, greedy=True, early_stop=True):
    '''
    Args
      words: (batch, )
      enc_states: (batch, hidden_size)
    '''
    states = self.init_dec_state(enc_states)
    seq_words, seq_word_logprobs = caption.utils.inference.sample_decode(
      words, self.step_fn, self.config.max_words_in_sent, 
      greedy=greedy, early_stop=early_stop, states=states)

    return seq_words, seq_word_logprobs

  def beam_search_decode(self, words, enc_states):
    '''
    Args:
      words: (batch, )
      enc_states: (batch, hidden_size)
    Returns:
      sent_pool: list, len=batch
        item=list, len=beam_width, 
          element=(sent_logprob, words, word_logprobs)
    '''
    states = self.init_dec_state(enc_states)
    sent_pool = caption.utils.inference.beam_search_decode(words, self.step_fn,
      self.config.max_words_in_sent, beam_width=self.config.beam_width, 
      sent_pool_size=self.config.sent_pool_size,
      expand_fn=self.expand_fn, select_fn=self.select_fn, 
      states=states)
    return sent_pool

