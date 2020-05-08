import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

def l2norm(inputs, dim=-1):
  # inputs: (batch, dim_ft)
  norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
  inputs = inputs / norm.clamp(min=1e-10)
  return inputs

def sequence_mask(lengths, max_len=None, inverse=False):
  ''' Creates a boolean mask from sequence lengths.
  '''
  # lengths: LongTensor, (batch, )
  batch_size = lengths.size(0)
  max_len = max_len or lengths.max()
  mask = torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1)
  if inverse:
    mask = mask.ge(lengths.unsqueeze(1))
  else:
    mask = mask.lt(lengths.unsqueeze(1))
  return mask

def subsequent_mask(size):
  '''Mask out subsequent position.
  Args
    size: the length of tgt words'''
  attn_shape = (1, size, size)
  # set the values below the 1th diagnose as 0
  mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
  mask = torch.from_numpy(mask) == 0
  return mask

def rnn_factory(rnn_type, **kwargs):
  rnn = getattr(nn, rnn_type.upper())(**kwargs)
  return rnn

