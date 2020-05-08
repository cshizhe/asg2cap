import os
import json
import numpy as np
import codecs

import torch.utils.data

from caption.utils.inference import BOS, EOS, UNK

class CaptionDatasetBase(torch.utils.data.Dataset):
  def __init__(self, word2int_file, ref_caption_file=None, 
    max_words_in_sent=20, is_train=False, return_label=False, _logger=None):
    if _logger is None:
      self.print_fn = print
    else:
      self.print_fn = _logger.info

    if word2int_file.endswith('json'):
      self.word2int = json.load(open(word2int_file))
    else:
      self.word2int = np.load(word2int_file)
    self.int2word = {i: w for w, i in self.word2int.items()}

    if ref_caption_file is not None:
      self.ref_captions = json.load(open(ref_caption_file))

    self.max_words_in_sent = max_words_in_sent
    self.is_train = is_train
    self.return_label = return_label

  def sent2int(self, str_sent):
    int_sent = [self.word2int.get(w, UNK) for w in str_sent.split()]
    return int_sent

  def pad_sents(self, int_sent, add_bos_eos=True):
    if add_bos_eos:
      sent = [BOS] + int_sent + [EOS]
    else:
      sent = int_sent
    sent = sent[:self.max_words_in_sent]
    num_pad = self.max_words_in_sent - len(sent)
    mask = [True]*len(sent) + [False] * num_pad
    sent = sent + [EOS] * num_pad
    return sent, mask

  def pad_or_trim_feature(self, attn_ft, max_len, average=False):
    seq_len, dim_ft = attn_ft.shape
    mask = np.zeros((max_len, ), np.bool)
    
    # pad
    if seq_len < max_len:
      new_ft = np.zeros((max_len, dim_ft), np.float32)
      new_ft[:seq_len] = attn_ft
      mask[:seq_len] = True
    elif seq_len == max_len:
      new_ft = attn_ft
      mask[:] = True
    # trim
    else:
      if average:
        idxs = np.round(np.linspace(0, seq_len, max_len+1)).astype(np.int32)
        new_ft = np.array([np.mean(attn_ft[idxs[i]: idxs[i+1]], axis=0) for i in range(max_len)])
      else:
        idxs = np.round(np.linspace(0, seq_len-1, max_len)).astype(np.int32)
        new_ft = attn_ft[idxs]
      mask[:] = True
    return new_ft, mask

      
