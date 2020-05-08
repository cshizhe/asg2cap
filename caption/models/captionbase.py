import os
import json
import collections
import numpy as np
import torch
import torch.nn as nn

from eval_cap.bleu.bleu import Bleu
from eval_cap.cider.cider import Cider
from eval_cap.meteor.meteor import Meteor
from eval_cap.rouge.rouge import Rouge

import caption.utils.inference
import framework.modelbase

DECODER = 'decoder'

class CaptionLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss = nn.CrossEntropyLoss(reduction='none')

  def forward(self, logits, caption_ids, caption_masks, reduce_mean=True):
    '''
      logits: shape=(batch*(seq_len-1), num_words)
      caption_ids: shape=(batch, seq_len)
      caption_masks: shape=(batch, seq_len)
    '''
    batch_size, seq_len = caption_ids.size()
    losses = self.loss(logits, caption_ids[:, 1:].contiguous().view(-1))
    onehot_caption_masks = caption_masks[:, 1:] > 0
    onehot_caption_masks = onehot_caption_masks.float()
    caption_masks = caption_masks[:, 1:].reshape(-1).float()
    if reduce_mean:
      loss = torch.sum(losses * caption_masks) / torch.sum(onehot_caption_masks)
    else:
      loss = torch.div(
        torch.sum((losses * caption_masks).view(batch_size, seq_len-1), 1),
        torch.sum(onehot_caption_masks, 1))
    return loss

class CaptionModelBase(framework.modelbase.ModelBase):
  def __init__(self, config, _logger=None, eval_loss=False, int2word_file=None, gpu_id=0):
    self.eval_loss = eval_loss
    self.scorers = {
      'bleu4': Bleu(4),
      'cider': Cider(),
    }
    if int2word_file is not None:
      self.int2sent = caption.utils.inference.IntToSentence(int2word_file)
    super().__init__(config, _logger=_logger, gpu_id=gpu_id)

  def build_loss(self):
    criterion = CaptionLoss()
    return criterion

  def validate(self, val_reader, step=None):
    self.eval_start()

    # current eval_loss only select one caption for each image/video
    if self.eval_loss:
      avg_loss, n_batches = 0, 0

    pred_sents, ref_sents = {}, {}
    # load dataset once
    for batch_data in val_reader:
      if self.eval_loss:
        loss = self.forward_loss(batch_data)
        avg_loss += loss.data.item()
        n_batches += 1
      pred_sent = self.validate_batch(batch_data)
      pred_sent = pred_sent.data.cpu().numpy()
      for i, name in enumerate(batch_data['names']):
        pred_sents[name] = [self.int2sent(pred_sent[i])]
        ref_sents[name] = batch_data['ref_sents'][name]

    if self.eval_loss:
      avg_loss /= n_batches
     
    # compute translation score (bleu, rouge)
    metrics = collections.OrderedDict()
    if self.eval_loss:
      metrics['loss'] = avg_loss
    for measure, scorer in self.scorers.items():
      score, _ = scorer.compute_score(ref_sents, pred_sents)
      if measure == 'bleu4':
        score = score[-1] 
        # bleu4 is the "mean" of 1-4 gram (np.exp(np.mean(np.log(actual_scores))))
        # which is the same as nltk.translate.bleu_score.corpus_bleu()
      metrics[measure] = score * 100
    return metrics

  def test(self, tst_reader, tst_pred_file, tst_model_file=None, outcap_format=0):
    if tst_model_file is not None:
      self.load_checkpoint(tst_model_file)
    self.eval_start()

    pred_sents = {}
    for batch_data in tst_reader:    
      greedy_or_beam = self.config.subcfgs[DECODER].greedy_or_beam
      pred_sent, sent_pool = self.test_batch(batch_data, greedy_or_beam)
      for i, name in enumerate(batch_data['names']):
        if isinstance(name, tuple):
          name = '_'.join([str(x) for x in name])
        pred_sents[name] = self.gen_out_caption_format(
          sent_pool[i], self.int2sent, outcap_format)

    output_dir = os.path.dirname(tst_pred_file)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    with open(tst_pred_file, 'w') as f:
      json.dump(pred_sents, f, indent=2)

  def gen_out_caption_format(self, sent_pool, int2sent, format=0):
    '''
    Args:
      sent_pool: list, [[topk_prob, topk_sent, word_probs], ...], sorted
      format: 
        0: [top1_sent]
        1: [top1_sent, prob, word_probs]
        2: [[topk_sent, prob], ...]
        3: [[topk_sent, prob, word_probs], ...]
    '''
    if format == 0:
      return [int2sent(sent_pool[0][1])]
    elif format == 1:
      sent = int2sent(sent_pool[0][1])
      return [sent, sent_pool[0][0].item(), [p.item() for p in sent_pool[0][2]]]
    elif format == 2:
      outs = []
      for item in sent_pool:
        if len(item) == 3:
          sent_prob, sent_ids, word_probs = item
        outs.append([int2sent(sent_ids), sent_prob.item()])
      return outs
    elif format == 3:
      outs = []
      for item in sent_pool:
        if len(item) == 3:
          sent_prob, sent_ids, word_probs = item
        outs.append([int2sent(sent_ids), sent_prob.item(), [p.item() for p in word_probs]])
      return outs

  def epoch_postprocess(self, epoch):
    super().epoch_postprocess(epoch)

    if DECODER in self.config.subcfgs:
      dec_cfg = self.config.subcfgs[DECODER]
      if dec_cfg.schedule_sampling and dec_cfg.ss_rate < dec_cfg.ss_max_rate:
        if (epoch+1) % dec_cfg.ss_increase_epoch == 0:
          dec_cfg.ss_rate = dec_cfg.ss_rate + dec_cfg.ss_increase_rate
          self.print_fn('schedule sampling rate %.4f'%(dec_cfg.ss_rate))

  ################################ DIY ################################
  def validate_batch(self, batch_data, addition_outs=None):
    '''
    Returns:
      pred_sent: list of int_sent
    '''
    raise NotImplementedError('implement validate_batch function')
    
  def test_batch(self, batch_data, greedy_or_beam):
    '''
    Returns:
      pred_sent: list of int_sent
      sent_pool:
    '''
    raise NotImplementedError

  def prepare_input_batch(self, batch_data, is_train=False):
    '''
    Return: dict of tensors
    '''
    raise NotImplementedError

  def forward_encoder(self, input_batch):
    '''
    Return: dict of encoder outputs
    '''
    raise NotImplementedError



