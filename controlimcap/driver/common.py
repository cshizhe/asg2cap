import os
import json
import argparse
import numpy as np

from eval_cap.bleu.bleu import Bleu
from eval_cap.meteor.meteor import Meteor
from eval_cap.cider.cider import Cider
from eval_cap.rouge.rouge import Rouge
from eval_cap.spice.spice import Spice


def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_cfg_file')
  parser.add_argument('path_cfg_file')
  parser.add_argument('mtype')
  parser.add_argument('--resume_file', default=None)
  parser.add_argument('--selfcritic', action='store_true', default=False)
  parser.add_argument('--eval_loss', action='store_true', default=False)
  parser.add_argument('--is_train', action='store_true', default=False)

  parser.add_argument('--eval_set', default='tst')
  parser.add_argument('--no_evaluate', action='store_true', default=False)
  parser.add_argument('--outcap_format', type=int, default=0)

  return parser


def evaluate_caption(ref_cap_file, pred_cap_file, ref_caps=None, 
  preds=None, scorer_names=None, outcap_format=0):
  if ref_caps is None:
    ref_caps = json.load(open(ref_cap_file))
  if preds is None:
    preds = json.load(open(pred_cap_file))

  if outcap_format == 1:
    outs = {}
    for key, value in preds.items():
      outs[key] = [value[0]]
    preds = outs
  elif outcap_format in [2, 3, 4]:
    outs = {}
    for key, value in preds.items():
      outs[key] = [value[0][0]]
    preds = outs

  refs = {}
  for key in preds.keys():
    refs[key] = ref_caps[key] 

  scorers = {
    'bleu4': Bleu(4),
    'meteor': Meteor(),
    'rouge': Rouge(),
    'cider': Cider(),
    'spice': Spice(),
  }
  if scorer_names is None:
    scorer_names = list(scorers.keys())

  scores = {}
  for measure_name in scorer_names:
    scorer = scorers[measure_name]
    s, _ = scorer.compute_score(refs, preds)
    if measure_name == 'bleu4':
      scores[measure_name] = s[-1] * 100
    else:
      scores[measure_name] = s * 100

  scorers['meteor'].meteor_p.kill()
  unique_words = set()
  sent_lens = []
  for key, value in preds.items():
    for sent in value:
      unique_words.update(sent.split())
      sent_lens.append(len(sent.split()))
  scores['num_words'] = len(unique_words)
  scores['avg_lens'] = np.mean(sent_lens)
  return scores



