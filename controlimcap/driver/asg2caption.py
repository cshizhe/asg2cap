import os
import sys
import json
import time
import numpy as np

import torch.utils.data.dataloader as dataloader
import framework.logbase
import framework.run_utils

import caption.models.attention

import controlimcap.readers.imgsgreader as imgsgreader
import controlimcap.models.graphattn
import controlimcap.models.graphflow
import controlimcap.models.graphmemory
import controlimcap.models.flatattn

from controlimcap.models.graphattn import ATTNENCODER

from controlimcap.driver.common import build_parser, evaluate_caption

def main():
  parser = build_parser()
  parser.add_argument('--max_attn_len', type=int, default=10)
  parser.add_argument('--num_workers', type=int, default=0)
  opts = parser.parse_args()

  if opts.mtype == 'node':
    model_cfg = caption.models.attention.AttnModelConfig()
  elif opts.mtype == 'node.role':
    model_cfg = controlimcap.models.flatattn.NodeRoleBUTDAttnModelConfig()
  elif opts.mtype in ['rgcn', 'rgcn.flow', 'rgcn.memory', 'rgcn.flow.memory']:
      model_cfg = controlimcap.models.graphattn.GraphModelConfig()
  model_cfg.load(opts.model_cfg_file)
  max_words_in_sent = model_cfg.subcfgs['decoder'].max_words_in_sent

  path_cfg = framework.run_utils.gen_common_pathcfg(opts.path_cfg_file, is_train=opts.is_train)

  if path_cfg.log_file is not None:
    _logger = framework.logbase.set_logger(path_cfg.log_file, 'trn_%f'%time.time())
  else:
    _logger = None

  if opts.mtype == 'node':
    model_fn = controlimcap.models.flatattn.NodeBUTDAttnModel
  elif opts.mtype == 'node.role':
    model_fn = controlimcap.models.flatattn.NodeRoleBUTDAttnModel
  elif opts.mtype == 'rgcn':
    model_fn = controlimcap.models.graphattn.RoleGraphBUTDAttnModel
    model_cfg.subcfgs[ATTNENCODER].max_attn_len = opts.max_attn_len
  elif opts.mtype == 'rgcn.flow':
    model_fn = controlimcap.models.graphflow.RoleGraphBUTDCFlowAttnModel
    model_cfg.subcfgs[ATTNENCODER].max_attn_len = opts.max_attn_len
  elif opts.mtype == 'rgcn.memory':
    model_fn = controlimcap.models.graphmemory.RoleGraphBUTDMemoryModel
    model_cfg.subcfgs[ATTNENCODER].max_attn_len = opts.max_attn_len
  elif opts.mtype == 'rgcn.flow.memory':
    model_fn = controlimcap.models.graphmemory.RoleGraphBUTDMemoryFlowModel
    model_cfg.subcfgs[ATTNENCODER].max_attn_len = opts.max_attn_len

  _model = model_fn(model_cfg, _logger=_logger, 
    int2word_file=path_cfg.int2word_file, eval_loss=opts.eval_loss)

  if opts.mtype in ['node', 'node.role']:
    reader_fn = imgsgreader.ImageSceneGraphFlatReader
    collate_fn = imgsgreader.flat_collate_fn
  elif opts.mtype in ['rgcn', 'rgcn.memory']:
    reader_fn = imgsgreader.ImageSceneGraphReader
    collate_fn = imgsgreader.sg_sparse_collate_fn
  elif opts.mtype in ['rgcn.flow', 'rgcn.flow.memory']:
    reader_fn = imgsgreader.ImageSceneGraphFlowReader
    collate_fn = imgsgreader.sg_sparse_flow_collate_fn

  if opts.is_train:
    model_cfg.save(os.path.join(path_cfg.log_dir, 'model.cfg'))
    path_cfg.save(os.path.join(path_cfg.log_dir, 'path.cfg'))
    json.dump(vars(opts), open(os.path.join(path_cfg.log_dir, 'opts.cfg'), 'w'), indent=2)

    trn_dataset = reader_fn(path_cfg.name_file['trn'], path_cfg.mp_ft_file['trn'],
      path_cfg.obj_ft_dir['trn'], path_cfg.region_anno_dir['trn'], path_cfg.word2int_file, 
      max_attn_len=opts.max_attn_len, max_words_in_sent=max_words_in_sent, 
      is_train=True, return_label=True, _logger=_logger)
    trn_reader = dataloader.DataLoader(trn_dataset, batch_size=model_cfg.trn_batch_size, 
      shuffle=True, collate_fn=collate_fn, num_workers=opts.num_workers)
    val_dataset = reader_fn(path_cfg.name_file['val'], path_cfg.mp_ft_file['val'],
      path_cfg.obj_ft_dir['val'], path_cfg.region_anno_dir['trn'], path_cfg.word2int_file, 
      max_attn_len=opts.max_attn_len, max_words_in_sent=max_words_in_sent, 
      is_train=False, return_label=True, _logger=_logger)
    val_reader = dataloader.DataLoader(val_dataset, batch_size=model_cfg.tst_batch_size, 
      shuffle=True, collate_fn=collate_fn, num_workers=opts.num_workers)

    _model.train(trn_reader, val_reader, path_cfg.model_dir, path_cfg.log_dir,
      resume_file=opts.resume_file)

  else:
    tst_dataset = reader_fn(path_cfg.name_file[opts.eval_set], path_cfg.mp_ft_file[opts.eval_set],
      path_cfg.obj_ft_dir[opts.eval_set], path_cfg.region_anno_dir[opts.eval_set], 
      path_cfg.word2int_file, max_attn_len=opts.max_attn_len, max_words_in_sent=max_words_in_sent, 
      is_train=False, return_label=False, _logger=None)
    tst_reader = dataloader.DataLoader(tst_dataset, batch_size=model_cfg.tst_batch_size, 
      shuffle=False, collate_fn=collate_fn, num_workers=opts.num_workers)

    tmp_ref_captions = json.load(open(os.path.join(os.path.dirname(path_cfg.region_anno_dir[opts.eval_set]), 'cleaned_%s_region_descriptions.json'%opts.eval_set)))
    ref_captions = {}
    for k, v in tmp_ref_captions.items():
      for rk, rv in v.items():
        ref_captions['%s_%s'%(k, rk)] = [rv['phrase']]

    model_str_scores = []
    if opts.resume_file is None:
      model_files = framework.run_utils.find_best_val_models(path_cfg.log_dir, path_cfg.model_dir)
    else:
      model_files = {'predefined': opts.resume_file}

    for measure_name, model_file in model_files.items():
      set_pred_dir = os.path.join(path_cfg.pred_dir, opts.eval_set)
      if not os.path.exists(set_pred_dir):
        os.makedirs(set_pred_dir)
      tst_pred_file = os.path.join(set_pred_dir, 
        os.path.splitext(os.path.basename(model_file))[0]+'.json')
      
      if not os.path.exists(tst_pred_file):
        _model.test(tst_reader, tst_pred_file, tst_model_file=model_file, 
          outcap_format=opts.outcap_format)
      if not opts.no_evaluate:
        scores = evaluate_caption(
          None, tst_pred_file, ref_caps=ref_captions,
          outcap_format=opts.outcap_format)
        str_scores = [measure_name, os.path.basename(model_file)]
        for score_name in ['num_words', 'bleu4', 'meteor', 'rouge', 'cider', 'spice', 'avg_lens']:
          str_scores.append('%.2f'%(scores[score_name]))
        str_scores = ','.join(str_scores)
        print(str_scores)
        model_str_scores.append(str_scores)

    if not opts.no_evaluate:
      score_log_file = os.path.join(path_cfg.pred_dir, opts.eval_set, 'scores.csv')
      with open(score_log_file, 'a') as f:
        for str_scores in model_str_scores:
          print(str_scores, file=f)


if __name__ == '__main__':
  main()
