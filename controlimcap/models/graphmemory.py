import torch
import torch.nn as nn

import caption.encoders.vanilla
import caption.models.captionbase

import controlimcap.encoders.gcn
import controlimcap.decoders.memory
import controlimcap.models.graphattn
import controlimcap.models.graphflow

MPENCODER = 'mp_encoder'
ATTNENCODER = 'attn_encoder'
DECODER = 'decoder'


class GraphBUTDMemoryModel(controlimcap.models.graphattn.GraphBUTDAttnModel):
  def build_submods(self):
    submods = {}
    submods[MPENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[MPENCODER])
    submods[ATTNENCODER] =controlimcap.encoders.gcn.RGCNEncoder(self.config.subcfgs[ATTNENCODER])
    submods[DECODER] = controlimcap.decoders.memory.MemoryDecoder(self.config.subcfgs[DECODER])
    return submods

class RoleGraphBUTDMemoryModel(controlimcap.models.graphattn.RoleGraphBUTDAttnModel):
  def build_submods(self):
    submods = {}
    submods[MPENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[MPENCODER])
    submods[ATTNENCODER] =controlimcap.encoders.gcn.RoleRGCNEncoder(self.config.subcfgs[ATTNENCODER])
    submods[DECODER] = controlimcap.decoders.memory.MemoryDecoder(self.config.subcfgs[DECODER])
    return submods


class GraphBUTDMemoryFlowModel(controlimcap.models.graphflow.GraphBUTDCFlowAttnModel):
  def build_submods(self):
    submods = {}
    submods[MPENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[MPENCODER])
    submods[ATTNENCODER] = controlimcap.encoders.gcn.RGCNEncoder(self.config.subcfgs[ATTNENCODER])
    submods[DECODER] = controlimcap.decoders.memory.MemoryFlowDecoder(self.config.subcfgs[DECODER])
    return submods

class RoleGraphBUTDMemoryFlowModel(controlimcap.models.graphflow.RoleGraphBUTDCFlowAttnModel):
  def build_submods(self):
    submods = {}
    submods[MPENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[MPENCODER])
    submods[ATTNENCODER] = controlimcap.encoders.gcn.RoleRGCNEncoder(self.config.subcfgs[ATTNENCODER])
    submods[DECODER] = controlimcap.decoders.memory.MemoryFlowDecoder(self.config.subcfgs[DECODER])
    return submods


