# -*- coding: utf-8 -*-
"""Train with optional Global Distance, Local Distance, Identification Loss."""
from __future__ import print_function

import torch
from torch.autograd import Variable
from torch.nn.parallel import DataParallel

import numpy as np
from aligned_reid.model.Model import Model
from aligned_reid.utils.utils import load_state_dict
from aligned_reid.utils.utils import set_devices


class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    print(len(ims))
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()
    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    global_feat, local_feat = self.model(ims)[:2]
    global_feat = global_feat.data.cpu().numpy()
    local_feat = local_feat.data.cpu().numpy()
    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return global_feat, local_feat

model = Model(local_conv_out_channels= 128, num_classes=751 )
# Model wrapper
model_w = DataParallel(model)

weight = './model_weight.pth'
map_location = (lambda storage, loc: storage)
sd = torch.load(weight, map_location=map_location)
load_state_dict(model, sd)
print('Loaded model weights from {}'.format(weight))
TVT, TMO = set_devices((0,))

FeatureExtractor = ExtractFeature(model_w, TVT)
global_feat, local_feat = FeatureExtractor( XXX )



