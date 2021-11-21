# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import BertModel

from utils import BaseClassifier, BaseEstimator


class BaselineModel(BaseClassifier): 
    """BERT baseline as implemented in the GoEmotions paper"""
    def __init__(self, n_labels): 
        super().__init__(
            BertModel.from_pretrained('bert-base-cased'), 
            nn.Linear(768, n_labels)
        )
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)


class BaselineEstimator(BaseEstimator): 
    def step(self, data, optim=None, scheduler=None): 
        for key, item in data.items(): 
            data[key] = item.to(self.device)
        if optim is not None: 
            optim.zero_grad()
        logits = self.model(
            data['input_ids'], 
            data['input_mask'], 
            data['seg_ids']
        )
        loss = self.criterion(logits, data['labels'])
        if optim is not None: 
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1) # TODO: not equivalent to tf.clip_by_global_norm
            optim.step()
            assert scheduler is not None, 'Scheduler is required'
            scheduler.step()
        return (
            loss.detach().cpu().item(), 
            torch.sigmoid(logits).detach().cpu(), 
            data['labels'].detach().cpu()
        )