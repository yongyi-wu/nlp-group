# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import BertModel, BertTokenizer

from utils import BaseClassifier, BaseEstimator



class LeamBERTModel(BaseClassifier):
    def __init__(self, bert_model, head_model, n_labels, radius):
        super().__init__(
            bert_model,
            head_model
        )
        self.n_labels = n_labels
        self.dropout_1 = nn.Dropout(p=0.2)
        self.radius = radius
        # model init
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, input_mask, seg_ids): 
        encoded = self.backbone(
            input_ids=input_ids, 
            attention_mask=input_mask, 
            token_type_ids=seg_ids, 
        )
        encoded = encoded[0] # tensor of size bs * (len(input_ids)+2) * 768
        encoded = self.droupout_1(encoded)
        logits = self.head(encoded, self.label_embed, self.radius)
        return logits





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
    def step(self, data): 
        logits = self.model(
            data['input_ids'].to(self.device, dtype=torch.long), 
            data['input_mask'].to(self.device, dtype=torch.long), 
            data['seg_ids'].to(self.device, dtype=torch.long)
        )
        if self.mode in {'train', 'dev'}: 
            # training or developmenting, ground true labels are provided
            if self.mode == 'train': 
                self.optimizer.zero_grad()
            loss = self.criterion(logits, data['labels'].to(self.device, dtype=torch.float))
            if self.mode == 'train': 
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1) # TODO: not equivalent to tf.clip_by_global_norm
                self.optimizer.step()
                if self.scheduler is not None: 
                    self.scheduler.step()
            return (
                loss.detach().cpu().item(), 
                torch.sigmoid(logits).detach().cpu().numpy(), 
                data['labels'].numpy()
            )
        elif self.mode == 'test': 
            # testing, no ground true label is provided
            return None, torch.sigmoid(logits).detach().cpu().numpy(), None
        else: 
            raise ValueError(self.mode)
