# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import BertModel

from utils import BaseClassifier, BaseEstimator


class LabelAwareModel(nn.Module): 
    def __init__(self, label_ids): 
        super().__init__()
        self.backbone = BertModel.from_pretrained('bert-base-cased')
        
        label_embds = nn.Parameter(
            self.backbone.embeddings.word_embeddings.weight[label_ids], 
            requires_grad=True
        )
        self.register_parameter('label_embds', label_embds)

        d_model = self.backbone.config.hidden_size
        W = nn.Parameter(
            torch.empty(label_embds.shape[-1], d_model), 
            requires_grad=True
        )
        nn.init.trunc_normal_(W.data, std=0.02)
        self.register_parameter('W', W)

        self.head = nn.Linear(2 * d_model, 1)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, input_mask, seg_ids): 
        encoded = self.backbone(
            input_ids=input_ids, 
            attention_mask=input_mask, 
            token_type_ids=seg_ids, 
        )
        hidden_states = encoded[0] # last_hidden_states, (B, l_seq, d_model)
        attn_scores = torch.einsum(
            'ne, ed, bld -> bnl', self.label_embds, self.W, hidden_states
        ) # (B, n_labels, l_seq)
        attn_weights = self.softmax(self.dropout(attn_scores))
        hidden_states = torch.einsum(
            'bnl, bld -> bnd', attn_weights, hidden_states
        ) # (B, n_labels, d_model)
        sentence_embd = encoded[1] # 'pooler_output', (B, d_model)
        hidden_states = torch.cat(
            (hidden_states, sentence_embd.unsqueeze(1).tile((1, len(self.label_embds), 1))), 
            dim=-1
        ) # (B, n_labels, 2 * d_model)
        logits = self.head(self.dropout(hidden_states)).squeeze(-1) # (B, n_labels)
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
