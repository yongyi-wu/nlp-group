# -*- coding: utf-8 -*-

import torch
from torch import nn
<<<<<<< HEAD
from transformers import BertModel, BertTokenizer

from utils import BaseClassifier, BaseEstimator



# class LeamBERTModel(BaseClassifier):
#     def __init__(self, bert_model, head_model, n_labels, radius):
#         super().__init__(
#             bert_model,
#             head_model
#         )
#         self.n_labels = n_labels
#         self.dropout_1 = nn.Dropout(p=0.2)
#         self.radius = radius
#         # model init
#         nn.init.trunc_normal_(self.head.weight, std=0.02)
#         nn.init.zeros_(self.head.bias)

#     def forward(self, input_ids, input_mask, seg_ids): 
#         encoded = self.backbone(
#             input_ids=input_ids, 
#             attention_mask=input_mask, 
#             token_type_ids=seg_ids, 
#         )
#         encoded = encoded[0] # tensor of size bs * (len(input_ids)+2) * 768
#         encoded = self.droupout_1(encoded)
#         logits = self.head(encoded, self.label_embed, self.radius)
#         return logits

def emotion_label_embeddings(emotions, bert_model, tokenizer, non_trainable=False):
    label_ids = tokenizer.convert_tokens_to_ids(emotions)
    label_embed = bert_model.embeddings.word_embeddings.weight[label_ids]
    label_embed = nn.Parameter(
        label_embed, 
        requires_grad=True
    )
    return label_embed

class LeamBERTModel(nn.Module):
    def __init__(self, bert_model, tokenizer, emotions, radius):
        super().__init__()
        self.bert_model = bert_model
        
        self.emotions = emotions
        self.n_labels = len(emotions)

        self.radius = radius
        label_embed = emotion_label_embeddings(emotions, bert_model, tokenizer)
        self.register_parameter('label_embed', label_embed)

        # all the layers used in LEAM_model P
        self.dropout = nn.Dropout(p=0.2) # word_embedding dropout
        self.conv_layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2*radius+1, padding=radius, stride=1)
        self.relu_layer = nn.ReLU(inplace=False)
        self.maxpool_layer = nn.MaxPool1d(self.n_labels) 
        self.linear_layer = nn.Linear(768, self.n_labels)
        
        # model head init
        # nn.init.trunc_normal_(self.head.weight, std=0.02)
        # nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, input_mask, seg_ids):
        # go thru bert_model
        encoded = self.bert_model(
            input_ids=input_ids, 
            attention_mask=input_mask, 
            token_type_ids=seg_ids, 
        )
        encoded = encoded[0] # 'take full embeddings: bs * (len(input_ids)+2) * 768 ' 
        encoded = self.dropout(encoded)

        # preparatory
        label_embed = self.label_embed
        radius = self.radius

        # similarity matrix G: bs * n_labels * n_words
        G = label_embed @ torch.transpose(encoded, 1,2) # bs * n_labels * n_words
        G_norm = torch.linalg.norm(label_embed, dim=-1, keepdim=True) @ torch.transpose(torch.linalg.norm(encoded, dim=-1, keepdim=True), 1, 2) # n_labels * n_words
        G /= G_norm

        # attention part: use conv1d
        # conv1d + relu
        n_words = list(encoded.size())[-2]
    
        conv_weight_full = []
        for i in range(self.n_labels):
            single_label_input = G[:, i:i+1, :] # bs * 1 * n_words
            conv_weight = self.conv_layer(single_label_input) # bs * 1 * n_words
            conv_weight_full.append(conv_weight)
        conv_weight_full = torch.cat(conv_weight_full, dim=1) # bs * n_labels * n_words

        res = self.relu_layer(conv_weight_full) # bs * n_labels * n_words

        # maxpool + softmax: bs * n_words
        attention_weight = self.maxpool_layer(torch.transpose(res, 1, 2)) # bs * n_words

        attention_weight = torch.softmax(attention_weight, dim=-1) # bs * n_words
        attention_weight = attention_weight.view(-1, n_words, 1) # bs * n_words * 1
        # apply attention_weight to input embeddings
        # output dim: bs * 768
        z = torch.sum(attention_weight * encoded, dim=1) # inner dim: bs * n_words * 768
        
        # final_transform
        return self.linear_layer(z) # bs * n_labels
=======
from transformers import BertModel

from utils import BaseClassifier, BaseEstimator
## added
from transformers import BertTokenizer, BertModel



class LabelAwareModel(nn.Module): 
    def __init__(self, label_ids): 
        super().__init__()
        self.backbone = BertModel.from_pretrained('bert-base-cased')
        
        label_embds = nn.Parameter(
            self.backbone.embeddings.word_embeddings.weight[label_ids].clone(), 
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
>>>>>>> 181557d7e18a777702d5b57644c0adb2651ea491

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
<<<<<<< HEAD
=======


## similar to LAA, just without softmax/normalization. 
class Naive_Label_Model(BaseClassifier): 
    """BERT baseline as implemented in the GoEmotions paper"""
    def __init__(self, n_labels, label_id, device): 
        super().__init__(
            BertModel.from_pretrained('bert-base-cased'), 
            nn.Linear(768, n_labels)
        )
        self.cos = nn.CosineSimilarity(dim = 1)
        self.lin1 = nn.Linear(768, 768)
        self.lin2 = nn.Linear(n_labels, n_labels) #added 
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        self.n_labels = n_labels
        self.label_id = label_id

        ## calculate label embedding
        self.device = device
            
    
    def forward(self,  input_ids, input_mask, seg_ids):
        encoded = self.backbone(
            input_ids=input_ids, 
            attention_mask=input_mask, 
            token_type_ids=seg_ids, 
            output_hidden_states=True,
        )
        label_emb = self.backbone.embeddings.word_embeddings.weight[self.label_id]
        sentence_embd = encoded[1] # 'pooler_output' 
        word_embd = encoded[0]

        logits = torch.Tensor(sentence_embd.shape[0], self.n_labels).to(self.device)
        #sentence_embd = self.lin1(sentence_embd)
        #for i in range(word_embd.shape[0]):
            #logits[i] = torch.mean(torch.matmul(word_embd[i], label_emb.T), dim=0)
        #logits = self.lin2(logits)
        logits = torch.mean(torch.matmul(word_embd, label_emb.T), dim=1)
        #logits = torch.matmul(word_embd, label_emb.T)
        #logits = torch.nn.functional.softmax(logits, dim=2)
        #logits = torch.mean(logits, dim=1) 
        #logits = self.lin2(logits)
        return logits


class Naive_Label_Estimator(BaseEstimator): 
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
            #probs = torch.softmax(logits, dim=1).to(self.device)
            #con_loss = torch.sum(torch.var(probs, dim = 1, unbiased=False), dim=-1).to(self.device)
            #loss -= 10 * con_loss
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


## punishes the variance of the output probabilities. Does not perform well
class var_loss_Model(BaseClassifier): 
    """BERT baseline as implemented in the GoEmotions paper"""
    def __init__(self, n_labels): 
        super().__init__(
            BertModel.from_pretrained('bert-base-cased'), 
            nn.Linear(768, n_labels)
        )
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

class var_loss_Estimator(BaseEstimator): 
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
                probs = torch.sigmoid(logits).to(self.device)
                var_loss = torch.sum(torch.var(probs, dim = 1, unbiased=False), dim=-1).to(self.device)
                loss -= 0.2 * var_loss
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
>>>>>>> 181557d7e18a777702d5b57644c0adb2651ea491
