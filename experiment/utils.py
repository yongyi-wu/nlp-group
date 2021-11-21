# -*- coding: utf-8 -*-

import os
import sys
import random
import logging
from datetime import datetime

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score


def seed_everything(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_if_not_exists(new_dir): 
    if not os.path.exists(new_dir): 
        os.system('mkdir -p {}'.format(new_dir))


def config_logging(log_dir): 
    make_if_not_exists(log_dir)
    file_handler = logging.FileHandler(
        filename=os.path.join(log_dir, 'goemotions.log')
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s\t%(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S', 
        handlers=[file_handler, stdout_handler]
    )
    logger = logging.getLogger('train')
    return logger


def convert_to_unicode(text): 
    """Converts text to Unicode (if it's not already), assuming utf-8 input"""
    if isinstance(text, str): 
        return text
    elif isinstance(text, bytes): 
        return text.decode("utf-8", "ignore")
    else: 
        raise ValueError("Unsupported string type: {}".format(type(text)))


class BaseEstimator(object): 
    """A wrapper class to perform training, evluation or testing while accumulating and logging results"""
    def __init__(
        self, 
        model, 
        criterion, 
        optim=None, 
        scheduler=None, 
        logger=None, 
        writer=None, 
        device='cpu'
    ): 
        self.model = model
        self.criterion = criterion
        self.optim = optim
        self.scheduler = scheduler
        self.logger = logger
        self.writer = writer
        self.device = device

        self.epochs = 0
        self.train_step = 0
        self.eval_step = 0

    def step(self, data, optim=None, scheduler=None): 
        """
        This function is responsible for feeding the data into the model, performing backpropagation 
        and, if training, update optimizer and scheduler. 

        Input
        ----------
        data: 
            A dictionary of mini-batched input obtained from Dataset.__getitem__, each with shape (B, max_len)
        optim: 
            Optimizer; if None, perform prediciton only

        Output
        ----------
        loss: 
            A scalar for the entire batch
        yhat: 
            Model predictions as the probability for each label, shape (B, n_labels)
        y: 
            None or an np.ndarray ground true label for the batch, shape (B, n_labels)
        """
        raise NotImplementedError('Implement it in the child class!')

    def _train_epoch(self, trainloader, evalloader=None): 
        self.model.train()
        tbar = tqdm(trainloader, dynamic_ncols=True)
        for data in tbar: 
            loss, yhat, y = self.step(data, self.optim, self.scheduler)
            self.train_step += 1
            tbar.set_description('train_loss - {:.4f}'.format(loss))
            if self.writer is not None: 
                self.writer.add_scalar('train/loss', loss, self.train_step)
                self.writer.add_scalar('train/auc', roc_auc_score(y, yhat), self.train_step)
        if evalloader is not None: 
            return self.eval(evalloader)
        else: 
            return -1, -1

    def train(self, cfg, trainloader, evalloader=None): 
        assert self.optim is not None, 'Optimizer is required'
        assert hasattr(cfg, 'output_dir'), 'Output directory must be specified'
        make_if_not_exists(cfg.output_dir)
        for _ in range(cfg.epochs): 
            loss, auc = self._train_epoch(trainloader, evalloader)
            self.epochs += 1
            checkpoint_path = os.path.join(cfg.output_dir, '{}.pt'.format(datetime.now().strftime('%m-%d_%H-%M')))
            self.save(checkpoint_path)
            if self.logger is not None: 
                self.logger.info('[EPOCH {}]\tdev_loss - {:.4f} dev_auc - {:.4f}'.format(self.epochs, loss, auc))
                self.logger.info('[CHECKPOINT] {}'.format(checkpoint_path))

    def eval(self, evalloader, phase='eval'): 
        self.eval_step += 1
        self.model.eval()
        tbar = tqdm(evalloader, dynamic_ncols=True)
        val_loss = []
        ys = []
        yhats = []
        for data in tbar: 
            loss, out, y = self.step(data)
            tbar.set_description('{}_loss - {:.4f}'.format(phase, loss))
            val_loss.append(loss)
            ys.append(y)
            yhats.append(out)
        ys = np.concatenate(ys, dim=0)
        yhats = np.concatenate(yhats, dim=0)
        loss = np.mean(val_loss).item()
        auc = roc_auc_score(ys, yhats)
        if self.writer is not None: 
            self.writer.add_scalar('{}/loss'.format(phase), loss, self.eval_step)
            self.writer.add_scalar('{}/auc'.format(phase), auc, self.eval_step)
        return yhats, ys

    def test(self, testloader): 
        self.eval_step = 0
        return self.eval(testloader, phase='test')

    def save(self, checkpoint_path): 
        checkpoint = {
            'epochs': self.epochs, 
            'train_step': self.train_step, 
            'eval_step': self.eval_step, 
            'model': self.model.state_dict(), 
            'optimizer': self.optim.state_dict(), 
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

    def load(self, checkpoint_path): 
        print('Loading checkpoint {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.epochs = checkpoint['epochs']
        self.train_step = checkpoint['train_step']
        self.eval_step = checkpoint['eval_step']
        self.model.load_state_dict(checkpoint['model'])
        if self.optim is not None: 
            self.optim.load_state_dict(checkpoint['optimizer'])
        else: 
            print('Optimizer is not loaded')
        if self.scheduler is not None: 
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else: 
            print('Scheduler is not loaded')


class BaseClassifier(nn.Module): 
    """A generic class that combines a representation learner and a classifier"""
    def __init__(self, backbone, head): 
        """
        Input
        ----------
        backbone: 
            Usually a (pretrained) BertModel, or any sentence embedding extractor
        head: 
            Usually a FFN, or any NN for classification
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, input_mask, seg_ids): 
        """
        Input
        ----------
        input_ids: 
            Raw text -> tokens -> ids, starting with id of [CLS] and ending with id of [SEP] (or padding id 0), shape (B, max_len)
        input_mask: 
            1 for tokens that are not padded and hence not masked and 0 otherwise, shape (B, max_len)
        set_ids: 
            Segment identifiers, but in this case 0 for all tokens, shape (B, max_len)

        Output
        ----------
        logits: 
            Logit of being true for each label, shape (B, n_labels)
        """
        encoded = self.backbone(
            input_ids=input_ids, 
            attention_mask=input_mask, 
            token_type_ids=seg_ids
        )
        sentence_embd = encoded['pooler_output']
        logits = self.head(sentence_embd)
        return logits
