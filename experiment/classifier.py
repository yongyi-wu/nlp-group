# -*- coding: utf-8 -*-

import os
import json
import argparse
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import BertTokenizer
import pandas as pd

from data import GoEmotionsDataset
from models import BaselineModel, BaselineEstimator, LabelAwareModel
from utils import make_if_not_exists, seed_everything, config_logging


def parse(): 
    parser = argparse.ArgumentParser('GoEmotions multi-label classifier')
    parser.add_argument('exp_name', type=str, help='Experiment name (be specific and distinguishable)')
    parser.add_argument('--data_dir', type=str, default=os.path.join('..', 'goemotions', 'data'), help='Directory with datasets and the emotion txt file')
    parser.add_argument('--output_dir', type=str, default='out', help='Directory to output results')
    run = parser.add_argument_group(title='Run', description='Parameters related to running the script')
    run.add_argument('--max_len', type=int, default=30, help='Maximum sequence length to the Transformers')
    run.add_argument('--checkpoint', type=str, help='Path to a saved checkpoint')
    run.add_argument('--no_train', action='store_true', help='Skip the training phase')
    run.add_argument('--test_file', type=str, default='test.tsv', choices=['dev.tsv', 'test.tsv'], help='Dataset to evaluate')
    run.add_argument('--seed', type=int, default=0, help='Random seed for reproduction')
    train = parser.add_argument_group(title='Training', description='Parameters related to training the model')
    train.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    train.add_argument('--lr', type=float, default=5E-5, help='Initial learning rate')
    train.add_argument('--n_epochs', type=int, default=4, help='Number of training epochs')
    train.add_argument('--warmup_proportion', type=float, default=0.1, help='Proportion of training steps to do linear lr warmup')
    train.add_argument('--pred_thold', type=float, default=0.3, help='Threshold for predicting each emotion')
    cfg = parser.parse_args()
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    return cfg


def main(): 
    cfg = parse()
    seed_everything(cfg.seed)
    make_if_not_exists(cfg.output_dir)
    logger = config_logging(cfg.output_dir)
    time = datetime.now().strftime('%m-%d_%H-%M')
    writer = SummaryWriter(
        os.path.join(cfg.output_dir, time)
    )
    logger.info(json.dumps(vars(cfg)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Preparing data...')
    with open(os.path.join(cfg.data_dir, 'emotions.txt')) as f: 
        emotions = f.read().splitlines()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    if not cfg.no_train: 
        trainset = GoEmotionsDataset(
            os.path.join(cfg.data_dir, 'train.tsv'), 
            tokenizer, 
            len(emotions), 
            cfg.max_len
        )
        trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
        devset = GoEmotionsDataset(
            os.path.join(cfg.data_dir, 'dev.tsv'), 
            tokenizer, 
            len(emotions), 
            cfg.max_len
        )
        devloader = DataLoader(devset, batch_size=cfg.batch_size, num_workers=4)
    if cfg.test_file is not None: 
        testset = GoEmotionsDataset(
            os.path.join(cfg.data_dir, cfg.test_file), 
            tokenizer, 
            len(emotions), 
            cfg.max_len, 
            is_test=True
        )
        testloader = DataLoader(testset, batch_size=cfg.batch_size, num_workers=4)

    print('Preparing the model...')
    # model = BaselineModel(len(emotions)).to(device) # TODO: change to your model!
    label_ids = tokenizer.convert_tokens_to_ids(emotions)
    model = LabelAwareModel(label_ids).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    if cfg.no_train: 
        optimizer = None
        scheduler = None
    else: 
        no_weight_decay = lambda param_name: any(
            no_decay_name in param_name for no_decay_name in ['LayerNorm', 'layer_norm', 'bias']
        )
        optimizer = optim.AdamW(
            [
                {'params': [param for param_name, param in model.named_parameters() if not no_weight_decay(param_name)]}, 
                {'params': [param for param_name, param in model.named_parameters() if no_weight_decay(param_name)], 'weight_decay': 0}
            ], 
            lr=cfg.lr, 
            betas=(0.9, 0.999), 
            eps=1E-6, 
            weight_decay=0.01
        )
        train_steps = cfg.n_epochs * len(trainloader)
        warmup_steps = cfg.warmup_proportion * train_steps
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, 
            lambda global_step: max(
                0, 
                min(global_step / warmup_steps, 1 - (global_step - warmup_steps) / train_steps)
            ) # slanted triangular lr
        )
    estimator = BaselineEstimator( # TODO: change to your estimator!
        model, 
        tokenizer, 
        criterion, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        logger=logger, 
        writer=writer, 
        pred_thold=cfg.pred_thold, 
        device=device, 
        # add other hyperparameters here
    )

    print('Running the model...')
    if not cfg.no_train: 
        logger.info('Training...')
        estimator.train(cfg, trainloader, devloader)
    if cfg.test_file is not None: 
        logger.info('Testing {}...'.format(cfg.test_file))
        probs, _ = estimator.test(testloader)
        assert probs.shape[0] == len(testset) and probs.shape[1] == len(emotions)
        prediction_file = '{}_{}_prediction.tsv'.format(time, cfg.test_file.split('.')[0])
        df = pd.DataFrame(probs, columns=emotions)
        df.to_csv(os.path.join(cfg.output_dir, prediction_file), sep='\t', index=False)
        logger.info('Writing prediction of {} to {}'.format(cfg.test_file, prediction_file))


if __name__ == '__main__': 
    main()
